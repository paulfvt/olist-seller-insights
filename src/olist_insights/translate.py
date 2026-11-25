"""Traduction PT -> EN des avis clients, en local, avec un cache persistant.

Vue d'ensemble
--------------
1. On lit les textes d'avis (silver.order_reviews.review_pt).
2. Pour chaque texte, on calcule une empreinte (hash SHA-256). La clé du cache est
   (empreinte, nom du modèle) : un texte déjà traduit par ce modèle ne repart jamais
   au modèle, même si on relance le script ou si on reconstruit la base principale.
3. Les textes manquants sont traduits par lots (batch) par un modèle MarianMT
   d'Helsinki-NLP, et chaque lot est enregistré dès qu'il est fini : si le script est
   interrompu, on reprend là où on s'était arrêté.
4. On construit silver.review_translations (review_id, review_pt, review_en) par
   jointure entre les avis et le cache.

Pourquoi un modèle local plutôt qu'une API ?
-------------------------------------------
Gratuit et sans quota, données qui ne sortent pas de la machine, résultat stable.
Et on ne veut pas que le LLM qu'on évalue plus tard soit aussi celui qui a traduit
ses propres entrées.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import sys
import time
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from olist_insights.config import load_settings
from olist_insights.logging_utils import setup_logging

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Helsinki-NLP/opus-mt-ROMANCE-en"
CACHE_FILENAME = "translations.duckdb"

# Un "traducteur" est simplement une fonction : liste de textes PT -> liste de textes EN
# (même longueur, même ordre). Le code du cache ne dépend que de ce contrat, ce qui
# permet de le tester avec un faux traducteur instantané, sans télécharger de modèle.
Translator = Callable[[list[str]], list[str]]


def text_hash(text: str) -> str:
    """Empreinte SHA-256 du texte (identique à la fonction sha256() de DuckDB)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def has_letters(text: str) -> bool:
    """Vrai si le texte contient au moins une lettre.

    Un texte sans lettre ("👍", "...", "10") n'a rien à traduire. Pire, un modèle de
    traduction face à une entrée qu'il ne comprend pas a tendance à « halluciner » une
    boucle ("No, no, no, no..."). Ces textes sont donc recopiés tels quels, sans modèle.
    Équivalent SQL utilisé plus bas : regexp_matches(texte, '\\pL').
    """
    return any(ch.isalpha() for ch in text)


# Une boucle = un même groupe de 1 à 3 mots répété au moins 4 fois d'affilée
# ("no, no, no, no", "all right, all right, all right, all right"),
# ou une suite de ponctuation isolée (". . . . . .", repérée à l'étape 6).
_LOOP = re.compile(r"\b(\w+(?:\W+\w+){0,2}?)(?:\W+\1\b){3,}", re.IGNORECASE)
_PUNCT_LOOP = re.compile(r"(?:[.•·,]\s+){6,}")

# Découpage en phrases : après . ! ? suivis d'un espace, ou à chaque retour à la ligne.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")

# Omission probable (défaut repéré à l'étape 6) : un texte de plusieurs phrases dont la
# traduction fait moins de 60 % de la longueur. MarianMT a tendance à ne traduire que la
# première phrase d'une entrée qui en contient plusieurs.
MIN_LEN_FOR_RATIO = 30
OMISSION_RATIO = 0.6


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]


def join_pieces(pieces: list[str]) -> str:
    """Recolle des phrases traduites séparément.

    Une phrase source sans ponctuation finale (« produto bom » sur sa propre ligne) donne
    une traduction sans point, qui se collerait à la suivante (« good product great
    site »). On ajoute un point aux morceaux qui n'en ont pas, sauf au dernier : la
    lecture y gagne, et le nombre de phrases reste celui de la source (sinon le contrôle
    d'omission signalerait ces textes à chaque passage, sans fin).
    """
    fixed = [p if i == len(pieces) - 1 or p.rstrip()[-1:] in ".!?…" else p.rstrip() + "."
             for i, p in enumerate(pieces)]
    return " ".join(fixed)


def _is_shouting(sentence: str) -> bool:
    """Phrase surtout en MAJUSCULES, même règle que normalize_case (sinon le tri des textes
    à réparer et la réparation ne seraient pas d'accord, et on retraduirait sans fin)."""
    return is_mostly_upper(sentence)


def is_untreated_caps(source: str, translation: str) -> bool:
    """Source avec une phrase en MAJUSCULES ET traduction encore en majuscules : traduite
    avant la mise en casse de phrase (étape 6). Le critère porte aussi sur la TRADUCTION
    pour rester idempotent : une fois retraduit, le texte sort en casse normale et n'est
    plus sélectionné."""
    return (any(_is_shouting(s) for s in split_sentences(source))
            and any(_is_shouting(s) for s in split_sentences(translation)))


def count_sentences(text: str) -> int:
    return sum(1 for s in split_sentences(text) if has_letters(s))


def is_probable_omission(source: str, translation: str) -> bool:
    """Texte de plusieurs phrases dont la traduction en a moins, ou est beaucoup trop courte.

    Le critère du nombre de phrases a été ajouté après la relecture humaine de l'étape 6 :
    une phrase courte perdue (« Ainda estou no aguardo. ») ne fait pas assez baisser la
    longueur pour être vue par le seul critère du rapport de longueur. Il signale aussi
    des fusions sans perte (« blanc. Et vert. » -> « blanc, et vert. ») : on accepte ces
    faux positifs, les retraduire ne coûte que du temps.
    """
    n_source = count_sentences(source)
    if n_source < 2:
        return False
    too_short = len(source) >= MIN_LEN_FOR_RATIO and len(translation) / len(source) < OMISSION_RATIO
    return too_short or count_sentences(translation) < n_source


def has_repetition_loop(text: str) -> bool:
    """Détecte une boucle de répétition, symptôme classique d'hallucination.

    Un modèle de génération choisit chaque token en fonction des précédents. S'il
    hésite (texte en MAJUSCULES, fautes de frappe, texte très court), il peut retomber
    sur un motif qu'il vient de produire et ne plus en sortir : la répétition devient
    la suite la plus probable d'elle-même.
    """
    return bool(_LOOP.search(text) or _PUNCT_LOOP.search(text))


def is_mostly_upper(text: str, min_letters: int = 8, share: float = 0.7) -> bool:
    """Au moins 70 % des lettres en majuscules (et assez de lettres pour ne pas être un sigle).

    Pas « 100 % » : les noms de vendeurs anonymisés par Olist sont en minuscules
    (« GOSTO DE COMPRAR NA stark ») et feraient échouer un simple isupper().
    """
    letters = [ch for ch in text if ch.isalpha()]
    return len(letters) >= min_letters and sum(ch.isupper() for ch in letters) / len(letters) >= share


def normalize_case(text: str) -> str:
    """Un texte surtout en majuscules passe en casse de phrase (le modèle a vu peu de
    MAJUSCULES à l'entraînement, il les traduit mal). Les autres textes sont inchangés."""
    return text.capitalize() if is_mostly_upper(text) else text


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class TranslationCache:
    """Cache des traductions dans un fichier DuckDB dédié.

    Fichier séparé de la base principale : son cycle de vie est différent. On peut
    supprimer et reconstruire olist.duckdb sans perdre des heures de traduction.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.con = duckdb.connect(str(path))
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS translations (
                text_hash       VARCHAR NOT NULL,
                model           VARCHAR NOT NULL,
                source_text     VARCHAR NOT NULL,
                translated_text VARCHAR NOT NULL,
                translated_at   TIMESTAMP NOT NULL,
                PRIMARY KEY (text_hash, model)
            )
            """
        )

    def missing(self, hashes: Iterable[str], model: str) -> set[str]:
        """Empreintes absentes du cache pour ce modèle."""
        wanted = set(hashes)
        if not wanted:
            return set()
        known = {
            h
            for (h,) in self.con.execute(
                "SELECT text_hash FROM translations WHERE model = ? AND text_hash IN (SELECT unnest(?))",
                [model, list(wanted)],
            ).fetchall()
        }
        return wanted - known

    def add(self, model: str, sources: list[str], translations: list[str]) -> None:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        rows = [(text_hash(s), model, s, t, now) for s, t in zip(sources, translations, strict=True)]
        # ON CONFLICT DO NOTHING : rejouer un lot déjà enregistré ne crée pas d'erreur
        self.con.executemany(
            "INSERT INTO translations VALUES (?, ?, ?, ?, ?) ON CONFLICT DO NOTHING", rows
        )

    def purge_loops(self, model: str) -> int:
        """Supprime du cache les traductions qui bouclent (alors que la source ne boucle pas).

        Elles seront retraduites au prochain passage, avec le rattrapage de
        load_marian_translator. Retourne le nombre d'entrées supprimées.
        """
        return self._purge(model, lambda src, out: has_repetition_loop(out) and not has_repetition_loop(src))

    def purge_omissions(self, model: str) -> int:
        """Supprime du cache les omissions probables, retraduites ensuite phrase par phrase."""
        return self._purge(model, is_probable_omission)

    def purge_untreated_caps(self, model: str) -> int:
        """Supprime du cache les textes en majuscules traduits sans mise en casse de phrase."""
        return self._purge(model, is_untreated_caps)

    def _purge(self, model: str, is_bad: Callable[[str, str], bool]) -> int:
        rows = self.con.execute(
            "SELECT text_hash, source_text, translated_text FROM translations WHERE model = ?", [model]
        ).fetchall()
        bad = [h for h, src, out in rows if is_bad(src, out)]
        if bad:
            self.con.execute(
                "DELETE FROM translations WHERE model = ? AND text_hash IN (SELECT unnest(?))", [model, bad]
            )
        return len(bad)

    def size(self, model: str) -> int:
        (n,) = self.con.execute("SELECT count(*) FROM translations WHERE model = ?", [model]).fetchone()
        return int(n)

    def close(self) -> None:
        self.con.close()


# ---------------------------------------------------------------------------
# Traduction par lots avec cache
# ---------------------------------------------------------------------------


def translate_texts(
    texts: Iterable[str],
    translator: Translator,
    cache: TranslationCache,
    model: str,
    batch_size: int = 32,
    log_every: int = 20,
) -> int:
    """Traduit les textes absents du cache et les y enregistre. Retourne le nombre traduit.

    - Les doublons sont traduits une seule fois (dédoublonnage par empreinte).
    - Les textes vides ou sans lettre sont ignorés (voir has_letters).
    - Les textes sont triés par longueur avant d'être groupés en lots : dans un lot,
      tous les textes sont complétés (padding) jusqu'à la longueur du plus long.
      Grouper des textes de longueur proche évite de calculer sur du remplissage.
    - Chaque lot est enregistré dès qu'il est traduit (reprise possible après interruption).
    """
    unique = {text_hash(t): t for t in texts if t and has_letters(t)}
    todo_hashes = cache.missing(unique, model)
    todo = sorted((unique[h] for h in todo_hashes), key=len)
    if not todo:
        logger.info("Rien à traduire : %d textes déjà en cache", len(unique))
        return 0

    logger.info("%d textes uniques, %d déjà en cache, %d à traduire",
                len(unique), len(unique) - len(todo), len(todo))
    start = time.perf_counter()
    done = 0
    for i in range(0, len(todo), batch_size):
        batch = todo[i : i + batch_size]
        cache.add(model, batch, translator(batch))
        done += len(batch)
        batch_no = i // batch_size + 1
        if batch_no % log_every == 0 or done == len(todo):
            elapsed = time.perf_counter() - start
            rate = done / elapsed
            eta_min = (len(todo) - done) / rate / 60
            logger.info("%6d/%d traduits | %.1f textes/s | reste ~%.0f min", done, len(todo), rate, eta_min)
    return done


# ---------------------------------------------------------------------------
# Modèle MarianMT (Hugging Face)
# ---------------------------------------------------------------------------


def load_marian_translator(model_name: str = DEFAULT_MODEL, num_beams: int = 4) -> Translator:
    """Charge le modèle et renvoie une fonction de traduction.

    Le premier appel télécharge le modèle (~300 Mo) dans le cache Hugging Face
    (~/.cache/huggingface), les suivants le relisent depuis le disque.
    """
    # Import tardif : torch et transformers sont lourds à importer. Le reste du module
    # (cache, tests) n'en a pas besoin.
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()  # mode inférence : désactive le dropout, résultats déterministes
    # La config du modèle fixe aussi max_length=512 : on le retire pour que notre
    # plafond max_new_tokens soit le seul en jeu (sinon transformers avertit du conflit).
    model.generation_config.max_length = None

    def generate(batch: list[str], **extra) -> list[str]:
        # 1. Tokenisation : texte -> identifiants de sous-mots (sentencepiece).
        #    padding=True complète les textes courts du lot ; truncation coupe au-delà
        #    de 512 tokens (nos avis font au plus ~210 caractères, donc jamais coupés).
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        # 2. Génération : le décodeur produit la phrase anglaise token par token.
        #    num_beams=4 : recherche en faisceau (beam search). On garde les 4 meilleures
        #    hypothèses à chaque pas, plus lent que le choix glouton (1) mais plus fluide.
        #    inference_mode : pas de calcul de gradients, moins de mémoire et plus rapide.
        # 3. Garde-fou de longueur : une traduction fait rarement plus de 2x la longueur
        #    de l'entrée. Sans plafond, une boucle d'hallucination continue jusqu'à 512
        #    tokens et un seul texte ralentit tout son lot. Le plafond est calculé sur le
        #    plus long texte du lot (input_ids.shape[1] = nombre de tokens après padding).
        max_new = 2 * inputs["input_ids"].shape[1] + 10
        with torch.inference_mode():
            outputs = model.generate(**inputs, num_beams=num_beams, max_new_tokens=max_new, **extra)
        # 4. Détokenisation : identifiants -> texte, sans les tokens spéciaux (<pad>, </s>).
        return tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def translate_segments(segments: list[str]) -> list[str]:
        # Phrase en MAJUSCULES -> casse de phrase avant traduction : le modèle a vu très peu
        # de majuscules à l'entraînement et y hallucine (étape 6 : « IN THE CHAIR: MR MARTIN »).
        outputs = generate([normalize_case(s) for s in segments])
        # 5. Rattrapage des boucles d'hallucination (~0,1 % des avis). On ne retraduit que
        #    les textes fautifs (et seulement si le texte source ne boucle pas lui-même),
        #    en casse normale et avec no_repeat_ngram_size=3 : le décodeur a l'interdiction
        #    de produire deux fois la même suite de 3 tokens. Pas appliqué à tout le monde,
        #    car cela empêcherait aussi les répétitions légitimes de l'original.
        bad = [i for i, (src, out) in enumerate(zip(segments, outputs))
               if has_repetition_loop(out) and not has_repetition_loop(src)]
        if bad:
            retried = generate([normalize_case(segments[i]) for i in bad], no_repeat_ngram_size=3)
            for i, out in zip(bad, retried):
                outputs[i] = out
        return outputs

    def translate(batch: list[str]) -> list[str]:
        # 6. Phrase par phrase : avec plusieurs phrases en entrée, MarianMT ne traduit
        #    parfois que la première et ignore le reste (« omission », étape 6). On découpe
        #    chaque texte en phrases, on les traduit toutes dans un seul lot, puis on
        #    recolle. Les morceaux sans lettre (emoji) sont gardés tels quels.
        pieces = [split_sentences(text) or [text] for text in batch]
        to_translate = [p for text_pieces in pieces for p in text_pieces if has_letters(p)]
        translated = iter(translate_segments(to_translate)) if to_translate else iter([])
        return [join_pieces([next(translated) if has_letters(p) else p for p in text_pieces])
                for text_pieces in pieces]

    return translate


# ---------------------------------------------------------------------------
# Intégration au pipeline
# ---------------------------------------------------------------------------


def select_texts(con: duckdb.DuckDBPyConnection, limit: int | None = None) -> list[str]:
    """Textes d'avis distincts à traduire. Ordre par review_id (identifiants aléatoires) :
    un --limit donne un échantillon pseudo-aléatoire mais reproductible."""
    sql = "SELECT review_pt FROM silver.order_reviews WHERE has_text GROUP BY review_id, review_pt ORDER BY review_id"
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    return [t for (t,) in con.execute(sql).fetchall()]


def build_review_translations(con: duckdb.DuckDBPyConnection, cache_path: Path, model: str) -> tuple[int, int]:
    """(Re)construit silver.review_translations : une ligne par avis avec texte.

    review_en vaut NULL si l'avis n'est pas encore traduit. Un texte sans lettre
    (emoji, ponctuation) est recopié tel quel. Retourne (avis, traduits).
    """
    con.execute(f"ATTACH '{cache_path}' AS tcache (READ_ONLY)")
    try:
        con.execute(
            """
            CREATE OR REPLACE TABLE silver.review_translations AS
            SELECT
                r.review_id,
                r.review_pt,
                CASE WHEN regexp_matches(r.review_pt, '\\pL') THEN t.translated_text
                     ELSE r.review_pt END                            AS review_en,
                CASE WHEN regexp_matches(r.review_pt, '\\pL') THEN t.model
                     ELSE 'copied (no letters)' END                  AS translation_model
            FROM (SELECT DISTINCT review_id, review_pt FROM silver.order_reviews WHERE has_text) r
            LEFT JOIN tcache.translations t
                   ON t.text_hash = sha256(r.review_pt) AND t.model = ?
            """,
            [model],
        )
    finally:
        con.execute("DETACH tcache")
    (n, n_en) = con.execute("SELECT count(*), count(review_en) FROM silver.review_translations").fetchone()
    return int(n), int(n_en)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Traduction PT -> EN des avis, avec cache.")
    parser.add_argument("--limit", type=int, default=None, help="nombre d'avis (pilote) ; par défaut tous")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--repair-loops", action="store_true",
                        help="supprime du cache les traductions qui bouclent, pour les retraduire")
    parser.add_argument("--repair-omissions", action="store_true",
                        help="supprime du cache les omissions probables, retraduites phrase par phrase")
    parser.add_argument("--repair-caps", action="store_true",
                        help="supprime du cache les textes en MAJUSCULES traduits sans mise en casse")
    args = parser.parse_args(argv)

    settings = load_settings()
    setup_logging(settings.log_level)
    # Les bibliothèques réseau loguent chaque requête HTTP en INFO : on ne garde que leurs alertes.
    for noisy in ("httpx", "httpx2", "huggingface_hub", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    cache_path = settings.cache_dir / CACHE_FILENAME

    # 1. Lire les textes puis fermer la base : elle reste libre pendant la traduction.
    with duckdb.connect(str(settings.duckdb_path), read_only=True) as con:
        texts = select_texts(con, args.limit)

    # 2. Traduire ce qui manque.
    cache = TranslationCache(cache_path)
    try:
        if args.repair_loops:
            logger.info("Boucles de répétition retirées du cache : %d", cache.purge_loops(args.model))
        if args.repair_omissions:
            logger.info("Omissions probables retirées du cache : %d", cache.purge_omissions(args.model))
        if args.repair_caps:
            logger.info("Textes en majuscules retirés du cache : %d", cache.purge_untreated_caps(args.model))
        if cache.missing((text_hash(t) for t in texts if has_letters(t)), args.model):
            logger.info("Chargement du modèle %s", args.model)
            translator = load_marian_translator(args.model, args.num_beams)
            translate_texts(texts, translator, cache, args.model, args.batch_size)
        else:
            logger.info("Tous les textes demandés sont déjà en cache, modèle non chargé")
        logger.info("Cache : %d traductions pour %s (%s)", cache.size(args.model), args.model, cache_path)
    finally:
        cache.close()

    # 3. Exposer le résultat dans Silver.
    with duckdb.connect(str(settings.duckdb_path)) as con:
        n, n_en = build_review_translations(con, cache_path, args.model)
    logger.info("silver.review_translations : %d avis, %d traduits (%.1f %%)", n, n_en, 100 * n_en / max(n, 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
