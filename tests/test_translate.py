import os

import duckdb
import pytest

from olist_insights.translate import (
    DEFAULT_MODEL,
    TranslationCache,
    build_review_translations,
    has_letters,
    has_repetition_loop,
    normalize_case,
    text_hash,
    translate_texts,
)


class FakeTranslator:
    """Traducteur instantané qui enregistre ce qu'on lui envoie."""

    def __init__(self, fail_on_call: int | None = None):
        self.calls: list[list[str]] = []
        self.fail_on_call = fail_on_call

    def __call__(self, batch: list[str]) -> list[str]:
        self.calls.append(batch)
        if self.fail_on_call is not None and len(self.calls) == self.fail_on_call:
            raise RuntimeError("panne simulée")
        return [f"EN({t})" for t in batch]

    @property
    def sent(self) -> list[str]:
        return [t for batch in self.calls for t in batch]


@pytest.fixture
def cache(tmp_path):
    c = TranslationCache(tmp_path / "cache" / "t.duckdb")
    yield c
    c.close()


def _cached(cache, model="m") -> dict[str, str]:
    return dict(cache.con.execute(
        "SELECT source_text, translated_text FROM translations WHERE model = ?", [model]).fetchall())


def test_hash_matches_duckdb_sha256():
    text = "Não recebi o produto ✓"
    (db_hash,) = duckdb.execute("SELECT sha256(?)", [text]).fetchone()
    assert text_hash(text) == db_hash


def test_translates_each_text_once(cache):
    fake = FakeTranslator()
    n = translate_texts(["ótimo", "ruim", "ótimo", "", "ruim", "👍", "..."], fake, cache, "m")
    assert n == 2
    assert sorted(fake.sent) == ["ruim", "ótimo"]  # doublons, vide et textes sans lettre écartés
    assert _cached(cache) == {"ótimo": "EN(ótimo)", "ruim": "EN(ruim)"}


def test_second_run_hits_cache(cache):
    translate_texts(["a", "b"], FakeTranslator(), cache, "m")
    fake = FakeTranslator()
    assert translate_texts(["a", "b", "c"], fake, cache, "m") == 1
    assert fake.sent == ["c"]


def test_model_is_part_of_the_key(cache):
    translate_texts(["a"], FakeTranslator(), cache, "model-1")
    fake = FakeTranslator()
    assert translate_texts(["a"], fake, cache, "model-2") == 1
    assert cache.size("model-1") == cache.size("model-2") == 1


def test_batches_sorted_by_length(cache):
    fake = FakeTranslator()
    translate_texts(["ccc", "a", "dddd", "bb"], fake, cache, "m", batch_size=2)
    assert fake.calls == [["a", "bb"], ["ccc", "dddd"]]


def test_completed_batches_survive_a_crash(tmp_path):
    path = tmp_path / "t.duckdb"
    cache = TranslationCache(path)
    with pytest.raises(RuntimeError):
        translate_texts(["a", "bb", "ccc"], FakeTranslator(fail_on_call=2), cache, "m", batch_size=1)
    cache.close()
    # reprise : le premier lot est déjà en cache, seuls les deux autres repartent
    cache = TranslationCache(path)
    fake = FakeTranslator()
    assert translate_texts(["a", "bb", "ccc"], fake, cache, "m", batch_size=1) == 2
    assert fake.sent == ["bb", "ccc"]
    cache.close()


def test_build_review_translations(tmp_path):
    cache_path = tmp_path / "t.duckdb"
    cache = TranslationCache(cache_path)
    cache.add(DEFAULT_MODEL, ["Muito bom"], ["Very good"])
    cache.add("autre-modele", ["Ruim"], ["Bad"])
    cache.close()

    with duckdb.connect(":memory:") as con:
        con.execute("CREATE SCHEMA silver")
        con.execute(
            """
            CREATE TABLE silver.order_reviews AS SELECT * FROM (VALUES
                ('r1', 'o1', 'Muito bom', true),
                ('r1', 'o2', 'Muito bom', true),   -- avis partagé par 2 commandes
                ('r2', 'o3', 'Ruim', true),        -- traduit, mais par un autre modèle
                ('r3', 'o4', NULL, false),         -- sans texte
                ('r4', 'o5', '👍👍', true)          -- sans lettre : recopié tel quel
            ) t(review_id, order_id, review_pt, has_text)
            """
        )
        assert build_review_translations(con, cache_path, DEFAULT_MODEL) == (3, 2)
        rows = dict(con.execute("SELECT review_id, review_en FROM silver.review_translations").fetchall())
        assert rows == {"r1": "Very good", "r2": None, "r4": "👍👍"}


@pytest.mark.parametrize("text, expected", [
    ("Não recebi", True), ("ótimo", True), ("👍", False), ("...", False), ("10/10", False),
])
def test_has_letters(text, expected):
    assert has_letters(text) is expected


def test_has_letters_matches_duckdb_regex():
    # Python (isalpha) et SQL (\pL) doivent faire le même tri, sinon un texte serait
    # ignoré côté Python mais attendu traduit côté SQL (ou l'inverse).
    texts = ["Não", "👍", "...", "çá", "10", "ok!", "★★★"]
    sql = [r for (r,) in duckdb.execute(
        r"SELECT regexp_matches(t, '\pL') FROM unnest(?::VARCHAR[]) u(t)", [texts]).fetchall()]
    assert sql == [has_letters(t) for t in texts]


@pytest.mark.parametrize("text, expected", [
    ("No, no, no, no, no, no", True),
    ("All right, all right, all right, all right, thanks.", True),
    ("CUSTOMS CUSTOMS CUSTOMS CUSTOMS", True),
    ("Very, very good product", False),         # répétition légitime courte
    ("No, no, no. I didn't get it.", False),   # 3 fois : sous le seuil
    ("I bought a shower and received another shower.", False),
])
def test_has_repetition_loop(text, expected):
    assert has_repetition_loop(text) is expected


def test_normalize_case():
    assert normalize_case("MUITO BOM MESMO") == "Muito bom mesmo"
    assert normalize_case("Produto OK") == "Produto OK"  # pas tout en majuscules : inchangé


def test_purge_loops(cache):
    cache.add("m", ["Nao", "Muito bom", "não não não não"],
              ["No, no, no, no, no", "Very good", "no no no no"])
    # "Nao" boucle -> supprimé ; "não não..." boucle déjà dans la source -> gardé
    assert cache.purge_loops("m") == 1
    assert set(_cached(cache)) == {"Muito bom", "não não não não"}


# --- Test avec le vrai modèle : lent (téléchargement + chargement), lancé sur demande ---
@pytest.mark.skipif(os.getenv("RUN_SLOW") != "1", reason="lancer avec RUN_SLOW=1")
def test_real_marian_model_translates_portuguese():
    from olist_insights.translate import load_marian_translator

    translate = load_marian_translator()
    out = translate(["Não recebi o produto.", "Produto muito bom, chegou antes do prazo."])
    # Un traducteur a plusieurs formulations correctes ("I didn't get / receive the
    # product") : on vérifie le sens par des mots-clés, pas une phrase exacte.
    first, second = (o.lower() for o in out)
    assert "product" in first and ("receive" in first or "get" in first)
    assert "good" in second
    # omission repérée à l'étape 6 : la 2e phrase était perdue avant la traduction phrase par phrase
    (multi,) = translate(["Prazo de entrega não foi cumprido. Encomenda ainda não chegou."])
    assert "order" in multi.lower() or "package" in multi.lower()
