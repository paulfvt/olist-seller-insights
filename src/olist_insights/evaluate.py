"""Évaluation a posteriori de la traduction et des explications LLM.

Pourquoi une étape séparée ?
----------------------------
Les contrôles de explain.py tournent au moment de la génération, avec le même contexte
que celui envoyé au modèle. Ici on mesure APRÈS COUP, par un chemin indépendant :
- les citations sont revérifiées contre les avis rechargés depuis la base à partir de
  leurs identifiants réels (et pas contre le prompt) ;
- les résultats sont agrégés par modèle et version de prompt, ce qui documente les choix.
Ce qu'aucun contrôle automatique ne voit (une traduction fidèle mais maladroite, un
symptôme présenté comme une cause) passe par une évaluation humaine : le script produit
des grilles CSV à remplir, puis calcule les scores.

Sorties (dossier reports/eval/) :
- translation_checks.json   : contrôles automatiques sur toutes les traductions
- translation_sample.csv    : 30 avis PT/EN à noter à la main
- llm_checks.json           : revérification de toutes les explications stockées
- llm_human_review.csv      : une ligne par cause à juger à la main
- human_scores.json         : scores des grilles une fois remplies (--summarize)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

import duckdb

from olist_insights.config import PROJECT_ROOT, load_settings
from olist_insights.explain import (
    NOT_EVIDENCE,
    SellerContext,
    _normalize,
    load_facts,
    validate_explanation,
)
from olist_insights.logging_utils import setup_logging
from olist_insights.translate import (
    MIN_LEN_FOR_RATIO,
    OMISSION_RATIO,
    has_repetition_loop,
    is_probable_omission,
)

logger = logging.getLogger(__name__)

EVAL_DIR = PROJECT_ROOT / "reports" / "eval"
SAMPLE_PER_SCORE = 6  # 6 avis x 5 notes = 30 avis à vérifier à la main
RATIO_HIGH = 2.0       # traduction plus de 2x plus longue que l'original : ajout probable

# Mots portugais fréquents dans les avis et absents de l'anglais : s'ils restent dans la
# traduction, un morceau n'a pas été traduit. Liste courte et précise, plutôt que longue et
# bruitée (on veut peu de faux positifs, quitte à rater des cas).
PT_LEFTOVERS = {
    "não", "nao", "produto", "muito", "entrega", "entregue", "recebi", "chegou", "prazo",
    "ótimo", "otimo", "loja", "compra", "pedido", "obrigado", "obrigada", "veio", "ainda",
    "também", "tambem", "porém", "bem", "gostei", "recomendo", "qualidade", "dias",
}
_WORD = re.compile(r"[a-zà-ÿ]+")


# ---------------------------------------------------------------------------
# Traduction
# ---------------------------------------------------------------------------


def length_ratio_flag(pt: str, en: str) -> str | None:
    """'omission' (phrase perdue, même règle que translate.py), 'long' (ajout probable) ou None."""
    if is_probable_omission(pt, en):
        return "omission"
    if len(pt) >= MIN_LEN_FOR_RATIO and len(en) / len(pt) > RATIO_HIGH:
        return "long"
    return None


def portuguese_leftovers(en: str) -> set[str]:
    return set(_WORD.findall(en.lower())) & PT_LEFTOVERS


def translation_checks(con: duckdb.DuckDBPyConnection, n_examples: int = 5) -> dict:
    rows = con.execute(
        "SELECT review_id, review_pt, review_en, translation_model FROM silver.review_translations"
    ).fetchall()
    translated = [r for r in rows if r[2] is not None and not r[3].startswith("copied")]
    flags: dict[str, list] = {"loop": [], "omission": [], "long": [], "pt_leftover": []}
    for rid, pt, en, _ in translated:
        if has_repetition_loop(en) and not has_repetition_loop(pt):
            flags["loop"].append((rid, pt, en))
        if (flag := length_ratio_flag(pt, en)) is not None:
            flags[flag].append((rid, pt, en))
        if words := portuguese_leftovers(en):
            flags["pt_leftover"].append((rid, pt, en, sorted(words)))
    n = len(translated)
    return {
        "n_reviews": len(rows),
        "n_translated_by_model": n,
        "n_copied_no_letters": sum(1 for r in rows if r[3] and r[3].startswith("copied")),
        "n_missing": sum(1 for r in rows if r[2] is None),
        "flags": {
            name: {
                "count": len(items),
                "rate_pct": round(100 * len(items) / max(n, 1), 2),
                "examples": [dict(zip(("review_id", "pt", "en", "words"), item)) for item in items[:n_examples]],
            }
            for name, items in flags.items()
        },
        "notes": {
            "omission": f"texte d'au moins 2 phrases et {MIN_LEN_FOR_RATIO} caractères, len(EN)/len(PT) < {OMISSION_RATIO}",
            "long": f"len(EN)/len(PT) > {RATIO_HIGH}",
            "pt_leftover": "mot portugais fréquent resté dans la traduction",
        },
    }


def translation_sample(con: duckdb.DuckDBPyConnection, per_score: int = SAMPLE_PER_SCORE) -> list[dict]:
    """Échantillon stratifié par note, déterministe (ordre par empreinte de l'identifiant).

    Stratifié : un tirage au hasard simple donnerait surtout des avis 5 étoiles (les plus
    nombreux), alors que ce sont les avis négatifs que le LLM utilise pour expliquer.
    """
    rows = con.execute(
        """
        SELECT review_id, score, review_pt, review_en FROM (
            SELECT t.review_id, r.review_score AS score, t.review_pt, t.review_en,
                   row_number() OVER (PARTITION BY r.review_score ORDER BY md5(t.review_id)) AS rn
            FROM silver.review_translations t
            JOIN (SELECT DISTINCT review_id, review_score FROM silver.order_reviews) r USING (review_id)
            WHERE t.review_en IS NOT NULL AND NOT t.translation_model LIKE 'copied%'
        )
        WHERE rn <= ?
        ORDER BY score, review_id
        """,
        [per_score],
    ).fetchall()
    return [{"review_id": rid, "score": score, "review_pt": pt, "review_en": en,
             "adequacy_0_2": "", "fluency_0_2": "", "notes": ""} for rid, score, pt, en in rows]


# ---------------------------------------------------------------------------
# Explications LLM
# ---------------------------------------------------------------------------


def stored_explanations(con: duckdb.DuckDBPyConnection) -> list[dict]:
    rows = con.execute(
        """
        SELECT seller_id, provider, model, prompt_version, valid, attempts, explanation, issues, review_ids
        FROM gold.seller_explanations ORDER BY provider, model, prompt_version, seller_id
        """
    ).fetchall()
    keys = ("seller_id", "provider", "model", "prompt_version", "valid", "attempts", "explanation", "issues", "review_ids")
    out = []
    for row in rows:
        d = dict(zip(keys, row))
        for k in ("explanation", "issues", "review_ids"):
            d[k] = json.loads(d[k]) if d[k] else None
        out.append(d)
    return out


def reviews_from_ids(con: duckdb.DuckDBPyConnection, seller_id: str, review_ids: dict[str, str]) -> dict[str, dict]:
    """Recharge les avis cités depuis la base, en vérifiant qu'ils appartiennent bien au vendeur.

    Chemin indépendant de explain.load_reviews : si l'échantillonnage ou le prompt avaient
    un défaut, cette revérification ne le partagerait pas.
    """
    if not review_ids:
        return {}
    found = dict(con.execute(
        """
        SELECT s.review_id, t.review_en
        FROM silver.seller_reviews s JOIN silver.review_translations t USING (review_id)
        WHERE s.seller_id = ? AND s.review_id IN (SELECT unnest(?))
        """,
        [seller_id, list(review_ids.values())],
    ).fetchall())
    return {ref: {"review_id": rid, "text": found[rid]} for ref, rid in review_ids.items() if rid in found}


def recheck_explanation(con: duckdb.DuckDBPyConnection, record: dict) -> dict:
    """Revalide une explication stockée et mesure son ancrage dans les données."""
    expl = record["explanation"]
    ctx = SellerContext(record["seller_id"], load_facts(con, record["seller_id"]),
                        reviews_from_ids(con, record["seller_id"], record["review_ids"] or {}))
    _, issues = validate_explanation(expl, ctx)
    causes = expl.get("causes", [])
    quotes = [r for c in causes for r in c.get("supporting_reviews", [])]
    kpis = [k for c in causes for k in c.get("supporting_kpis", [])]
    quotes_ok = sum(1 for q in quotes
                    if q["review_ref"] in ctx.reviews
                    and _normalize(q["quote"]) in _normalize(ctx.reviews[q["review_ref"]]["text"]))
    kpis_ok = sum(1 for k in kpis
                  if k["kpi_name"] in ctx.facts and k["kpi_name"] not in NOT_EVIDENCE
                  and abs(k["value"] - ctx.facts[k["kpi_name"]]) < 1e-6)
    return {
        "valid_on_recheck": not issues,
        "issues": issues,
        "n_causes": len(causes),
        "n_quotes": len(quotes),
        "n_quotes_verified": quotes_ok,
        "n_kpis": len(kpis),
        "n_kpis_verified": kpis_ok,
        "n_distinct_reviews_cited": len({q["review_ref"] for q in quotes}),
    }


def _issue_type(issue: str) -> str:
    """Regroupe les messages d'erreur par type (sans les valeurs propres à chaque vendeur)."""
    for pattern, label in [
        ("schéma", "schema"), ("JSON", "json"), ("nombre non fourni", "invented_number"),
        ("KPI inconnu", "unknown_kpi"), ("valeur fournie", "wrong_kpi_value"),
        ("n'est pas pire", "kpi_not_worse_than_marketplace"), ("pas une preuve", "risk_score_as_evidence"),
        ("avis inexistant", "unknown_review"), ("n'apparaît pas", "quote_not_in_review"),
        ("aucun KPI", "cause_without_kpi"), ("aucune cause", "nothing_to_report_but_kpi_worse"),
        ("cause_index", "bad_cause_index"),
    ]:
        if pattern in issue:
            return label
    return "other"


def llm_checks(con: duckdb.DuckDBPyConnection) -> dict:
    """Agrégats par (fournisseur, modèle, version de prompt)."""
    groups: dict[str, dict] = {}
    for rec in stored_explanations(con):
        key = f"{rec['provider']}/{rec['model']}/{rec['prompt_version']}"
        g = groups.setdefault(key, {"n": 0, "valid_at_generation": 0, "attempts": [], "issue_types": Counter(),
                                    "rechecked": 0, "valid_on_recheck": 0, "n_causes": 0, "n_quotes": 0,
                                    "n_quotes_verified": 0, "n_kpis": 0, "n_kpis_verified": 0,
                                    "nothing_to_report": 0})
        g["n"] += 1
        g["valid_at_generation"] += bool(rec["valid"])
        g["attempts"].append(rec["attempts"])
        for issue in rec["issues"] or []:
            g["issue_types"][_issue_type(issue)] += 1
        if rec["valid"] and rec["explanation"]:
            r = recheck_explanation(con, rec)
            g["rechecked"] += 1
            g["valid_on_recheck"] += r["valid_on_recheck"]
            g["nothing_to_report"] += r["n_causes"] == 0
            for k in ("n_causes", "n_quotes", "n_quotes_verified", "n_kpis", "n_kpis_verified"):
                g[k] += r[k]

    report = {}
    for key, g in groups.items():
        report[key] = {
            "n_sellers": g["n"],
            "valid_at_generation_pct": round(100 * g["valid_at_generation"] / g["n"], 1),
            "avg_attempts": round(sum(g["attempts"]) / g["n"], 2),
            "issues_of_invalid_answers": dict(g["issue_types"].most_common()),
            "valid_answers_rechecked": g["rechecked"],
            "still_valid_on_recheck": g["valid_on_recheck"],
            "nothing_to_report": g["nothing_to_report"],
            "avg_causes": round(g["n_causes"] / max(g["rechecked"] - g["nothing_to_report"], 1), 2),
            "quotes_verified_pct": round(100 * g["n_quotes_verified"] / max(g["n_quotes"], 1), 1),
            "kpis_verified_pct": round(100 * g["n_kpis_verified"] / max(g["n_kpis"], 1), 1),
        }
    return report


def llm_human_review_rows(con: duckdb.DuckDBPyConnection, model: str, prompt_version: str) -> list[dict]:
    """Une ligne par cause des explications valides d'une configuration, à juger à la main."""
    rows = []
    for rec in stored_explanations(con):
        if (rec["model"], rec["prompt_version"]) != (model, prompt_version) or not rec["valid"]:
            continue
        reviews = reviews_from_ids(con, rec["seller_id"], rec["review_ids"] or {})
        for i, cause in enumerate(rec["explanation"]["causes"]):
            rows.append({
                "seller_id": rec["seller_id"],
                "cause_index": i,
                "title": cause["title"],
                "explanation": cause["explanation"],
                "kpis": "; ".join(f"{k['kpi_name']}={k['value']:g}" for k in cause["supporting_kpis"]),
                "quotes": " | ".join(f"{q['review_ref']}: {q['quote']}" for q in cause["supporting_reviews"]),
                "full_reviews": " | ".join(f"{q['review_ref']}: {reviews.get(q['review_ref'], {}).get('text', '?')}"
                                           for q in cause["supporting_reviews"]),
                # à remplir par l'évaluateur : y / n
                "supported_by_reviews": "", "is_cause_not_symptom": "", "actionable": "", "notes": "",
            })
    return rows


# ---------------------------------------------------------------------------
# Grilles humaines remplies -> scores
# ---------------------------------------------------------------------------


def summarize_translation_grades(rows: list[dict]) -> dict | None:
    graded = [r for r in rows if str(r.get("adequacy_0_2", "")).strip() != ""]
    if not graded:
        return None
    adequacy = [int(r["adequacy_0_2"]) for r in graded]
    fluency = [int(r["fluency_0_2"]) for r in graded if str(r.get("fluency_0_2", "")).strip() != ""]
    by_score: dict[str, list[int]] = {}
    for r in graded:
        by_score.setdefault(str(r["score"]), []).append(int(r["adequacy_0_2"]))
    return {
        "n_graded": len(graded),
        "adequacy_mean_0_2": round(sum(adequacy) / len(adequacy), 2),
        "meaning_preserved_pct": round(100 * sum(a == 2 for a in adequacy) / len(adequacy), 1),
        "meaning_lost_pct": round(100 * sum(a == 0 for a in adequacy) / len(adequacy), 1),
        "fluency_mean_0_2": round(sum(fluency) / len(fluency), 2) if fluency else None,
        "adequacy_mean_by_review_score": {k: round(sum(v) / len(v), 2) for k, v in sorted(by_score.items())},
    }


def summarize_cause_grades(rows: list[dict]) -> dict | None:
    out = {}
    for col in ("supported_by_reviews", "is_cause_not_symptom", "actionable"):
        answers = [str(r.get(col, "")).strip().lower() for r in rows]
        answers = [a for a in answers if a in {"y", "n"}]
        if answers:
            out[f"{col}_pct"] = round(100 * answers.count("y") / len(answers), 1)
            out[f"{col}_n"] = len(answers)
    return out or None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _write_csv(path: Path, rows: list[dict]) -> None:
    """Grille pensée pour Excel en français : séparateur ';' et BOM UTF-8 (utf-8-sig),
    sinon Excel met tout dans une colonne et abîme les accents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    """Lit la grille quel que soit le séparateur (';' Excel FR, ',' Google Sheets ou LibreOffice)."""
    with path.open(newline="", encoding="utf-8-sig") as f:
        header = f.readline()
        f.seek(0)
        delimiter = ";" if header.count(";") >= header.count(",") else ","
        return list(csv.DictReader(f, delimiter=delimiter))


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Évaluation de la traduction et des explications LLM.")
    parser.add_argument("--summarize", action="store_true",
                        help="calcule les scores des grilles CSV remplies à la main")
    parser.add_argument("--model", default="openai/gpt-oss-120b", help="configuration à juger à la main")
    parser.add_argument("--prompt-version", default=None, help="par défaut : celle de explain.py")
    parser.add_argument("--force", action="store_true", help="écrase les grilles CSV même si elles sont remplies")
    parser.add_argument("--grader", default="non précisé",
                        help="qui a rempli les grilles (écrit dans le rapport, par transparence)")
    args = parser.parse_args(argv)

    settings = load_settings()
    setup_logging(settings.log_level)
    sample_path = EVAL_DIR / "translation_sample.csv"
    causes_path = EVAL_DIR / "llm_human_review.csv"

    if args.summarize:
        scores = {
            "grader": args.grader,
            "translation": summarize_translation_grades(_read_csv(sample_path)) if sample_path.exists() else None,
            "llm_causes": summarize_cause_grades(_read_csv(causes_path)) if causes_path.exists() else None,
            # grilles des versions de prompt précédentes, gardées pour comparer avant / après
            "llm_causes_previous": {p.stem.rsplit("_", 1)[1]: summarize_cause_grades(_read_csv(p))
                                    for p in sorted(EVAL_DIR.glob("llm_human_review_v*.csv"))},
        }
        _write_json(EVAL_DIR / "human_scores.json", scores)
        logger.info("Scores humains : %s", json.dumps(scores, ensure_ascii=False))
        return 0

    from olist_insights.explain import PROMPT_VERSION
    prompt_version = args.prompt_version or PROMPT_VERSION
    with duckdb.connect(str(settings.duckdb_path), read_only=True) as con:
        t_checks = translation_checks(con)
        _write_json(EVAL_DIR / "translation_checks.json", t_checks)
        for name, f in t_checks["flags"].items():
            logger.info("traduction | %-12s %5d (%.2f %%)", name, f["count"], f["rate_pct"])

        l_checks = llm_checks(con)
        _write_json(EVAL_DIR / "llm_checks.json", l_checks)
        for key, g in l_checks.items():
            logger.info("LLM | %-40s %2d vendeurs | valides %5.1f %% | essais moy. %.2f | citations vérifiées %5.1f %% "
                        "| KPI vérifiés %5.1f %% | revalidés %d/%d",
                        key, g["n_sellers"], g["valid_at_generation_pct"], g["avg_attempts"],
                        g["quotes_verified_pct"], g["kpis_verified_pct"],
                        g["still_valid_on_recheck"], g["valid_answers_rechecked"])

        # Les grilles déjà remplies ne sont pas écrasées (des heures de relecture humaine).
        grids = [(sample_path, translation_sample(con), "adequacy_0_2"),
                 (causes_path, llm_human_review_rows(con, args.model, prompt_version), "supported_by_reviews")]
        for path, rows, grade_col in grids:
            if path.exists() and not args.force and any(r.get(grade_col, "").strip() for r in _read_csv(path)):
                logger.warning("%s déjà rempli : conservé (--force pour l'écraser)", path.name)
            elif rows:
                _write_csv(path, rows)
                logger.info("Grille à remplir : %s (%d lignes)", path, len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
