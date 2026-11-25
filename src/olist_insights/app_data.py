"""Accès aux données pour l'application Streamlit (étape 7).

Toute la logique est ici, testable sans Streamlit ; app.py ne fait que de l'affichage.
L'application ne fait AUCUN appel LLM : elle lit les explications déjà générées et validées
(gold.seller_explanations). Elle tourne donc sans clé API, sans quota, et n'affiche jamais
une explication qui n'a pas passé les contrôles.
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd

from olist_insights.explain import BENCHMARKS, PROMPT_VERSION, load_facts

DEFAULT_LLM_MODEL = "openai/gpt-oss-120b"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# chemins ancrés sur le projet : l'app marche quel que soit le dossier d'où on la lance
MODEL_REPORT = PROJECT_ROOT / "reports" / "model_metrics.json"
HUMAN_SCORES = PROJECT_ROOT / "reports" / "eval" / "human_scores.json"

# Libellés affichés pour les KPI comparés à la marketplace (mêmes clés que le LLM a vues)
KPI_LABELS = {
    "late_delivery_rate_pct": "Livraisons en retard (%)",
    "last_90d_late_delivery_rate_pct": "Retards, 90 derniers jours (%)",
    "late_carrier_handoff_rate_pct": "Remises tardives au transporteur (%)",
    "cancel_rate_pct": "Annulations (%)",
    "avg_rating": "Note moyenne (/5)",
    "last_90d_avg_rating": "Note moyenne, 90 derniers jours (/5)",
    "low_rating_rate_pct": "Notes 1-2 étoiles (%)",
    "last_90d_low_rating_rate_pct": "Notes 1-2 étoiles, 90 derniers jours (%)",
}


def connect(db_path: Path) -> duckdb.DuckDBPyConnection:
    # lecture seule : l'application ne doit jamais modifier la base ni bloquer le pipeline
    return duckdb.connect(str(db_path), read_only=True)


def risk_ranking(con: duckdb.DuckDBPyConnection, prompt_version: str = PROMPT_VERSION,
                 model: str = DEFAULT_LLM_MODEL) -> pd.DataFrame:
    """Vendeurs notés par le modèle, du plus risqué au moins risqué, avec un indicateur
    « explication disponible » (explication valide pour cette version de prompt)."""
    return con.execute(
        """
        SELECT r.risk_rank, r.seller_id, r.risk_score, r.state, r.main_category,
               r.n_reviews_90d::INTEGER AS n_reviews_90d, r.avg_rating_90d, r.low_rate_90d,
               r.late_rate_90d, r.late_handoff_rate_90d,
               e.seller_id IS NOT NULL AS has_explanation
        FROM gold.seller_risk r
        LEFT JOIN gold.seller_explanations e
               ON e.seller_id = r.seller_id AND e.valid
              AND e.prompt_version = ? AND e.model = ?
        ORDER BY r.risk_rank
        """,
        [prompt_version, model],
    ).df()


def kpi_comparison(facts: dict[str, float]) -> pd.DataFrame:
    """Tableau vendeur / marketplace pour chaque KPI de référence présent, avec un drapeau
    « pire que la marketplace » calculé exactement comme dans la validation du LLM."""
    rows = []
    for name, (ref_name, higher_is_worse) in BENCHMARKS.items():
        if name not in facts or ref_name not in facts:
            continue
        seller, ref = facts[name], facts[ref_name]
        rows.append({
            "kpi": KPI_LABELS.get(name, name),
            "vendeur": seller,
            "marketplace": ref,
            "pire_que_marketplace": (seller > ref) if higher_is_worse else (seller < ref),
        })
    return pd.DataFrame(rows, columns=["kpi", "vendeur", "marketplace", "pire_que_marketplace"])


def seller_facts(con: duckdb.DuckDBPyConnection, seller_id: str) -> dict[str, float]:
    # mêmes chiffres que ceux envoyés au LLM : ce que l'on affiche = ce qu'il a vu
    return load_facts(con, seller_id)


def seller_monthly(con: duckdb.DuckDBPyConnection, seller_id: str) -> pd.DataFrame:
    return con.execute(
        """
        SELECT month, n_orders, n_reviews, avg_rating, low_rating_rate, late_rate
        FROM gold.seller_monthly WHERE seller_id = ? ORDER BY month
        """,
        [seller_id],
    ).df()


def seller_explanation(con: duckdb.DuckDBPyConnection, seller_id: str,
                       prompt_version: str = PROMPT_VERSION, model: str = DEFAULT_LLM_MODEL) -> dict | None:
    """Explication valide stockée pour ce vendeur, ou None (jamais une explication invalide)."""
    row = con.execute(
        """
        SELECT explanation, review_ids, provider, model, prompt_version, attempts, created_at
        FROM gold.seller_explanations
        WHERE seller_id = ? AND valid AND prompt_version = ? AND model = ?
        """,
        [seller_id, prompt_version, model],
    ).fetchone()
    if row is None:
        return None
    explanation, review_ids, provider, model_, version, attempts, created_at = row
    return {
        "explanation": json.loads(explanation),
        "review_ids": json.loads(review_ids),
        "provider": provider, "model": model_, "prompt_version": version,
        "attempts": attempts, "created_at": created_at,
    }


def cited_reviews(con: duckdb.DuckDBPyConnection, seller_id: str, review_ids: dict[str, str]) -> dict[str, dict]:
    """Avis cités (« R3 » -> note, texte portugais d'origine, traduction), relus en base.
    Le filtre sur seller_id empêche d'afficher l'avis d'un autre vendeur."""
    if not review_ids:
        return {}
    rows = con.execute(
        """
        SELECT DISTINCT s.review_id, s.review_score, s.review_pt, t.review_en
        FROM silver.seller_reviews s JOIN silver.review_translations t USING (review_id)
        WHERE s.seller_id = ? AND s.review_id IN (SELECT unnest(?))
        """,
        [seller_id, list(review_ids.values())],
    ).fetchall()
    by_id = {rid: {"score": int(score), "pt": pt, "en": en} for rid, score, pt, en in rows}
    return {ref: by_id[rid] for ref, rid in review_ids.items() if rid in by_id}


def model_summary(path: Path = MODEL_REPORT) -> dict | None:
    """Chiffres clés du modèle sur le jeu de test, pour les afficher honnêtement à côté du score."""
    if not path.is_file():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    selected = report["selected_model"]
    test = report["test"]

    def pick(name: str) -> dict:
        m = test[name]
        return {"pr_auc": m["all"]["pr_auc"], "precision_top10": m["all"]["precision_top10"],
                "pr_auc_previously_ok": m["previously_ok"]["pr_auc"]}

    return {
        "selected_model": selected,
        "prevalence": test["constant_prior"]["all"]["prevalence"],
        "model": pick(selected),
        "rule": pick("rule_persistence"),
        "constant": pick("constant_prior"),
        "target": report["target"],
    }


def eval_summary(path: Path = HUMAN_SCORES) -> dict | None:
    """Scores de relecture (traduction, causes LLM), avec le nom du correcteur."""
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
