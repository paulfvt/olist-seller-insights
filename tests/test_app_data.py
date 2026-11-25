import json
from pathlib import Path

import duckdb
import pytest

from olist_insights import app_data
from olist_insights.config import load_settings

EXPL = {"summary": "s", "causes": [], "improvements": []}


@pytest.fixture
def con():
    with duckdb.connect(":memory:") as c:
        c.execute("CREATE SCHEMA gold; CREATE SCHEMA silver")
        c.execute("""CREATE TABLE gold.seller_risk AS SELECT * FROM (VALUES
            ('s1', 0.9, 1, 'SP', 'toys', 10.0, 3.1, 0.4, 0.2, 0.3),
            ('s2', 0.5, 2, 'RJ', 'auto', 6.0, 4.5, 0.0, 0.0, 0.1))
            t(seller_id, risk_score, risk_rank, state, main_category, n_reviews_90d,
              avg_rating_90d, low_rate_90d, late_rate_90d, late_handoff_rate_90d)""")
        c.execute("""CREATE TABLE gold.seller_explanations (seller_id VARCHAR, provider VARCHAR, model VARCHAR,
            prompt_version VARCHAR, valid BOOLEAN, attempts INTEGER, explanation JSON, issues JSON,
            review_ids JSON, created_at TIMESTAMP)""")
        rows = [  # s1 : valide en v4 ; s2 : invalide en v4 et valide seulement en v3
            ("s1", "v4", True), ("s2", "v4", False), ("s2", "v3", True)]
        for sid, version, valid in rows:
            c.execute("INSERT INTO gold.seller_explanations VALUES (?, 'groq', ?, ?, ?, 1, ?, '[]', ?, now())",
                      [sid, app_data.DEFAULT_LLM_MODEL, version, valid, json.dumps(EXPL),
                       json.dumps({"R1": "rA", "R2": "rB"})])
        c.execute("""CREATE TABLE silver.seller_reviews AS SELECT * FROM (VALUES
            ('rA', 's1', 'o1', 1, 'Atrasou'), ('rA', 's1', 'o2', 1, 'Atrasou'),  -- avis partagé par 2 commandes
            ('rB', 's2', 'o3', 5, 'Ótimo'))
            t(review_id, seller_id, order_id, review_score, review_pt)""")
        c.execute("""CREATE TABLE silver.review_translations AS SELECT * FROM (VALUES
            ('rA', 'It was late'), ('rB', 'Great')) t(review_id, review_en)""")
        yield c


def test_risk_ranking_flags_only_valid_current_explanations(con):
    df = app_data.risk_ranking(con, prompt_version="v4")
    assert list(df["seller_id"]) == ["s1", "s2"]  # trié par rang
    # s2 a une explication valide, mais d'une ancienne version de prompt : non signalée
    assert list(df["has_explanation"]) == [True, False]


def test_seller_explanation_never_returns_invalid_or_other_version(con):
    assert app_data.seller_explanation(con, "s1", "v4")["explanation"] == EXPL
    assert app_data.seller_explanation(con, "s2", "v4") is None   # invalide
    assert app_data.seller_explanation(con, "s3", "v4") is None   # inconnu


def test_cited_reviews_checks_ownership_and_deduplicates(con):
    got = app_data.cited_reviews(con, "s1", {"R1": "rA", "R2": "rB"})
    # rB appartient à s2 : ne doit pas s'afficher comme un avis de s1 ; rA n'apparaît qu'une fois
    assert got == {"R1": {"score": 1, "pt": "Atrasou", "en": "It was late"}}


def test_kpi_comparison_uses_validation_direction():
    facts = {"late_delivery_rate_pct": 10.0, "marketplace_late_delivery_rate_pct": 6.9,
             "avg_rating": 4.2, "marketplace_avg_rating": 4.04,
             "cancel_rate_pct": 1.0}  # pas de référence marketplace : ligne absente
    df = app_data.kpi_comparison(facts).set_index("kpi")
    assert df.loc["Livraisons en retard (%)", "pire_que_marketplace"]   # plus haut = pire
    assert not df.loc["Note moyenne (/5)", "pire_que_marketplace"]      # plus haut = mieux
    assert len(df) == 2


def test_kpi_comparison_empty():
    assert app_data.kpi_comparison({}).empty


def test_model_summary(tmp_path):
    assert app_data.model_summary(tmp_path / "absent.json") is None
    m = {"all": {"pr_auc": 0.2, "precision_top10": 0.3}, "previously_ok": {"pr_auc": 0.1}}
    report = {"selected_model": "logreg", "target": {"bad_threshold": 0.25},
              "test": {"constant_prior": {"all": {**m["all"], "prevalence": 0.12}, "previously_ok": m["previously_ok"]},
                       "rule_persistence": m, "logreg": m}}
    path = tmp_path / "m.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    s = app_data.model_summary(path)
    assert s["prevalence"] == 0.12 and s["model"]["precision_top10"] == 0.3


# --- Intégration : l'application complète sur la vraie base (ignorée si absente) ---
DB = load_settings().duckdb_path
APP = Path(__file__).resolve().parents[1] / "app.py"


@pytest.mark.skipif(not DB.is_file(), reason="base DuckDB absente")
def test_app_runs_without_errors_on_real_data():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception, at.exception
    # fiche vendeur par défaut : le plus risqué avec explication, causes affichées
    assert any("Cause 1" in e.label for e in at.expander)
    # vendeur sans explication : message d'aide, pas de plantage
    box = at.selectbox[0]
    no_expl = next(opt for opt in box.options if "explication" not in opt)
    box.select(no_expl).run()
    assert not at.exception, at.exception
    assert any("Pas d'explication" in i.value for i in at.info)
