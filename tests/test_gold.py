import datetime as dt

import duckdb
import pytest

from olist_insights.bronze import TABLES, load_all
from olist_insights.config import load_settings
from olist_insights import gold, silver



@pytest.fixture
def con(silver_con):  # silver_con : voir conftest.py
    gold.build_all(silver_con)
    return silver_con


def _kpis(con, seller_id: str) -> dict:
    return con.execute("SELECT * FROM gold.seller_kpis WHERE seller_id = ?", [seller_id]).df().to_dict("records")[0]


def test_sales_kpis(con):
    s1 = _kpis(con, "s1")
    assert s1["n_orders"] == 4 and s1["n_orders_valid"] == 3
    assert s1["cancel_rate"] == pytest.approx(0.25)
    assert s1["n_items"] == 4                                 # o3 annulée exclue
    assert float(s1["revenue"]) == pytest.approx(200)         # 100 + 50 + 30 + 20, sans les 999 annulés
    assert float(s1["avg_order_value"]) == pytest.approx(200 / 3)
    assert float(s1["freight_ratio"]) == pytest.approx(22 / 200)


def test_multi_seller_order_split(con):
    # o4 : chaque vendeur ne reçoit que ses propres articles
    assert float(_kpis(con, "s2")["revenue"]) == pytest.approx(40 + 60)


def test_delivery_kpis(con):
    s1 = _kpis(con, "s1")
    assert s1["n_delivered"] == 3
    assert s1["late_rate"] == pytest.approx(1 / 3)            # o1 livrée LE jour estimé : pas en retard
    assert s1["avg_delay_days"] == pytest.approx((0 + 2 - 5) / 3)
    assert s1["late_handoff_rate"] == pytest.approx(1 / 3)    # o2 ; o3 sans remise ignorée
    s2 = _kpis(con, "s2")
    assert s2["n_delivered"] == 1 and s2["late_rate"] == 0    # o5 pas livrée : hors calcul


def test_rating_kpis(con):
    s1 = _kpis(con, "s1")
    assert s1["n_reviews"] == 2 and s1["avg_rating"] == pytest.approx(3)
    assert s1["low_rating_rate"] == pytest.approx(0.5)
    assert s1["n_reviews_with_text"] == 2


def test_main_category(con):
    assert _kpis(con, "s1")["main_category"] == "bed_bath_table"  # 3 articles contre 2
    assert _kpis(con, "s1")["n_categories"] == 2
    assert _kpis(con, "s2")["main_category"] == "toys"


def test_monthly(con):
    rows = {
        (r["seller_id"], r["month"]): r
        for r in con.execute("SELECT * FROM gold.seller_monthly").df().to_dict("records")
    }
    jan, feb = (dt.datetime(2018, m, 1) for m in (1, 2))
    s1_jan = rows[("s1", jan)]
    assert s1_jan["n_orders"] == 2 and float(s1_jan["revenue"]) == pytest.approx(180)
    assert s1_jan["late_rate"] == pytest.approx(0.5)
    assert s1_jan["n_reviews"] == 1 and s1_jan["avg_rating"] == 5
    # mois avec ventes mais sans avis : note NULL, pas 0
    s2_feb = rows[("s2", feb)]
    assert s2_feb["n_reviews"] == 0 and s2_feb["avg_rating"] != s2_feb["avg_rating"]  # NaN
    # mois avec un avis mais sans vente : la ligne existe quand même (avis rangé au mois de l'avis)
    s2_mar = rows[("s2", dt.datetime(2018, 3, 1))]
    assert s2_mar["n_orders"] == 0 and s2_mar["avg_rating"] == 2


def test_consistency_checks_pass(con):
    assert all(gold.consistency_checks(con).values())


def test_build_is_idempotent(con):
    first = con.execute("SELECT * FROM gold.seller_kpis ORDER BY seller_id").fetchall()
    gold.build_all(con)
    assert con.execute("SELECT * FROM gold.seller_kpis ORDER BY seller_id").fetchall() == first


# --- Intégration sur les vraies données (ignorée si les CSV sont absents) ---
RAW_DIR = load_settings().raw_dir
HAS_RAW = all((RAW_DIR / f).is_file() for f in TABLES.values())


@pytest.mark.skipif(not HAS_RAW, reason="CSV Olist absents de data/raw")
def test_real_data_gold_consistency():
    with duckdb.connect(":memory:") as con:
        load_all(con, RAW_DIR)
        silver.build_all(con)
        counts = gold.build_all(con)
        assert counts["seller_kpis"] == 3095
        failed = [name for name, ok in gold.consistency_checks(con).items() if not ok]
        assert failed == []
