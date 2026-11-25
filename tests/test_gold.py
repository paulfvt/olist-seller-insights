import datetime as dt

import duckdb
import pytest

from olist_insights.bronze import TABLES, load_all
from olist_insights.config import load_settings
from olist_insights import gold, silver

# --- Silver minimal, valeurs attendues calculées à la main -------------------------
#  o1 s1      livrée le jour estimé (15h)       -> pas en retard, remise à temps
#  o2 s1 x2   livrée 2 jours après l'estimation -> en retard, remise tardive
#  o3 s1      annulée                            -> hors CA, compte dans l'annulation
#  o4 s1+s2   livrée 5 jours en avance, commande à deux vendeurs
#  o5 s2      expédiée, pas encore livrée        -> hors taux de retard

SILVER_SQL = """
CREATE SCHEMA silver;
CREATE TABLE silver.orders AS SELECT
    order_id, order_status,
    purchased_at::TIMESTAMP AS purchased_at,
    estimated_delivery_at::TIMESTAMP AS estimated_delivery_at,
    delivered_customer_at::TIMESTAMP AS delivered_customer_at,
    delivered_carrier_at::TIMESTAMP AS delivered_carrier_at,
    (order_status = 'delivered' AND delivered_customer_at IS NOT NULL) AS is_delivered
FROM (VALUES
    ('o1', 'delivered', '2018-01-10 10:00', '2018-01-20', '2018-01-20 15:00', '2018-01-11 09:00'),
    ('o2', 'delivered', '2018-01-15 10:00', '2018-01-20', '2018-01-22 10:00', '2018-01-18 09:00'),
    ('o3', 'canceled',  '2018-02-01 10:00', '2018-02-10', NULL,               NULL),
    ('o4', 'delivered', '2018-02-05 10:00', '2018-02-15', '2018-02-10 10:00', '2018-02-06 09:00'),
    ('o5', 'shipped',   '2018-02-20 10:00', '2018-03-01', NULL,               '2018-02-21 09:00')
) t(order_id, order_status, purchased_at, estimated_delivery_at, delivered_customer_at, delivered_carrier_at);

CREATE TABLE silver.order_items AS SELECT
    order_id, seller_id, product_id,
    price::DECIMAL(10,2) AS price, freight_value::DECIMAL(10,2) AS freight_value,
    shipping_limit_at::TIMESTAMP AS shipping_limit_at
FROM (VALUES
    ('o1', 's1', 'p1', 100, 10, '2018-01-12'),
    ('o2', 's1', 'p1',  50,  5, '2018-01-17'),
    ('o2', 's1', 'p2',  30,  5, '2018-01-17'),
    ('o3', 's1', 'p1', 999,  0, '2018-02-03'),
    ('o4', 's1', 'p2',  20,  2, '2018-02-07'),
    ('o4', 's2', 'p3',  40,  4, '2018-02-07'),
    ('o5', 's2', 'p3',  60,  6, '2018-02-22')
) t(order_id, seller_id, product_id, price, freight_value, shipping_limit_at);

CREATE TABLE silver.seller_reviews AS SELECT
    review_id, seller_id, review_score, has_text, review_created_at::TIMESTAMP AS review_created_at
FROM (VALUES
    ('r1', 's1', 5, true,  '2018-01-21'),
    ('r2', 's1', 1, true,  '2018-02-01'),
    ('r5', 's2', 2, false, '2018-03-01')
) t(review_id, seller_id, review_score, has_text, review_created_at);

CREATE TABLE silver.products AS SELECT * FROM (VALUES
    ('p1', 'bed_bath_table'), ('p2', 'toys'), ('p3', 'toys')
) t(product_id, category_en);

CREATE TABLE silver.sellers AS SELECT * FROM (VALUES
    ('s1', 'SP', 'sao paulo'), ('s2', 'RJ', 'rio de janeiro')
) t(seller_id, state, city);
"""


@pytest.fixture
def con():
    with duckdb.connect(":memory:") as connection:
        connection.execute(SILVER_SQL)
        gold.build_all(connection)
        yield connection


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
