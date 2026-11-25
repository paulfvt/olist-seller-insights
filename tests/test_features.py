import datetime as dt

import pandas as pd
import pytest

from olist_insights.features import TargetConfig, build_dataset, build_features, build_labels, month_starts

T = dt.date(2018, 2, 1)
CFG = TargetConfig(min_past_reviews=1, min_future_reviews=1)

# o8 (vendeur s1) : achetée avant T, en retard à T, remise au transporteur et livrée APRÈS T.
# C'est le cas piège : sa livraison existe dans la base, mais n'était pas connue à T.
O8_SQL = """
INSERT INTO silver.orders VALUES
    ('o8', 'delivered', '2018-01-25 10:00', '2018-01-30', '2018-02-20 10:00', '2018-02-02 09:00', true);
INSERT INTO silver.order_items VALUES ('o8', 's1', 'p2', 70, 7, '2018-01-27');
"""

# Ramène la base à ce qu'on savait le jour T : tout événement daté de T ou après disparaît.
AS_OF_T_SQL = """
DELETE FROM silver.seller_reviews WHERE review_created_at >= DATE '2018-02-01';
DELETE FROM silver.order_items WHERE order_id IN (SELECT order_id FROM silver.orders WHERE purchased_at >= DATE '2018-02-01');
DELETE FROM silver.orders WHERE purchased_at >= DATE '2018-02-01';
UPDATE silver.orders SET delivered_customer_at = NULL WHERE delivered_customer_at >= DATE '2018-02-01';
UPDATE silver.orders SET delivered_carrier_at = NULL WHERE delivered_carrier_at >= DATE '2018-02-01';
"""


@pytest.fixture
def con(silver_con):
    silver_con.execute(O8_SQL)
    return silver_con


def test_features_ignore_everything_after_cutoff(con):
    """Test anti-fuite : features sur toute la base == features sur la base « vue à T »."""
    full = build_features(con, [T], CFG)
    con.execute(AS_OF_T_SQL)
    as_of_t = build_features(con, [T], CFG)
    assert len(full) == 1  # s1 seulement (s2 n'a aucun avis avant T)
    pd.testing.assert_frame_equal(full, as_of_t)


def test_feature_values(con):
    s1 = build_features(con, [T], CFG).iloc[0]
    assert s1["n_reviews_90d"] == 1 and s1["avg_rating_90d"] == 5  # r2 est daté de T : futur
    assert s1["n_delivered_90d"] == 2                 # o1, o2 ; o8 livrée après T
    assert s1["late_rate_90d"] == pytest.approx(0.5)  # o2
    assert s1["late_handoff_rate_90d"] == pytest.approx(0.5)  # o2 ; remise de o8 après T
    assert s1["overdue_open_rate"] == pytest.approx(1 / 3)    # o8 en retard à T sur o1, o2, o8
    assert s1["n_orders_90d"] == 3                    # o3, achetée le jour T, exclue
    assert s1["seniority_days"] == 22                 # première commande le 2018-01-10
    # égalité 2 articles / 2 articles -> ordre alphabétique, résultat reproductible
    assert s1["main_category"] == "bed_bath_table"


def test_labels_use_only_the_horizon(con):
    labels = build_labels(con, [T], CFG).set_index("seller_id")
    assert labels.loc["s1", "n_reviews_future"] == 1   # r2 (note 1), daté de T
    assert labels.loc["s1", "label"] == 1
    assert labels.loc["s2", "label"] == 1              # r5 : note 2, dans les 90 jours
    short = build_labels(con, [T], TargetConfig(horizon_days=20, min_future_reviews=1))
    assert set(short["seller_id"]) == {"s1"}           # r5 (2018-03-01) hors d'un horizon de 20 jours


def test_population_and_minimum_future_reviews(con):
    assert build_features(con, [T], TargetConfig(min_past_reviews=2)).empty
    # s1 a 1 avis futur : écarté si on en exige 2 (cible trop incertaine)
    assert build_dataset(con, [T], TargetConfig(min_past_reviews=1, min_future_reviews=2)).empty


def test_month_starts():
    assert month_starts("2017-11", "2018-01") == [dt.date(2017, 11, 1), dt.date(2017, 12, 1), dt.date(2018, 1, 1)]
