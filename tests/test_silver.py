import duckdb
import pandas as pd
import pytest

from olist_insights.bronze import TABLES, load_all
from olist_insights.config import load_settings
from olist_insights.silver import (
    DataQualityError,
    build_all,
    count_cast_failures,
    quality_report,
)

# --- Bronze minimal construit à la main (tout en texte, comme le vrai Bronze) ---

ORDERS = [
    # o1 : normale, livrée
    dict(order_id="o1", customer_id="c1", order_status="delivered",
         order_purchase_timestamp="2018-01-01 10:00:00", order_approved_at="2018-01-01 11:00:00",
         order_delivered_carrier_date="2018-01-02 10:00:00", order_delivered_customer_date="2018-01-05 10:00:00",
         order_estimated_delivery_date="2018-01-10 00:00:00"),
    # o2 : "delivered" mais sans date de livraison, transporteur avant l'achat
    dict(order_id="o2", customer_id="c2", order_status="delivered",
         order_purchase_timestamp="2018-02-01 10:00:00", order_approved_at=None,
         order_delivered_carrier_date="2018-01-31 10:00:00", order_delivered_customer_date=None,
         order_estimated_delivery_date="2018-02-10 00:00:00"),
    # o3 : même client que o1 ; o4 : deux vendeurs ; o5 : annulée sans articles
    *[dict(order_id=o, customer_id="c1", order_status=s,
           order_purchase_timestamp="2018-01-01 10:00:00", order_approved_at=None,
           order_delivered_carrier_date=None, order_delivered_customer_date=None,
           order_estimated_delivery_date="2018-01-10 00:00:00")
      for o, s in [("o3", "shipped"), ("o4", "shipped"), ("o5", "canceled")]],
]

ITEMS = [
    dict(order_id="o1", order_item_id="1", product_id="p1", seller_id="s1",
         shipping_limit_date="2018-01-03 00:00:00", price="10.50", freight_value="2.00"),
    dict(order_id="o2", order_item_id="1", product_id="p2", seller_id="s2",
         shipping_limit_date="2018-02-03 00:00:00", price="99.90", freight_value="0.00"),
    dict(order_id="o3", order_item_id="1", product_id="p1", seller_id="s1",
         shipping_limit_date="2018-01-03 00:00:00", price="10.50", freight_value="2.00"),
    dict(order_id="o4", order_item_id="1", product_id="p1", seller_id="s1",
         shipping_limit_date="2018-01-03 00:00:00", price="10.50", freight_value="2.00"),
    dict(order_id="o4", order_item_id="2", product_id="p2", seller_id="s2",
         shipping_limit_date="2018-01-03 00:00:00", price="5.00", freight_value="1.00"),
]

REVIEWS = [
    # r1 : un seul avis pour deux commandes (o1, o3) du même vendeur s1
    *[dict(review_id="r1", order_id=o, review_score="5", review_comment_title=None,
           review_comment_message="  Muito\r\n\r\n   bom   produto ",
           review_creation_date="2018-01-06 00:00:00", review_answer_timestamp="2018-01-07 00:00:00")
      for o in ["o1", "o3"]],
    # r2 et r3 : deux avis différents pour la même commande o2
    dict(review_id="r2", order_id="o2", review_score="5", review_comment_title=None,
         review_comment_message="   ", review_creation_date="2018-02-05 00:00:00",
         review_answer_timestamp="2018-02-05 00:00:00"),
    dict(review_id="r3", order_id="o2", review_score="1", review_comment_title="Ruim",
         review_comment_message="Não recebi o produto\r\n", review_creation_date="2018-02-12 00:00:00",
         review_answer_timestamp="2018-02-13 00:00:00"),
    # r4 : commande à deux vendeurs -> non attribuable ; r5 : commande sans articles
    dict(review_id="r4", order_id="o4", review_score="3", review_comment_title=None,
         review_comment_message="ok", review_creation_date="2018-01-15 00:00:00",
         review_answer_timestamp="2018-01-15 00:00:00"),
    dict(review_id="r5", order_id="o5", review_score="1", review_comment_title=None,
         review_comment_message=None, review_creation_date="2018-01-15 00:00:00",
         review_answer_timestamp="2018-01-15 00:00:00"),
]

PAYMENTS = [dict(order_id="o1", payment_sequential="1", payment_type="credit_card",
                 payment_installments="3", payment_value="12.50")]
CUSTOMERS = [dict(customer_id="c1", customer_unique_id="u1", customer_zip_code_prefix="01310",
                  customer_city="sao paulo", customer_state="SP")]
SELLERS = [dict(seller_id=s, seller_zip_code_prefix="04195", seller_city="sao paulo", seller_state="SP")
           for s in ["s1", "s2"]]
PRODUCTS = [
    dict(product_id="p1", product_category_name="bebes", product_photos_qty="1", product_weight_g="200"),
    dict(product_id="p2", product_category_name=None, product_photos_qty=None, product_weight_g="50"),
    dict(product_id="p3", product_category_name="pc_gamer", product_photos_qty="2", product_weight_g="900"),
]
TRANSLATION = [dict(product_category_name="bebes", product_category_name_english="baby")]


def _make_bronze(con, table: str, rows: list[dict]) -> None:
    df = pd.DataFrame(rows).astype(object)  # object : None reste None, tout le reste est du texte
    con.register("tmp_df", df)
    con.execute(f"CREATE OR REPLACE TABLE bronze.{table} AS SELECT * FROM tmp_df")
    con.unregister("tmp_df")


@pytest.fixture
def con():
    with duckdb.connect(":memory:") as connection:
        connection.execute("CREATE SCHEMA bronze")
        for table, rows in {
            "orders": ORDERS, "order_items": ITEMS, "order_reviews": REVIEWS,
            "order_payments": PAYMENTS, "customers": CUSTOMERS, "sellers": SELLERS,
            "products": PRODUCTS, "category_translation": TRANSLATION,
        }.items():
            _make_bronze(connection, table, rows)
        yield connection


def _rows(con, sql: str) -> list[dict]:
    return con.execute(sql).df().to_dict("records")


def test_types_are_explicit(con):
    build_all(con)
    types = dict(con.execute(
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_schema = 'silver' AND table_name IN ('orders', 'order_items', 'order_reviews')"
    ).fetchall())
    assert types["purchased_at"] == "TIMESTAMP"
    assert types["price"] == "DECIMAL(10,2)"
    assert types["review_score"] == "INTEGER"


def test_cast_failure_stops_pipeline(con):
    con.execute("UPDATE bronze.order_items SET price = '10,50' WHERE order_id = 'o1'")
    assert count_cast_failures(con) == {"order_items.price": 1}
    with pytest.raises(DataQualityError, match="order_items.price"):
        build_all(con)


def test_order_flags(con):
    build_all(con)
    o2 = _rows(con, "SELECT * FROM silver.orders WHERE order_id = 'o2'")[0]
    assert o2["is_delivered"] is False
    assert o2["flag_delivered_without_date"] is True
    assert o2["flag_carrier_before_purchase"] is True
    o1 = _rows(con, "SELECT * FROM silver.orders WHERE order_id = 'o1'")[0]
    assert o1["is_delivered"] is True and not o1["flag_delivered_without_date"]


def test_review_text_cleaning(con):
    build_all(con)
    texts = dict(con.execute("SELECT review_id, review_pt FROM silver.order_reviews").fetchall())
    assert texts["r1"] == "Muito\nbom produto"  # \r retiré, ligne vide fusionnée, espaces réduits
    assert texts["r2"] is None                  # que des espaces -> NULL
    assert texts["r3"] == "Não recebi o produto"  # retour à la ligne final retiré


def test_review_key_is_review_and_order(con):
    build_all(con)
    n, n_keys = con.execute(
        "SELECT count(*), count(DISTINCT (review_id, order_id)) FROM silver.order_reviews"
    ).fetchone()
    assert n == n_keys == len(REVIEWS)
    flags = dict(con.execute(
        "SELECT review_id, review_shared_across_orders FROM silver.order_reviews WHERE order_id IN ('o1', 'o4')"
    ).fetchall())
    assert flags == {"r1": True, "r4": False}


def test_seller_reviews_attribution(con):
    build_all(con)
    rows = _rows(con, "SELECT review_id, seller_id, order_id FROM silver.seller_reviews ORDER BY review_id")
    assert rows == [
        {"review_id": "r1", "seller_id": "s1", "order_id": "o1"},  # compté une seule fois
        {"review_id": "r2", "seller_id": "s2", "order_id": "o2"},
        {"review_id": "r3", "seller_id": "s2", "order_id": "o2"},  # 2e avis de la même commande : gardé
    ]  # r4 (deux vendeurs) et r5 (sans articles) exclus


def test_product_categories(con):
    build_all(con)
    cats = {r[0]: (r[1], r[2]) for r in con.execute(
        "SELECT product_id, category_pt, category_en FROM silver.products").fetchall()}
    assert cats == {
        "p1": ("bebes", "baby"),
        "p2": ("unknown", "unknown"),
        "p3": ("pc_gamer", "pc_gamer"),  # pas de traduction : nom portugais conservé
    }


def test_quality_report(con):
    build_all(con)
    report = quality_report(con)
    assert report["orders_delivered_without_date"] == 1
    assert report["reviews_shared_across_orders"] == 1
    assert report["orders_with_multiple_reviews"] == 1
    assert report["reviews_not_attributed"] == 2


def test_build_is_idempotent(con):
    first = build_all(con)
    assert build_all(con) == first


# --- Intégration sur les vraies données (ignorée si les CSV sont absents) ---
RAW_DIR = load_settings().raw_dir
HAS_RAW = all((RAW_DIR / f).is_file() for f in TABLES.values())


@pytest.mark.skipif(not HAS_RAW, reason="CSV Olist absents de data/raw")
def test_real_data_silver_invariants():
    with duckdb.connect(":memory:") as con:
        load_all(con, RAW_DIR)
        counts = build_all(con)
        assert counts["orders"] == 99441
        # clés uniques
        assert con.execute(
            "SELECT count(*) = count(DISTINCT (review_id, seller_id)) FROM silver.seller_reviews"
        ).fetchone()[0]
        # un avis partagé a bien le même contenu sur toutes ses commandes (hypothèse de any_value)
        (n_inconsistent,) = con.execute(
            "SELECT count(*) FROM (SELECT review_id FROM silver.order_reviews GROUP BY 1 "
            "HAVING count(DISTINCT (review_score, coalesce(review_pt, ''))) > 1)"
        ).fetchone()
        assert n_inconsistent == 0
