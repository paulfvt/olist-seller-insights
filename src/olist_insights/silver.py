"""Couche Silver : tables typées et nettoyées, construites depuis Bronze.

Règles (issues du profilage des données) :
- typage explicite. On vérifie d'abord qu'aucune valeur ne résiste à la conversion
  (TRY_CAST + comptage), puis on convertit avec CAST : une valeur imprévue fait
  échouer le pipeline au lieu de devenir NULL en silence ;
- pas d'imputation : les valeurs manquantes restent NULL, les incohérences sont
  signalées par des colonnes flag_*, jamais supprimées ;
- avis : la clé est (review_id, order_id), car un même avis peut couvrir plusieurs
  commandes passées ensemble ;
- attribution avis -> vendeur : uniquement pour les commandes à vendeur unique, et
  un avis compte une seule fois par vendeur.
"""

from __future__ import annotations

import logging
import sys

import duckdb

from olist_insights.config import load_settings
from olist_insights.logging_utils import setup_logging

logger = logging.getLogger(__name__)

SILVER_SCHEMA = "silver"


class DataQualityError(Exception):
    """Données Bronze incompatibles avec les règles Silver."""


# (table bronze, colonne, type cible) : conversions à contrôler avant de construire Silver
CASTS: list[tuple[str, str, str]] = [
    ("orders", "order_purchase_timestamp", "TIMESTAMP"),
    ("orders", "order_approved_at", "TIMESTAMP"),
    ("orders", "order_delivered_carrier_date", "TIMESTAMP"),
    ("orders", "order_delivered_customer_date", "TIMESTAMP"),
    ("orders", "order_estimated_delivery_date", "TIMESTAMP"),
    ("order_items", "order_item_id", "INTEGER"),
    ("order_items", "shipping_limit_date", "TIMESTAMP"),
    ("order_items", "price", "DECIMAL(10,2)"),
    ("order_items", "freight_value", "DECIMAL(10,2)"),
    ("order_reviews", "review_score", "INTEGER"),
    ("order_reviews", "review_creation_date", "TIMESTAMP"),
    ("order_reviews", "review_answer_timestamp", "TIMESTAMP"),
    ("order_payments", "payment_sequential", "INTEGER"),
    ("order_payments", "payment_installments", "INTEGER"),
    ("order_payments", "payment_value", "DECIMAL(10,2)"),
    ("products", "product_photos_qty", "INTEGER"),
    ("products", "product_weight_g", "INTEGER"),
]


def count_cast_failures(con: duckdb.DuckDBPyConnection, casts: list[tuple[str, str, str]] = CASTS) -> dict[str, int]:
    """Nombre de valeurs non nulles qui ne se convertissent pas, par 'table.colonne'."""
    failures = {}
    for table, column, sql_type in casts:
        (n,) = con.execute(
            f"SELECT count(*) FROM bronze.{table} "
            f"WHERE {column} IS NOT NULL AND TRY_CAST({column} AS {sql_type}) IS NULL"
        ).fetchone()
        if n:
            failures[f"{table}.{column}"] = int(n)
    return failures


def _define_macros(con: duckdb.DuckDBPyConnection) -> None:
    # Nettoyage de texte : retire les \r (fins de ligne Windows), fusionne les lignes vides
    # et les espaces autour des retours à la ligne, réduit les espaces multiples,
    # retire les blancs en début et fin (y compris \n : trim() de DuckDB ne retire que
    # les espaces) et transforme une chaîne vide en NULL.
    con.execute(
        r"""
        CREATE OR REPLACE TEMP MACRO clean_text(x) AS
            NULLIF(regexp_replace(regexp_replace(regexp_replace(replace(x, chr(13), ''),
                '\s*\n\s*', chr(10), 'g'), '[ \t]+', ' ', 'g'), '^\s+|\s+$', '', 'g'), '')
        """
    )


def build_orders(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE OR REPLACE TABLE silver.orders AS
        SELECT
            order_id,
            customer_id,
            order_status,
            CAST(order_purchase_timestamp AS TIMESTAMP)      AS purchased_at,
            CAST(order_approved_at AS TIMESTAMP)             AS approved_at,
            CAST(order_delivered_carrier_date AS TIMESTAMP)  AS delivered_carrier_at,
            CAST(order_delivered_customer_date AS TIMESTAMP) AS delivered_customer_at,
            CAST(order_estimated_delivery_date AS TIMESTAMP) AS estimated_delivery_at,
            (order_status = 'delivered' AND order_delivered_customer_date IS NOT NULL) AS is_delivered,
            (order_status = 'delivered' AND order_delivered_customer_date IS NULL)     AS flag_delivered_without_date,
            coalesce(CAST(order_delivered_carrier_date AS TIMESTAMP)
                     < CAST(order_purchase_timestamp AS TIMESTAMP), false)             AS flag_carrier_before_purchase,
            coalesce(CAST(order_delivered_customer_date AS TIMESTAMP)
                     < CAST(order_delivered_carrier_date AS TIMESTAMP), false)         AS flag_delivered_before_carrier
        FROM bronze.orders
        """
    )


def build_order_items(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE OR REPLACE TABLE silver.order_items AS
        SELECT
            order_id,
            CAST(order_item_id AS INTEGER)          AS order_item_id,
            product_id,
            seller_id,
            CAST(shipping_limit_date AS TIMESTAMP)  AS shipping_limit_at,
            CAST(price AS DECIMAL(10,2))            AS price,
            CAST(freight_value AS DECIMAL(10,2))    AS freight_value
        FROM bronze.order_items
        """
    )


def build_order_payments(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE OR REPLACE TABLE silver.order_payments AS
        SELECT
            order_id,
            CAST(payment_sequential AS INTEGER)    AS payment_sequential,
            payment_type,
            CAST(payment_installments AS INTEGER)  AS payment_installments,
            CAST(payment_value AS DECIMAL(10,2))   AS payment_value
        FROM bronze.order_payments
        """
    )


def build_order_reviews(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE OR REPLACE TABLE silver.order_reviews AS
        SELECT
            review_id,
            order_id,
            CAST(review_score AS INTEGER)              AS review_score,
            clean_text(review_comment_title)           AS review_title,
            clean_text(review_comment_message)         AS review_pt,
            clean_text(review_comment_message) IS NOT NULL AS has_text,
            CAST(review_creation_date AS TIMESTAMP)    AS review_created_at,
            CAST(review_answer_timestamp AS TIMESTAMP) AS review_answered_at,
            count(*) OVER (PARTITION BY order_id) > 1  AS order_has_multiple_reviews,
            count(*) OVER (PARTITION BY review_id) > 1 AS review_shared_across_orders
        FROM bronze.order_reviews
        """
    )


def build_customers(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE OR REPLACE TABLE silver.customers AS
        SELECT customer_id, customer_unique_id,
               customer_zip_code_prefix AS zip_code_prefix,
               customer_city AS city, customer_state AS state
        FROM bronze.customers
        """
    )


def build_sellers(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE OR REPLACE TABLE silver.sellers AS
        SELECT seller_id,
               seller_zip_code_prefix AS zip_code_prefix,
               seller_city AS city, seller_state AS state
        FROM bronze.sellers
        """
    )


def build_products(con: duckdb.DuckDBPyConnection) -> None:
    # Catégorie absente -> 'unknown' ; catégorie sans traduction -> nom portugais conservé.
    con.execute(
        """
        CREATE OR REPLACE TABLE silver.products AS
        SELECT
            p.product_id,
            coalesce(p.product_category_name, 'unknown')                              AS category_pt,
            coalesce(t.product_category_name_english, p.product_category_name, 'unknown') AS category_en,
            CAST(p.product_photos_qty AS INTEGER) AS photos_qty,
            CAST(p.product_weight_g AS INTEGER)   AS weight_g
        FROM bronze.products p
        LEFT JOIN bronze.category_translation t USING (product_category_name)
        """
    )


def build_seller_reviews(con: duckdb.DuckDBPyConnection) -> None:
    """Avis rattachés à un vendeur : une ligne par (review_id, seller_id).

    - seules les commandes à vendeur unique sont gardées (sinon on ne sait pas qui est noté) ;
    - un avis partagé par plusieurs commandes du même vendeur ne compte qu'une fois.
    """
    con.execute(
        """
        CREATE OR REPLACE TABLE silver.seller_reviews AS
        WITH single_seller_orders AS (
            SELECT order_id, any_value(seller_id) AS seller_id
            FROM silver.order_items
            GROUP BY order_id
            HAVING count(DISTINCT seller_id) = 1
        )
        SELECT
            r.review_id,
            s.seller_id,
            min(r.order_id)            AS order_id,
            any_value(r.review_score)  AS review_score,
            any_value(r.review_pt)     AS review_pt,
            any_value(r.has_text)      AS has_text,
            any_value(r.review_created_at) AS review_created_at
        FROM silver.order_reviews r
        JOIN single_seller_orders s USING (order_id)
        GROUP BY r.review_id, s.seller_id
        """
    )


BUILDERS = [
    build_orders,
    build_order_items,
    build_order_payments,
    build_order_reviews,
    build_customers,
    build_sellers,
    build_products,
    build_seller_reviews,  # en dernier : dépend de order_items et order_reviews
]


def build_all(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Construit toutes les tables Silver. Échoue si une conversion de type est impossible."""
    failures = count_cast_failures(con)
    if failures:
        raise DataQualityError(f"Conversions impossibles : {failures}")

    con.execute(f"CREATE SCHEMA IF NOT EXISTS {SILVER_SCHEMA}")
    _define_macros(con)
    counts = {}
    for builder in BUILDERS:
        builder(con)
        table = builder.__name__.removeprefix("build_")
        (counts[table],) = con.execute(f"SELECT count(*) FROM silver.{table}").fetchone()
    return counts


def quality_report(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Indicateurs de qualité à surveiller (signalés, pas corrigés)."""
    (report,) = con.execute(
        """
        SELECT {
            'orders_delivered_without_date':   (SELECT count(*) FROM silver.orders WHERE flag_delivered_without_date),
            'orders_carrier_before_purchase':  (SELECT count(*) FROM silver.orders WHERE flag_carrier_before_purchase),
            'orders_delivered_before_carrier': (SELECT count(*) FROM silver.orders WHERE flag_delivered_before_carrier),
            'reviews_shared_across_orders':    (SELECT count(DISTINCT review_id) FROM silver.order_reviews WHERE review_shared_across_orders),
            'orders_with_multiple_reviews':    (SELECT count(DISTINCT order_id) FROM silver.order_reviews WHERE order_has_multiple_reviews),
            'reviews_with_text':               (SELECT count(*) FROM silver.order_reviews WHERE has_text),
            'reviews_not_attributed':          (SELECT count(DISTINCT review_id) FROM silver.order_reviews)
                                               - (SELECT count(DISTINCT review_id) FROM silver.seller_reviews),
            'seller_reviews_with_text':        (SELECT count(*) FROM silver.seller_reviews WHERE has_text),
            'products_unknown_category':       (SELECT count(*) FROM silver.products WHERE category_pt = 'unknown')
        }
        """
    ).fetchone()
    return {k: int(v) for k, v in report.items()}


def main() -> int:
    settings = load_settings()
    setup_logging(settings.log_level)
    with duckdb.connect(str(settings.duckdb_path)) as con:
        try:
            counts = build_all(con)
        except DataQualityError as exc:
            logger.error("%s", exc)
            return 1
        for table, n in counts.items():
            logger.info("silver.%-16s %9d lignes", table, n)
        for name, n in quality_report(con).items():
            logger.info("qualité | %-33s %7d", name, n)
    logger.info("Silver OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
