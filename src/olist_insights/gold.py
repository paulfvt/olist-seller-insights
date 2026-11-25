"""Couche Gold : indicateurs (KPI) par vendeur, construits depuis Silver.

Deux tables :
- gold.seller_kpis    : une ligne par vendeur, sur toute la période ;
- gold.seller_monthly : une ligne par (vendeur, mois), pour suivre l'évolution.

Définitions (choisies pour être justes envers le vendeur) :
- chiffre d'affaires = somme des prix des articles du vendeur (hors frais de port, qui
  rémunèrent le transporteur). payment_value n'est pas utilisable : il est par commande,
  et une commande à plusieurs vendeurs ne se répartit pas ;
- commande annulée = statut 'canceled' ou 'unavailable' (produit indisponible) : exclue
  du chiffre d'affaires, comptée dans le taux d'annulation ;
- retard client = livrée après la date estimée, comparaison au JOUR (la date estimée
  n'a pas d'heure). Calculé uniquement sur les commandes livrées ;
- remise tardive au transporteur = remise après shipping_limit_at : c'est le retard
  imputable au vendeur, contrairement au retard de livraison qui peut venir du transporteur ;
- notes : silver.seller_reviews (commandes à vendeur unique, un avis compté une fois),
  datées par la date de l'avis, c'est-à-dire quand l'information devient connue.
Chaque moyenne est accompagnée de son effectif : une note moyenne sur 2 avis ne veut
rien dire, c'est à l'utilisateur de la table de fixer un seuil.
"""

from __future__ import annotations

import logging
import sys

import duckdb

from olist_insights.config import load_settings
from olist_insights.logging_utils import setup_logging

logger = logging.getLogger(__name__)

GOLD_SCHEMA = "gold"
CANCELED_STATUSES = ("canceled", "unavailable")

# Une ligne par (vendeur, commande) : une commande à plusieurs vendeurs compte pour
# chacun d'eux, avec seulement ses propres articles.
_SELLER_ORDERS = f"""
    WITH items AS (
        SELECT seller_id, order_id,
               count(*)               AS n_items,
               sum(price)             AS revenue,
               sum(freight_value)     AS freight,
               max(shipping_limit_at) AS shipping_limit_at
        FROM silver.order_items
        GROUP BY seller_id, order_id
    )
    SELECT
        i.*,
        o.order_status,
        o.purchased_at,
        o.order_status IN {CANCELED_STATUSES}               AS is_canceled,
        o.is_delivered,
        o.delivered_customer_at::DATE > o.estimated_delivery_at::DATE AS is_late,
        date_diff('day', o.estimated_delivery_at::DATE,
                  o.delivered_customer_at::DATE)            AS delay_days,
        date_diff('hour', o.purchased_at, o.delivered_customer_at) / 24.0 AS delivery_days,
        o.delivered_carrier_at > i.shipping_limit_at        AS is_late_handoff
    FROM items i
    JOIN silver.orders o USING (order_id)
"""


def build_seller_kpis(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        f"""
        CREATE OR REPLACE TABLE gold.seller_kpis AS
        WITH so AS ({_SELLER_ORDERS}),
        sales AS (
            SELECT
                seller_id,
                count(*)                                         AS n_orders,
                count(*) FILTER (is_canceled)                    AS n_canceled,
                sum(n_items) FILTER (NOT is_canceled)            AS n_items,
                coalesce(sum(revenue) FILTER (NOT is_canceled), 0) AS revenue,
                coalesce(sum(freight) FILTER (NOT is_canceled), 0) AS freight,
                count(*) FILTER (is_delivered)                   AS n_delivered,
                avg(is_late::INT) FILTER (is_delivered)          AS late_rate,
                avg(delay_days) FILTER (is_delivered)            AS avg_delay_days,
                avg(delivery_days) FILTER (is_delivered)         AS avg_delivery_days,
                avg(is_late_handoff::INT)                        AS late_handoff_rate,
                min(purchased_at)                                AS first_order_at,
                max(purchased_at)                                AS last_order_at
            FROM so
            GROUP BY seller_id
        ),
        ratings AS (
            SELECT
                seller_id,
                count(*)                                AS n_reviews,
                avg(review_score)                       AS avg_rating,
                avg((review_score <= 2)::INT)           AS low_rating_rate,
                count(*) FILTER (has_text)              AS n_reviews_with_text
            FROM silver.seller_reviews
            GROUP BY seller_id
        ),
        main_category AS (
            -- catégorie qui regroupe le plus d'articles vendus ; égalité -> ordre alphabétique
            -- (arg_max départagerait au hasard : résultat non reproductible)
            SELECT seller_id, first(category_en ORDER BY n DESC, category_en) AS main_category,
                   count(*) AS n_categories
            FROM (
                SELECT i.seller_id, p.category_en, count(*) AS n
                FROM silver.order_items i JOIN silver.products p USING (product_id)
                GROUP BY ALL
            )
            GROUP BY seller_id
        )
        SELECT
            s.seller_id,
            sl.state,
            sl.city,
            mc.main_category,
            mc.n_categories,
            s.n_orders,
            s.n_orders - s.n_canceled                       AS n_orders_valid,
            s.n_canceled::DOUBLE / s.n_orders               AS cancel_rate,
            coalesce(s.n_items, 0)                          AS n_items,
            s.revenue,
            s.revenue / nullif(s.n_orders - s.n_canceled, 0) AS avg_order_value,
            s.freight / nullif(s.revenue, 0)                AS freight_ratio,
            s.n_delivered,
            s.late_rate,
            s.avg_delay_days,
            s.avg_delivery_days,
            s.late_handoff_rate,
            coalesce(r.n_reviews, 0)                        AS n_reviews,
            r.avg_rating,
            r.low_rating_rate,
            coalesce(r.n_reviews_with_text, 0)              AS n_reviews_with_text,
            s.first_order_at,
            s.last_order_at
        FROM sales s
        LEFT JOIN silver.sellers sl USING (seller_id)
        LEFT JOIN ratings r USING (seller_id)
        LEFT JOIN main_category mc USING (seller_id)
        ORDER BY s.revenue DESC
        """
    )


def build_seller_monthly(con: duckdb.DuckDBPyConnection) -> None:
    """Une ligne par (vendeur, mois) où le vendeur a vendu OU reçu un avis.

    Ventes et livraisons sont rangées au mois d'achat ; les notes au mois de l'avis.
    Un mois sans avis a avg_rating NULL (pas 0) : absence d'information, pas mauvaise note.
    """
    con.execute(
        f"""
        CREATE OR REPLACE TABLE gold.seller_monthly AS
        WITH so AS ({_SELLER_ORDERS}),
        sales AS (
            SELECT
                seller_id,
                date_trunc('month', purchased_at)::DATE           AS month,
                count(*)                                          AS n_orders,
                count(*) FILTER (is_canceled)                     AS n_canceled,
                coalesce(sum(revenue) FILTER (NOT is_canceled), 0) AS revenue,
                count(*) FILTER (is_delivered)                    AS n_delivered,
                avg(is_late::INT) FILTER (is_delivered)           AS late_rate
            FROM so
            GROUP BY ALL
        ),
        ratings AS (
            SELECT
                seller_id,
                date_trunc('month', review_created_at)::DATE      AS month,
                count(*)                                          AS n_reviews,
                avg(review_score)                                 AS avg_rating,
                avg((review_score <= 2)::INT)                     AS low_rating_rate
            FROM silver.seller_reviews
            GROUP BY ALL
        )
        SELECT
            seller_id,
            month,
            coalesce(s.n_orders, 0)    AS n_orders,
            coalesce(s.n_canceled, 0)  AS n_canceled,
            coalesce(s.revenue, 0)     AS revenue,
            coalesce(s.n_delivered, 0) AS n_delivered,
            s.late_rate,
            coalesce(r.n_reviews, 0)   AS n_reviews,
            r.avg_rating,
            r.low_rating_rate
        FROM sales s
        FULL OUTER JOIN ratings r USING (seller_id, month)
        ORDER BY seller_id, month
        """
    )


BUILDERS = [build_seller_kpis, build_seller_monthly]


def build_all(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Construit les tables Gold. Retourne le nombre de lignes de chacune."""
    con.execute(f"CREATE SCHEMA IF NOT EXISTS {GOLD_SCHEMA}")
    counts = {}
    for builder in BUILDERS:
        builder(con)
        table = builder.__name__.removeprefix("build_")
        (counts[table],) = con.execute(f"SELECT count(*) FROM gold.{table}").fetchone()
    return counts


def consistency_checks(con: duckdb.DuckDBPyConnection) -> dict[str, bool]:
    """Invariants qui doivent toujours être vrais : Gold réconcilié avec Silver."""
    (checks,) = con.execute(
        f"""
        SELECT {{
            'one_row_per_seller':
                (SELECT count(*) = count(DISTINCT seller_id) FROM gold.seller_kpis),
            'all_sellers_with_sales':
                (SELECT count(*) FROM gold.seller_kpis)
                = (SELECT count(DISTINCT seller_id) FROM silver.order_items),
            'revenue_matches_silver':
                (SELECT sum(revenue) FROM gold.seller_kpis)
                = (SELECT sum(i.price) FROM silver.order_items i JOIN silver.orders o USING (order_id)
                   WHERE o.order_status NOT IN {CANCELED_STATUSES}),
            'monthly_revenue_matches_kpis':
                (SELECT sum(revenue) FROM gold.seller_monthly) = (SELECT sum(revenue) FROM gold.seller_kpis),
            'monthly_reviews_match_kpis':
                (SELECT sum(n_reviews) FROM gold.seller_monthly) = (SELECT sum(n_reviews) FROM gold.seller_kpis),
            'rates_between_0_and_1':
                (SELECT bool_and(coalesce(late_rate, 0) BETWEEN 0 AND 1
                             AND cancel_rate BETWEEN 0 AND 1
                             AND coalesce(low_rating_rate, 0) BETWEEN 0 AND 1) FROM gold.seller_kpis)
        }}
        """
    ).fetchone()
    return dict(checks)


def main() -> int:
    settings = load_settings()
    setup_logging(settings.log_level)
    with duckdb.connect(str(settings.duckdb_path)) as con:
        for table, n in build_all(con).items():
            logger.info("gold.%-16s %7d lignes", table, n)
        checks = consistency_checks(con)
    for name, ok in checks.items():
        logger.log(logging.INFO if ok else logging.ERROR, "contrôle | %-30s %s", name, "OK" if ok else "ÉCHEC")
    if not all(checks.values()):
        return 1
    logger.info("Gold OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
