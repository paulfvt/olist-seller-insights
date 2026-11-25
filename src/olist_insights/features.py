"""Jeu de données du modèle : instantanés (vendeur, date de coupure T) -> features + cible.

Règle d'or contre la fuite de données (data leakage)
----------------------------------------------------
Un modèle évalué avec des informations qu'il n'aurait pas eues au moment de prédire
paraît excellent et ne vaut rien en production. Donc :
- build_features ne lit QUE des événements datés strictement avant T : avis écrits
  avant T, livraisons constatées avant T, remises au transporteur avant T ;
- build_labels ne lit QUE la fenêtre [T, T + horizon) ;
- les deux sont des fonctions séparées, et un test vérifie qu'ajouter des événements
  après T ne change aucune feature.
Conséquence : pas de features tirées des tables Gold (calculées sur toute la période),
et pas de taux d'annulation (le statut final d'une commande n'est pas daté : une
commande passée avant T peut être annulée après T).

Cible (option B, choisie à l'étape 4)
-------------------------------------
label = 1 si, sur les `horizon_days` jours après T, au moins 25 % des avis du vendeur
sont des notes 1 ou 2 (« un client sur quatre mécontent »), calculé seulement si le
vendeur a reçu au moins `min_future_reviews` avis sur la fenêtre (en dessous, une
proportion n'a pas de sens). Population : vendeurs ayant au moins `min_past_reviews`
avis dans les `lookback_days` jours avant T.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import duckdb
import pandas as pd


@dataclass(frozen=True)
class TargetConfig:
    lookback_days: int = 90       # fenêtre « passé récent » des features
    horizon_days: int = 90        # fenêtre de la cible
    min_past_reviews: int = 5     # population : vendeur assez actif avant T
    min_future_reviews: int = 5   # cible calculable seulement avec assez d'avis futurs
    bad_threshold: float = 0.25   # part de notes 1-2 qui définit un « mauvais épisode »


NUMERIC_FEATURES = [
    # satisfaction passée
    "n_reviews_90d", "avg_rating_90d", "low_rate_90d", "text_rate_90d",
    "n_reviews_30d", "avg_rating_30d", "rating_trend",
    "n_reviews_hist", "avg_rating_hist", "low_rate_hist",
    # logistique
    "n_delivered_90d", "late_rate_90d", "avg_delay_days_90d", "avg_delivery_days_90d",
    "late_handoff_rate_90d", "overdue_open_rate",
    # activité
    "n_orders_90d", "n_orders_30d", "recent_share_30d", "revenue_90d",
    "avg_order_value_90d", "freight_ratio_90d",
    # profil
    "seniority_days", "n_categories_hist",
]
CATEGORICAL_FEATURES = ["main_category", "state"]


def _cutoffs_sql() -> str:
    return "(SELECT unnest(?::DATE[]) AS t)"


def build_features(
    con: duckdb.DuckDBPyConnection,
    cutoffs: list[dt.date],
    cfg: TargetConfig = TargetConfig(),
) -> pd.DataFrame:
    """Features au jour T pour les vendeurs de la population. Une ligne par (seller_id, cutoff).

    Toutes les conditions de date sont « < c.t » : rien de ce qui arrive le jour T ou
    après n'est visible.
    """
    lb = int(cfg.lookback_days)
    sql = f"""
    WITH cut AS {_cutoffs_sql()},
    seller_orders AS (
        SELECT i.seller_id, i.order_id,
               sum(i.price) AS revenue, sum(i.freight_value) AS freight,
               max(i.shipping_limit_at) AS shipping_limit_at,
               any_value(o.purchased_at) AS purchased_at,
               any_value(o.estimated_delivery_at) AS estimated_delivery_at,
               any_value(o.delivered_customer_at) AS delivered_customer_at,
               any_value(o.delivered_carrier_at) AS delivered_carrier_at
        FROM silver.order_items i JOIN silver.orders o USING (order_id)
        GROUP BY i.seller_id, i.order_id
    ),
    rev AS (
        SELECT c.t, r.seller_id,
            count(*) FILTER (r.review_created_at >= c.t - INTERVAL {lb} DAY)              AS n_reviews_90d,
            avg(r.review_score) FILTER (r.review_created_at >= c.t - INTERVAL {lb} DAY)   AS avg_rating_90d,
            avg((r.review_score <= 2)::INT) FILTER (r.review_created_at >= c.t - INTERVAL {lb} DAY) AS low_rate_90d,
            avg(r.has_text::INT) FILTER (r.review_created_at >= c.t - INTERVAL {lb} DAY)  AS text_rate_90d,
            count(*) FILTER (r.review_created_at >= c.t - INTERVAL 30 DAY)                AS n_reviews_30d,
            avg(r.review_score) FILTER (r.review_created_at >= c.t - INTERVAL 30 DAY)     AS avg_rating_30d,
            count(*)                                   AS n_reviews_hist,
            avg(r.review_score)                        AS avg_rating_hist,
            avg((r.review_score <= 2)::INT)            AS low_rate_hist
        FROM cut c JOIN silver.seller_reviews r ON r.review_created_at < c.t
        GROUP BY ALL
    ),
    sales AS (
        -- commandes passées dans la fenêtre (le statut final n'est pas utilisé : inconnu à T)
        SELECT c.t, so.seller_id,
            count(*)                                                      AS n_orders_90d,
            count(*) FILTER (so.purchased_at >= c.t - INTERVAL 30 DAY)    AS n_orders_30d,
            sum(so.revenue)                                               AS revenue_90d,
            sum(so.freight) / nullif(sum(so.revenue), 0)                  AS freight_ratio_90d,
            -- commandes dont la date estimée est passée à T sans livraison constatée avant T
            avg((so.estimated_delivery_at < c.t
                 AND (so.delivered_customer_at IS NULL OR so.delivered_customer_at >= c.t))::INT) AS overdue_open_rate
        FROM cut c JOIN seller_orders so
          ON so.purchased_at < c.t AND so.purchased_at >= c.t - INTERVAL {lb} DAY
        GROUP BY ALL
    ),
    deliv AS (
        -- livraisons CONSTATÉES dans la fenêtre (date de livraison < T)
        SELECT c.t, so.seller_id,
            count(*)                                                                    AS n_delivered_90d,
            avg((so.delivered_customer_at::DATE > so.estimated_delivery_at::DATE)::INT)  AS late_rate_90d,
            avg(date_diff('day', so.estimated_delivery_at::DATE, so.delivered_customer_at::DATE)) AS avg_delay_days_90d,
            avg(date_diff('hour', so.purchased_at, so.delivered_customer_at) / 24.0)    AS avg_delivery_days_90d
        FROM cut c JOIN seller_orders so
          ON so.delivered_customer_at < c.t AND so.delivered_customer_at >= c.t - INTERVAL {lb} DAY
        GROUP BY ALL
    ),
    handoff AS (
        SELECT c.t, so.seller_id,
            avg((so.delivered_carrier_at > so.shipping_limit_at)::INT) AS late_handoff_rate_90d
        FROM cut c JOIN seller_orders so
          ON so.delivered_carrier_at < c.t AND so.delivered_carrier_at >= c.t - INTERVAL {lb} DAY
        GROUP BY ALL
    ),
    profile AS (
        -- catégorie principale = la plus vendue ; en cas d'égalité, ordre alphabétique.
        -- (mode() départagerait au hasard selon l'exécution parallèle : non reproductible)
        SELECT t, seller_id,
            date_diff('day', min(first_order_at)::DATE, t)            AS seniority_days,
            count(*)                                                  AS n_categories_hist,
            first(category_en ORDER BY n_items DESC, category_en)     AS main_category
        FROM (
            SELECT c.t, i.seller_id, p.category_en,
                count(*)            AS n_items,
                min(o.purchased_at) AS first_order_at
            FROM cut c
            JOIN silver.orders o ON o.purchased_at < c.t
            JOIN silver.order_items i USING (order_id)
            JOIN silver.products p USING (product_id)
            GROUP BY c.t, i.seller_id, p.category_en
        )
        GROUP BY t, seller_id
    )
    SELECT
        rev.seller_id,
        rev.t AS cutoff,
        rev.* EXCLUDE (t, seller_id),
        rev.avg_rating_30d - rev.avg_rating_90d                  AS rating_trend,
        coalesce(deliv.n_delivered_90d, 0)                       AS n_delivered_90d,
        deliv.late_rate_90d, deliv.avg_delay_days_90d, deliv.avg_delivery_days_90d,
        handoff.late_handoff_rate_90d,
        sales.overdue_open_rate,
        coalesce(sales.n_orders_90d, 0)                          AS n_orders_90d,
        coalesce(sales.n_orders_30d, 0)                          AS n_orders_30d,
        sales.n_orders_30d / nullif(sales.n_orders_90d, 0)       AS recent_share_30d,
        coalesce(sales.revenue_90d, 0)                           AS revenue_90d,
        sales.revenue_90d / nullif(sales.n_orders_90d, 0)        AS avg_order_value_90d,
        sales.freight_ratio_90d,
        profile.seniority_days, profile.n_categories_hist, profile.main_category,
        s.state
    FROM rev
    LEFT JOIN sales   USING (t, seller_id)
    LEFT JOIN deliv   USING (t, seller_id)
    LEFT JOIN handoff USING (t, seller_id)
    LEFT JOIN profile USING (t, seller_id)
    LEFT JOIN silver.sellers s USING (seller_id)
    WHERE rev.n_reviews_90d >= ?
    ORDER BY cutoff, seller_id
    """
    df = con.execute(sql, [list(cutoffs), cfg.min_past_reviews]).df()
    df["cutoff"] = pd.to_datetime(df["cutoff"]).dt.date
    for col in NUMERIC_FEATURES:
        df[col] = df[col].astype("float64")  # DECIMAL -> float, NULL -> NaN
    return df[["seller_id", "cutoff", *NUMERIC_FEATURES, *CATEGORICAL_FEATURES]]


def build_labels(
    con: duckdb.DuckDBPyConnection,
    cutoffs: list[dt.date],
    cfg: TargetConfig = TargetConfig(),
) -> pd.DataFrame:
    """Cible sur la fenêtre [T, T + horizon). Seuls les vendeurs avec assez d'avis futurs."""
    df = con.execute(
        f"""
        SELECT r.seller_id, c.t AS cutoff,
               count(*)                        AS n_reviews_future,
               avg((r.review_score <= 2)::INT) AS low_rate_future
        FROM {_cutoffs_sql()} c
        JOIN silver.seller_reviews r
          ON r.review_created_at >= c.t AND r.review_created_at < c.t + INTERVAL {int(cfg.horizon_days)} DAY
        GROUP BY ALL
        HAVING count(*) >= ?
        """,
        [list(cutoffs), cfg.min_future_reviews],
    ).df()
    df["cutoff"] = pd.to_datetime(df["cutoff"]).dt.date
    df["label"] = (df["low_rate_future"] >= cfg.bad_threshold).astype(int)
    return df


def build_dataset(
    con: duckdb.DuckDBPyConnection,
    cutoffs: list[dt.date],
    cfg: TargetConfig = TargetConfig(),
) -> pd.DataFrame:
    """Features + cible. Les vendeurs de la population sans assez d'avis futurs sont écartés
    (cible incalculable) : biais de sélection connu, documenté dans le README."""
    features = build_features(con, cutoffs, cfg)
    labels = build_labels(con, cutoffs, cfg)
    return features.merge(labels, on=["seller_id", "cutoff"], how="inner")


def month_starts(start: str, end: str) -> list[dt.date]:
    """Premiers jours de mois de start à end inclus ('2017-03' -> 2017-03-01)."""
    return [d.date() for d in pd.date_range(f"{start}-01", f"{end}-01", freq="MS")]
