"""Données Silver minimales partagées par les tests Gold et features."""

import duckdb
import pytest

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
def silver_con():
    with duckdb.connect(":memory:") as connection:
        connection.execute(SILVER_SQL)
        yield connection
