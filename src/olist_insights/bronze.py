"""Couche Bronze : chargement des CSV bruts Olist dans DuckDB, sans transformation.

Principes :
- toutes les colonnes restent en texte (all_varchar) : aucun type deviné, donc aucune
  perte (codes postaux à zéro initial, dates mal formées...). Le typage se fait en Silver ;
- chaque ligne garde sa provenance (_source_file, _loaded_at) ;
- le chargement est idempotent : CREATE OR REPLACE, relancer donne le même résultat.
"""

from __future__ import annotations

import csv
import logging
import re
import sys
from pathlib import Path

import duckdb

from olist_insights.config import load_settings
from olist_insights.logging_utils import setup_logging

logger = logging.getLogger(__name__)

BRONZE_SCHEMA = "bronze"

# nom de table Bronze -> fichier CSV source
TABLES: dict[str, str] = {
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "order_payments": "olist_order_payments_dataset.csv",
    "order_reviews": "olist_order_reviews_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "products": "olist_products_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}

_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")


def _check_identifier(name: str) -> str:
    """Les noms de table sont insérés dans le SQL : on n'accepte que [a-z0-9_]."""
    if not _IDENTIFIER.match(name):
        raise ValueError(f"Nom de table invalide : {name!r}")
    return name


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def load_csv(
    con: duckdb.DuckDBPyConnection,
    csv_path: Path,
    table: str,
    schema: str = BRONZE_SCHEMA,
) -> int:
    """Charge un CSV dans schema.table (remplacée si elle existe). Retourne le nombre de lignes."""
    target = f"{_check_identifier(schema)}.{_check_identifier(table)}"
    con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    con.execute(
        f"""
        CREATE OR REPLACE TABLE {target} AS
        SELECT
            *,
            {_sql_literal(csv_path.name)} AS _source_file,
            current_timestamp AS _loaded_at
        FROM read_csv(
            {_sql_literal(str(csv_path))},
            header = true,
            delim = ',',
            quote = '"',
            escape = '"',
            all_varchar = true
        )
        """
    )
    (n_rows,) = con.execute(f"SELECT count(*) FROM {target}").fetchone()
    return int(n_rows)


def load_all(
    con: duckdb.DuckDBPyConnection,
    raw_dir: Path,
    tables: dict[str, str] = TABLES,
) -> dict[str, int]:
    """Charge tous les CSV attendus. Échoue avant tout chargement si un fichier manque."""
    missing = [f for f in tables.values() if not (raw_dir / f).is_file()]
    if missing:
        raise FileNotFoundError(f"CSV manquants dans {raw_dir} : {', '.join(missing)}")

    counts: dict[str, int] = {}
    for table, filename in tables.items():
        counts[table] = load_csv(con, raw_dir / filename, table)
        logger.info("%-28s <- %-40s %9d lignes", f"{BRONZE_SCHEMA}.{table}", filename, counts[table])
    return counts


def count_csv_records(csv_path: Path) -> int:
    """Compte les enregistrements avec le module csv de Python (contrôle indépendant de DuckDB).

    Un enregistrement peut s'étaler sur plusieurs lignes physiques (retour à la ligne
    dans un champ entre guillemets) : `wc -l` surestime, csv.reader compte juste.
    """
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # en-tête
        return sum(1 for _ in reader)


def find_count_mismatches(
    loaded: dict[str, int],
    raw_dir: Path,
    tables: dict[str, str] = TABLES,
) -> dict[str, tuple[int, int]]:
    """Retourne {table: (lignes DuckDB, enregistrements CSV)} pour les tables qui diffèrent."""
    mismatches = {}
    for table, filename in tables.items():
        expected = count_csv_records(raw_dir / filename)
        if loaded[table] != expected:
            mismatches[table] = (loaded[table], expected)
    return mismatches


def main() -> int:
    settings = load_settings()
    setup_logging(settings.log_level)
    settings.ensure_dirs()

    logger.info("Base DuckDB : %s", settings.duckdb_path)
    with duckdb.connect(str(settings.duckdb_path)) as con:
        counts = load_all(con, settings.raw_dir)

    mismatches = find_count_mismatches(counts, settings.raw_dir)
    if mismatches:
        for table, (got, expected) in mismatches.items():
            logger.error("%s : %d lignes chargées, %d attendues", table, got, expected)
        return 1
    logger.info("Bronze OK : %d tables, comptages identiques au module csv", len(counts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
