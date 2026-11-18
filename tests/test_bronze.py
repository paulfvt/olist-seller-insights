from pathlib import Path

import duckdb
import pytest

from olist_insights.bronze import (
    TABLES,
    count_csv_records,
    find_count_mismatches,
    load_all,
    load_csv,
)
from olist_insights.config import load_settings

REVIEWS_CSV = (
    '"review_id","order_id","review_score","review_comment_message"\n'
    '"r1","o1",5,"Ótimo produto"\n'
    '"r2","o2",1,"Não chegou.\nPéssimo, ""nunca mais"""\n'  # retour à la ligne + guillemets échappés
    '"r3","o3",4,\n'  # commentaire vide
)


@pytest.fixture
def con():
    with duckdb.connect(":memory:") as connection:
        yield connection


@pytest.fixture
def reviews_csv(tmp_path) -> Path:
    path = tmp_path / "reviews.csv"
    path.write_text(REVIEWS_CSV, encoding="utf-8")
    return path


def test_multiline_field_is_one_record(con, reviews_csv):
    assert load_csv(con, reviews_csv, "reviews") == 3
    assert count_csv_records(reviews_csv) == 3
    (msg,) = con.execute("SELECT review_comment_message FROM bronze.reviews WHERE review_id = 'r2'").fetchone()
    assert msg == 'Não chegou.\nPéssimo, "nunca mais"'


def test_all_columns_stay_text(con, tmp_path):
    path = tmp_path / "sellers.csv"
    path.write_text('"seller_id","seller_zip_code_prefix"\n"s1","01310"\n', encoding="utf-8")
    load_csv(con, path, "sellers")
    (zip_code,) = con.execute("SELECT seller_zip_code_prefix FROM bronze.sellers").fetchone()
    assert zip_code == "01310"  # le zéro initial n'est pas perdu
    types = {r[1] for r in con.execute("DESCRIBE bronze.sellers").fetchall() if not r[0].startswith("_")}
    assert types == {"VARCHAR"}


def test_lineage_columns(con, reviews_csv):
    load_csv(con, reviews_csv, "reviews")
    (src,) = con.execute("SELECT DISTINCT _source_file FROM bronze.reviews").fetchone()
    assert src == "reviews.csv"


def test_utf8_bom_header_is_clean(con, tmp_path):
    path = tmp_path / "translation.csv"
    path.write_text("product_category_name,product_category_name_english\nbebes,baby\n", encoding="utf-8-sig")
    load_csv(con, path, "translation")
    cols = [r[0] for r in con.execute("DESCRIBE bronze.translation").fetchall()]
    assert cols[0] == "product_category_name"


def test_reload_is_idempotent(con, reviews_csv):
    load_csv(con, reviews_csv, "reviews")
    assert load_csv(con, reviews_csv, "reviews") == 3


def test_invalid_table_name_rejected(con, reviews_csv):
    with pytest.raises(ValueError):
        load_csv(con, reviews_csv, "reviews; DROP TABLE x")


def test_load_all_fails_fast_on_missing_file(con, reviews_csv):
    tables = {"reviews": reviews_csv.name, "orders": "absent.csv"}
    with pytest.raises(FileNotFoundError, match="absent.csv"):
        load_all(con, reviews_csv.parent, tables)
    # rien n'a été chargé
    assert con.execute("SELECT count(*) FROM information_schema.tables").fetchone()[0] == 0


def test_mismatch_detection(reviews_csv):
    tables = {"reviews": reviews_csv.name}
    assert find_count_mismatches({"reviews": 3}, reviews_csv.parent, tables) == {}
    assert find_count_mismatches({"reviews": 4}, reviews_csv.parent, tables) == {"reviews": (4, 3)}


# --- Test d'intégration sur les vrais CSV (ignoré s'ils ne sont pas présents) ---
RAW_DIR = load_settings().raw_dir
HAS_RAW = all((RAW_DIR / f).is_file() for f in TABLES.values())


@pytest.mark.skipif(not HAS_RAW, reason="CSV Olist absents de data/raw")
def test_real_olist_files_load_completely(con):
    counts = load_all(con, RAW_DIR)
    assert find_count_mismatches(counts, RAW_DIR) == {}
    assert counts["sellers"] > 3000
