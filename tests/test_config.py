import random
from pathlib import Path

import numpy as np
import pytest

from olist_insights.config import PROJECT_ROOT, load_settings, set_seed

ENV_VARS = [
    "OLIST_DATA_DIR",
    "OLIST_RAW_DIR",
    "OLIST_CACHE_DIR",
    "OLIST_DUCKDB_PATH",
    "RANDOM_SEED",
    "LOG_LEVEL",
]


@pytest.fixture
def clean_env(monkeypatch, tmp_path) -> Path:
    """Isole le test du shell et du vrai .env : on pointe vers un .env inexistant."""
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    return tmp_path / "missing.env"


def test_defaults_point_inside_project(clean_env):
    s = load_settings(clean_env)
    assert s.data_dir == PROJECT_ROOT / "data"
    assert s.raw_dir == PROJECT_ROOT / "data" / "raw"
    assert s.duckdb_path == PROJECT_ROOT / "data" / "olist.duckdb"
    assert s.random_seed == 42
    assert s.log_level == "INFO"


def test_relative_env_path_is_resolved_from_root(clean_env, monkeypatch):
    monkeypatch.setenv("OLIST_DATA_DIR", "other_data")
    s = load_settings(clean_env)
    assert s.data_dir == PROJECT_ROOT / "other_data"
    assert s.raw_dir == PROJECT_ROOT / "other_data" / "raw"


def test_absolute_env_path_is_kept(clean_env, monkeypatch, tmp_path):
    monkeypatch.setenv("OLIST_DUCKDB_PATH", str(tmp_path / "x.duckdb"))
    monkeypatch.setenv("RANDOM_SEED", "7")
    s = load_settings(clean_env)
    assert s.duckdb_path == tmp_path / "x.duckdb"
    assert s.random_seed == 7


def test_ensure_dirs_creates_folders(clean_env, monkeypatch, tmp_path):
    monkeypatch.setenv("OLIST_DATA_DIR", str(tmp_path / "data"))
    s = load_settings(clean_env)
    s.ensure_dirs()
    assert s.raw_dir.is_dir() and s.cache_dir.is_dir()


def test_set_seed_is_reproducible():
    set_seed(123)
    first = (random.random(), np.random.rand())
    set_seed(123)
    assert (random.random(), np.random.rand()) == first
