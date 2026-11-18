"""Configuration centrale : chemins, graine aléatoire, niveau de log.

Chaque valeur peut être surchargée par une variable d'environnement ou par le
fichier .env à la racine. Les chemins relatifs sont résolus depuis la racine du
projet, pour que les scripts marchent quel que soit le dossier d'où on les lance.
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from olist_insights.logging_utils import setup_logging

# src/olist_insights/config.py -> parents[2] = racine du projet
PROJECT_ROOT = Path(__file__).resolve().parents[2]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_dir: Path
    raw_dir: Path
    cache_dir: Path
    duckdb_path: Path
    random_seed: int
    log_level: str

    def ensure_dirs(self) -> None:
        """Crée les dossiers de données s'ils n'existent pas."""
        for directory in (self.data_dir, self.raw_dir, self.cache_dir):
            directory.mkdir(parents=True, exist_ok=True)


def _path_from_env(name: str, default: Path) -> Path:
    value = os.getenv(name)
    if not value:
        return default
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_settings(env_file: Path | None = None) -> Settings:
    """Lit la configuration (variables d'environnement, puis .env, puis défauts)."""
    # override=False : une variable déjà définie dans le shell garde la priorité sur .env
    load_dotenv(env_file or PROJECT_ROOT / ".env", override=False)
    data_dir = _path_from_env("OLIST_DATA_DIR", PROJECT_ROOT / "data")
    return Settings(
        project_root=PROJECT_ROOT,
        data_dir=data_dir,
        raw_dir=_path_from_env("OLIST_RAW_DIR", data_dir / "raw"),
        cache_dir=_path_from_env("OLIST_CACHE_DIR", data_dir / "cache"),
        duckdb_path=_path_from_env("OLIST_DUCKDB_PATH", data_dir / "olist.duckdb"),
        random_seed=int(os.getenv("RANDOM_SEED", "42")),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
    )


def set_seed(seed: int) -> None:
    """Fixe les graines aléatoires (Python, NumPy) pour des résultats reproductibles."""
    random.seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    # Contrôle rapide : python -m olist_insights.config
    settings = load_settings()
    setup_logging(settings.log_level)
    settings.ensure_dirs()
    for field, value in vars(settings).items():
        logger.info("%-13s = %s", field, value)
