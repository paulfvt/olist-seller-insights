"""Entraînement et évaluation du modèle de risque de baisse de satisfaction.

Déroulé
-------
1. Jeu de données par instantanés (features.py), découpé dans le TEMPS :
   train -> validation -> test. Entre deux ensembles, des mois sont sacrifiés (purge) :
   la cible d'un instantané couvre 90 jours, elle ne doit pas chevaucher les dates de
   l'ensemble suivant, sinon le modèle apprendrait sur la période qu'il est censé prédire.
2. Références (baselines) : une prédiction constante, puis une règle métier
   (« un vendeur déjà mal noté le restera »). Un modèle qui ne les bat pas ne sert à rien.
3. Deux modèles : régression logistique (lisible) et gradient boosting. Les réglages sont
   choisis sur la VALIDATION ; le meilleur est réentraîné sur train + validation, puis
   évalué UNE fois sur le test.
4. Métriques adaptées au déséquilibre (16 % de positifs) : PR-AUC (comparée à la part de
   positifs, qui est la PR-AUC d'un tirage au hasard), précision et rappel sur les 10 %
   de vendeurs jugés les plus à risque chaque mois, Brier score pour les probabilités.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from olist_insights.config import PROJECT_ROOT, load_settings, set_seed
from olist_insights.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TargetConfig,
    build_dataset,
    build_features,
    month_starts,
)
from olist_insights.logging_utils import setup_logging

logger = logging.getLogger(__name__)

# Découpage temporel. Chaque fenêtre de cible (90 jours) se termine avant le premier
# mois de l'ensemble suivant : 2017-09 + 90 j < 2017-12 ; 2018-01 + 90 j < 2018-04.
SPLITS: dict[str, tuple[str, str]] = {
    "train": ("2017-03", "2017-09"),
    "valid": ("2017-12", "2018-01"),
    "test": ("2018-04", "2018-06"),
}
TOP_SHARE = 0.10           # part des vendeurs signalés chaque mois (capacité d'action supposée)
SCORING_CUTOFF = "2018-09"  # instantané « actuel » pour l'application (pas de cible connue)
REPORT_PATH = PROJECT_ROOT / "reports" / "model_metrics.json"


# ---------------------------------------------------------------------------
# Modèles
# ---------------------------------------------------------------------------


def _preprocessor(scale: bool) -> ColumnTransformer:
    """Valeurs manquantes -> médiane + indicateur « était manquant » ; catégories -> one-hot.

    min_frequency regroupe les catégories rares en « infrequent » : sinon une catégorie
    vue 3 fois donne une colonne apprise sur du bruit. handle_unknown gère une catégorie
    jamais vue à l'entraînement.
    """
    numeric = [SimpleImputer(strategy="median", add_indicator=True)]
    if scale:  # la régression logistique est sensible à l'échelle, pas les arbres
        numeric.append(StandardScaler())
    return ColumnTransformer([
        ("num", make_pipeline(*numeric), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30,
                              sparse_output=False), CATEGORICAL_FEATURES),
    ])


def candidate_models(seed: int) -> dict[str, Pipeline]:
    """Petite grille, volontairement : avec ~1 600 lignes d'entraînement, une grande
    recherche d'hyperparamètres sur-apprendrait la validation elle-même."""
    models = {}
    for c in (0.05, 0.3, 1.0):
        # class_weight="balanced" : les erreurs sur la classe rare comptent plus
        models[f"logreg_C{c}"] = Pipeline([
            ("prep", _preprocessor(scale=True)),
            ("clf", LogisticRegression(C=c, class_weight="balanced", max_iter=2000)),
        ])
    for depth, lr in ((3, 0.05), (5, 0.05), (3, 0.1)):
        models[f"hgb_d{depth}_lr{lr}"] = Pipeline([
            ("prep", _preprocessor(scale=False)),
            ("clf", HistGradientBoostingClassifier(
                max_depth=depth, learning_rate=lr, max_iter=300, l2_regularization=1.0,
                early_stopping=True, validation_fraction=0.15, class_weight="balanced",
                random_state=seed)),
        ])
    return models


def persistence_score(df: pd.DataFrame) -> np.ndarray:
    """Règle métier de référence : le risque futur = la part de notes 1-2 des 90 derniers jours."""
    return df["low_rate_90d"].to_numpy()


# ---------------------------------------------------------------------------
# Métriques
# ---------------------------------------------------------------------------


def precision_recall_at_top(df: pd.DataFrame, score: np.ndarray, share: float = TOP_SHARE) -> tuple[float, float]:
    """Chaque mois, on signale les `share` vendeurs au score le plus haut.

    Précision = part des signalés qui ont vraiment un mauvais épisode ;
    rappel = part des mauvais épisodes qui ont été signalés. Moyenne sur les mois.
    Par mois, car en pratique on agit sur une liste mensuelle, pas sur tout l'historique.
    """
    tmp = df[["cutoff", "label"]].assign(score=score)
    precisions, recalls = [], []
    for _, g in tmp.groupby("cutoff"):
        k = max(1, int(round(share * len(g))))
        top = g.nlargest(k, "score")
        precisions.append(top["label"].mean())
        if g["label"].sum() > 0:
            recalls.append(top["label"].sum() / g["label"].sum())
    return float(np.mean(precisions)), float(np.mean(recalls))


def evaluate(df: pd.DataFrame, score: np.ndarray, is_proba: bool = True) -> dict[str, float]:
    y = df["label"].to_numpy()
    prevalence = float(y.mean())
    pr_auc = float(average_precision_score(y, score))
    p_top, r_top = precision_recall_at_top(df, score)
    out = {
        "n": int(len(y)),
        "prevalence": round(prevalence, 4),
        "pr_auc": round(pr_auc, 4),
        "pr_auc_lift": round(pr_auc / prevalence, 2),  # 1.0 = pas mieux que le hasard
        "roc_auc": round(float(roc_auc_score(y, score)), 4),
        "precision_top10": round(p_top, 4),
        "recall_top10": round(r_top, 4),
    }
    if is_proba:
        out["brier"] = round(float(brier_score_loss(y, score)), 4)
    return out


def evaluate_with_subgroup(df: pd.DataFrame, score: np.ndarray, cfg: TargetConfig, is_proba: bool = True) -> dict:
    """Métriques globales + sur les vendeurs « jusque-là corrects » (problème émergent)."""
    ok = (df["low_rate_90d"] < cfg.bad_threshold).to_numpy()
    return {
        "all": evaluate(df, score, is_proba),
        "previously_ok": evaluate(df[ok], score[ok], is_proba),
    }


# ---------------------------------------------------------------------------
# Pipeline complet
# ---------------------------------------------------------------------------


def split_dataset(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    parts = {}
    for name, (start, end) in SPLITS.items():
        months = set(month_starts(start, end))
        parts[name] = df[df["cutoff"].isin(months)].reset_index(drop=True)
    return parts


def _xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    return df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], df["label"].to_numpy()


def train_and_evaluate(df: pd.DataFrame, cfg: TargetConfig, seed: int) -> tuple[Pipeline, dict]:
    parts = split_dataset(df)
    for name, part in parts.items():
        logger.info("%-5s : %5d instantanés, %4.1f %% de positifs", name, len(part), 100 * part["label"].mean())

    # 1. Sélection sur la validation
    x_tr, y_tr = _xy(parts["train"])
    x_va, _ = _xy(parts["valid"])
    selection = {}
    for name, model in candidate_models(seed).items():
        model.fit(x_tr, y_tr)
        selection[name] = evaluate(parts["valid"], model.predict_proba(x_va)[:, 1])
        logger.info("valid | %-18s PR-AUC %.3f | précision top10 %.3f",
                    name, selection[name]["pr_auc"], selection[name]["precision_top10"])
    best_name = max(selection, key=lambda n: selection[n]["pr_auc"])
    best_logreg = max((n for n in selection if n.startswith("logreg")), key=lambda n: selection[n]["pr_auc"])
    logger.info("Retenu sur la validation : %s", best_name)
    # La règle de référence est aussi mesurée sur la validation : si elle fait aussi bien
    # que le modèle, il faut le savoir AVANT de regarder le test.
    rule_valid = evaluate(parts["valid"], persistence_score(parts["valid"]), is_proba=False)
    logger.info("valid | %-18s PR-AUC %.3f | précision top10 %.3f",
                "rule_persistence", rule_valid["pr_auc"], rule_valid["precision_top10"])
    if rule_valid["pr_auc"] >= selection[best_name]["pr_auc"]:
        logger.warning("Le modèle retenu ne bat pas la règle de référence sur la validation")

    # 2. Réentraînement sur train + validation, évaluation unique sur le test
    train_full = pd.concat([parts["train"], parts["valid"]], ignore_index=True)
    x_full, y_full = _xy(train_full)
    x_te, _ = _xy(parts["test"])
    test = parts["test"]
    results = {
        "constant_prior": evaluate_with_subgroup(test, np.full(len(test), y_full.mean()), cfg),
        "rule_persistence": evaluate_with_subgroup(test, persistence_score(test), cfg, is_proba=False),
    }
    fitted = {}
    for name in dict.fromkeys([best_logreg, best_name]):  # le modèle retenu + la meilleure logreg, pour comparer
        model = candidate_models(seed)[name].fit(x_full, y_full)
        fitted[name] = model
        results[name] = evaluate_with_subgroup(test, model.predict_proba(x_te)[:, 1], cfg)

    # 3. Ce qui compte pour le modèle : importance par permutation sur la validation
    #    (on mélange une colonne et on mesure la perte de PR-AUC). Calculée sur le modèle
    #    entraîné sur le train seul, pour ne pas toucher au test.
    probe = candidate_models(seed)[best_name].fit(x_tr, y_tr)
    imp = permutation_importance(probe, x_va, parts["valid"]["label"], scoring="average_precision",
                                 n_repeats=10, random_state=seed)
    importance = (pd.Series(imp.importances_mean, index=x_va.columns)
                  .sort_values(ascending=False).round(4).head(10).to_dict())

    report = {
        "target": asdict(cfg),
        "splits": SPLITS,
        "selection_on_valid": {**selection, "rule_persistence": rule_valid},
        "selected_model": best_name,
        "backtest_before_test": rolling_backtest(df, best_name, seed),
        "test": results,
        "permutation_importance_valid": importance,
    }
    return fitted[best_name], report


def rolling_backtest(df: pd.DataFrame, model_name: str, seed: int,
                     months: tuple[str, str] = ("2017-08", "2018-01")) -> dict[str, float]:
    """Évaluation glissante (rolling origin) sur les mois AVANT le test.

    Pour chaque mois m : entraînement sur les instantanés d'au moins 3 mois avant m
    (leur cible de 90 jours est terminée à la date m), évaluation sur m. Un seul jeu de
    validation tombe sur une période particulière (ici la crise logistique de début 2018) ;
    plusieurs mois donnent une comparaison plus stable. PR-AUC moyenne sur les mois.
    """
    scores = {"rule_persistence": [], model_name: []}
    for m in month_starts(*months):
        train = df[df["cutoff"] <= (pd.Timestamp(m) - pd.DateOffset(months=3)).date()]
        month = df[df["cutoff"] == m]
        if month.empty or month["label"].nunique() < 2:
            continue
        scores["rule_persistence"].append(average_precision_score(month["label"], persistence_score(month)))
        model = candidate_models(seed)[model_name].fit(*_xy(train))
        scores[model_name].append(average_precision_score(month["label"], model.predict_proba(_xy(month)[0])[:, 1]))
    return {name: round(float(np.mean(v)), 4) for name, v in scores.items()}


def score_current_sellers(con: duckdb.DuckDBPyConnection, model: Pipeline, cfg: TargetConfig) -> int:
    """Score de risque à la dernière date disponible -> gold.seller_risk (pour l'application)."""
    (cutoff,) = month_starts(SCORING_CUTOFF, SCORING_CUTOFF)
    feats = build_features(con, [cutoff], cfg)
    feats["risk_score"] = model.predict_proba(feats[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[:, 1]
    feats["risk_rank"] = feats["risk_score"].rank(ascending=False, method="first").astype(int)
    con.register("risk_df", feats)
    con.execute("CREATE SCHEMA IF NOT EXISTS gold")
    con.execute("CREATE OR REPLACE TABLE gold.seller_risk AS SELECT * FROM risk_df ORDER BY risk_rank")
    con.unregister("risk_df")
    return len(feats)


def main() -> int:
    settings = load_settings()
    setup_logging(settings.log_level)
    set_seed(settings.random_seed)
    cfg = TargetConfig()

    with duckdb.connect(str(settings.duckdb_path)) as con:
        # tous les mois du train au test, y compris les mois de purge (utiles au backtest ;
        # split_dataset ne garde que les mois de chaque ensemble)
        all_months = month_starts(SPLITS["train"][0], SPLITS["test"][1])
        df = build_dataset(con, all_months, cfg)
        model, report = train_and_evaluate(df, cfg, settings.random_seed)
        n_scored = score_current_sellers(con, model, cfg)

    model_path = settings.data_dir / "models" / "risk_model.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))

    logger.info("Test (évalué une seule fois) :")
    for name, res in report["test"].items():
        a, o = res["all"], res["previously_ok"]
        logger.info("  %-18s PR-AUC %.3f (x%.1f hasard) | précision top10 %.3f | rappel top10 %.3f "
                    "| jusque-là OK : PR-AUC %.3f (x%.1f)",
                    name, a["pr_auc"], a["pr_auc_lift"], a["precision_top10"], a["recall_top10"],
                    o["pr_auc"], o["pr_auc_lift"])
    logger.info("Backtest avant test (PR-AUC moyenne) : %s", report["backtest_before_test"])
    logger.info("Top features : %s", ", ".join(list(report["permutation_importance_valid"])[:5]))
    logger.info("gold.seller_risk : %d vendeurs scorés au %s", n_scored, SCORING_CUTOFF)
    logger.info("Rapport : %s | modèle : %s", REPORT_PATH, model_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
