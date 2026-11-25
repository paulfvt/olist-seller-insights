import datetime as dt

import numpy as np
import pandas as pd
import pytest

from olist_insights.features import TargetConfig, month_starts
from olist_insights.model import SPLITS, evaluate, precision_recall_at_top


def test_splits_do_not_overlap_in_time():
    """La cible d'un instantané couvre `horizon_days` : elle doit se terminer avant le
    premier instantané de l'ensemble suivant (sinon fuite entre train et test)."""
    horizon = dt.timedelta(days=TargetConfig().horizon_days)
    names = list(SPLITS)
    for current, following in zip(names, names[1:]):
        last = month_starts(*SPLITS[current])[-1]
        first_next = month_starts(*SPLITS[following])[0]
        assert last + horizon <= first_next, f"{current} chevauche {following}"


def _df(labels, cutoffs):
    return pd.DataFrame({"label": labels, "cutoff": cutoffs})


def test_precision_recall_at_top_is_computed_per_month():
    # 2 mois de 10 vendeurs ; on signale le top 10 % = 1 vendeur par mois
    df = _df([1, 1] + [0] * 8 + [1] + [0] * 9, ["m1"] * 10 + ["m2"] * 10)
    score = np.array([0.9, 0.1] + [0.0] * 8 + [0.0] + [0.5] * 9)
    precision, recall = precision_recall_at_top(df, score, share=0.10)
    assert precision == pytest.approx((1 + 0) / 2)   # m1 : bon vendeur signalé ; m2 : raté
    assert recall == pytest.approx((1 / 2 + 0) / 2)  # m1 : 1 positif sur 2 ; m2 : 0 sur 1


def test_evaluate_perfect_and_random_scores():
    labels = [1, 0, 0, 0] * 25
    df = _df(labels, ["m1"] * 100)
    perfect = evaluate(df, np.array(labels, dtype=float))
    assert perfect["pr_auc"] == 1.0 and perfect["pr_auc_lift"] == pytest.approx(4.0)  # 1 / 25 %
    constant = evaluate(df, np.full(100, 0.25))
    assert constant["pr_auc"] == pytest.approx(0.25)  # score constant = hasard = prévalence
