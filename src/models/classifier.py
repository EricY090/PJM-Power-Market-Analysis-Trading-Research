"""Direction model: P(RT > DA) for each hour.

* Logistic regression (the plan's baseline) on the preprocessed tradable features.
* Gradient boosting (optional comparison, item 4.4).
* Metrics against honest baselines: the training base rate as a constant probability, and
  "always predict the majority class" (RT < DA).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, brier_score_loss, f1_score, log_loss, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.models.common import FeatureSpec, preprocessor


def make_logit(spec: FeatureSpec, C: float = 0.1) -> Pipeline:
    return Pipeline([("prep", preprocessor(spec)),
                     ("model", LogisticRegression(C=C, max_iter=2000))])


def make_gbm(spec: FeatureSpec, learning_rate: float = 0.05, max_iter: int = 300, max_depth: int = 3,
             min_samples_leaf: int = 200, l2_regularization: float = 1.0) -> Pipeline:
    return Pipeline([("prep", preprocessor(spec, scale=False)),
                     ("model", HistGradientBoostingClassifier(
                         learning_rate=learning_rate, max_iter=max_iter, max_depth=max_depth,
                         min_samples_leaf=min_samples_leaf, l2_regularization=l2_regularization,
                         early_stopping=False, random_state=0))])


def classification_metrics(y: np.ndarray, p: np.ndarray, spread: np.ndarray | None = None,
                           base_rate: float | None = None) -> pd.Series:
    """Discrimination, calibration and directional metrics for predicted P(RT > DA)."""
    y, p = np.asarray(y).astype(int), np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    pred = (p > 0.5).astype(int)
    out = {
        "n": len(y),
        "base_rate_rt_above_da": y.mean(),
        "roc_auc": roc_auc_score(y, p),
        "log_loss": log_loss(y, p),
        "brier": brier_score_loss(y, p),
        "accuracy": accuracy_score(y, pred),
        "precision_up": precision_score(y, pred, zero_division=0),
        "recall_up": recall_score(y, pred, zero_division=0),
        "f1_up": f1_score(y, pred, zero_division=0),
        "share_predicted_up": pred.mean(),
        "majority_class_accuracy": max(y.mean(), 1 - y.mean()),
    }
    if base_rate is not None:      # constant-probability baseline from the training period
        q = np.full_like(p, base_rate)
        out["baseline_log_loss"] = log_loss(y, q, labels=[0, 1])
        out["baseline_brier"] = brier_score_loss(y, q)
        out["brier_skill_score"] = 1 - out["brier"] / out["baseline_brier"]
    if spread is not None:         # dollar-weighted hit rate: are we right on the hours that matter?
        s = np.asarray(spread, dtype=float)
        correct = (pred == (s > 0).astype(int))
        out["dollar_weighted_hit_rate"] = np.sum(np.abs(s) * correct) / np.sum(np.abs(s))
    return pd.Series(out)


def reliability_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Calibration: mean predicted vs observed frequency within predicted-probability deciles."""
    d = pd.DataFrame({"y": np.asarray(y).astype(int), "p": np.asarray(p, dtype=float)})
    d["bin"] = pd.qcut(d["p"], q=bins, duplicates="drop")
    return d.groupby("bin", observed=True).agg(n=("y", "size"), mean_predicted=("p", "mean"),
                                               observed_rate=("y", "mean")).reset_index()


def coverage_table(p: np.ndarray, spread: np.ndarray, thresholds=(0.5, 0.55, 0.6, 0.65, 0.7)) -> pd.DataFrame:
    """Trade only when confident: long spread if p > t, short if p < 1 - t.

    Reports how often we'd trade (coverage), the hit rate on those hours, and the average
    spread captured per traded hour (position x spread, before any costs).
    """
    p, s = np.asarray(p, dtype=float), np.asarray(spread, dtype=float)
    rows = []
    for t in thresholds:
        pos = np.where(p > t, 1, np.where(p < 1 - t, -1, 0))
        traded = pos != 0
        pnl = pos * s
        rows.append({
            "threshold": t, "coverage": traded.mean(), "hours": int(traded.sum()),
            "share_long": (pos == 1).sum() / max(1, traded.sum()),
            "hit_rate": (pnl[traded] > 0).mean() if traded.any() else np.nan,
            "mean_capture_per_traded_hour": pnl[traded].mean() if traded.any() else np.nan,
            "median_capture": np.median(pnl[traded]) if traded.any() else np.nan,
        })
    return pd.DataFrame(rows)


def logit_coefficients(pipe: Pipeline) -> pd.DataFrame:
    """Coefficients on standardized / one-hot inputs (log-odds per 1 std or per category)."""
    names = pipe.named_steps["prep"].get_feature_names_out()
    coef = pipe.named_steps["model"].coef_.ravel()
    return (pd.DataFrame({"feature": names, "coef": coef})
            .assign(abs_coef=lambda d: d["coef"].abs())
            .sort_values("abs_coef", ascending=False).reset_index(drop=True))
