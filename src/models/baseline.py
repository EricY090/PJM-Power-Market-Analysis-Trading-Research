"""Magnitude models.

1. Spread regression (plan item 4.3): OLS and ridge on the spread. The spread is extremely
   heavy-tailed (skew ~18), so models are *fit* on the spread clipped to ±$100 (spike hours
   would otherwise dominate the squared loss) but always *evaluated* on the raw spread.
2. Risk model (extension motivated by Phases 2–3): ridge on log(1 + |spread|), i.e. how big
   the spread is likely to be, regardless of sign. Useful for sizing and risk filters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src.models.common import FeatureSpec, preprocessor

TARGET_CLIP = 100.0


def make_ols(spec: FeatureSpec) -> Pipeline:
    return Pipeline([("prep", preprocessor(spec)), ("model", LinearRegression())])


def make_ridge(spec: FeatureSpec, alpha: float = 10.0) -> Pipeline:
    return Pipeline([("prep", preprocessor(spec)), ("model", Ridge(alpha=alpha))])


def make_gbm_regressor(spec: FeatureSpec, loss: str = "absolute_error", learning_rate: float = 0.05,
                       max_iter: int = 300, max_depth: int = 3, min_samples_leaf: int = 200) -> Pipeline:
    return Pipeline([("prep", preprocessor(spec, scale=False)),
                     ("model", HistGradientBoostingRegressor(
                         loss=loss, learning_rate=learning_rate, max_iter=max_iter, max_depth=max_depth,
                         min_samples_leaf=min_samples_leaf, early_stopping=False, random_state=0))])


def clip_target(y: pd.Series | np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(y, dtype=float), -TARGET_CLIP, TARGET_CLIP)


def regression_metrics(y: np.ndarray, pred: np.ndarray, train_y: np.ndarray | None = None) -> pd.Series:
    """Error metrics on the raw spread, with constant baselines from the training period."""
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    out = {
        "n": len(y),
        "mae": mean_absolute_error(y, pred),
        "rmse": np.sqrt(mean_squared_error(y, pred)),
        "median_ae": np.median(np.abs(y - pred)),
        "r2": r2_score(y, pred),
        "spearman": pd.Series(pred).corr(pd.Series(y), method="spearman"),
        "sign_accuracy": np.mean(np.sign(pred) == np.sign(y)),
    }
    if train_y is not None:
        for name, c in (("zero", 0.0), ("train_median", np.median(train_y)), ("train_mean", np.mean(train_y))):
            out[f"mae_baseline_{name}"] = mean_absolute_error(y, np.full_like(y, c))
        out["mae_skill_vs_median"] = 1 - out["mae"] / out["mae_baseline_train_median"]
    return pd.Series(out)


def risk_target(spread: pd.Series | np.ndarray) -> np.ndarray:
    return np.log1p(np.abs(np.asarray(spread, dtype=float)))


def risk_deciles(pred_risk: np.ndarray, spread: np.ndarray, q: int = 10) -> pd.DataFrame:
    """Realized |spread| by decile of predicted risk: does the model rank risky hours correctly?"""
    d = pd.DataFrame({"pred": np.asarray(pred_risk, dtype=float), "abs_spread": np.abs(np.asarray(spread, dtype=float))})
    d["decile"] = pd.qcut(d["pred"].rank(method="first"), q=q, labels=range(1, q + 1))
    return d.groupby("decile", observed=True).agg(
        hours=("abs_spread", "size"), mean_abs_spread=("abs_spread", "mean"),
        median_abs_spread=("abs_spread", "median"), p95_abs_spread=("abs_spread", lambda s: s.quantile(0.95)),
        share_above_50=("abs_spread", lambda s: (s > 50).mean())).reset_index()
