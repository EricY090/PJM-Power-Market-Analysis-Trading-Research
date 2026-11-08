"""Shared model plumbing: feature specification, preprocessing, grouped permutation importance."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.features import FEATURES, tradable_features

CATEGORICAL = ["hour_ending", "day_of_week", "month", "fc_load_regime"]
BINARY = ["is_holiday", "is_weekend"]
DROPPED = ["season"]          # fully determined by month


@dataclass
class FeatureSpec:
    numeric: list[str]
    categorical: list[str]
    binary: list[str]
    groups: dict[str, list[str]] = field(default_factory=dict)

    @property
    def columns(self) -> list[str]:
        return self.numeric + self.categorical + self.binary


def feature_spec(exclude_groups: tuple[str, ...] = ()) -> FeatureSpec:
    """Model inputs: all tradable features, optionally without whole feature groups (ablation)."""
    cols = [c for c in tradable_features() if c not in DROPPED and FEATURES[c].group not in exclude_groups]
    cat = [c for c in CATEGORICAL if c in cols]
    binary = [c for c in BINARY if c in cols]
    num = [c for c in cols if c not in cat and c not in binary]
    groups: dict[str, list[str]] = {}
    for c in cols:
        groups.setdefault(FEATURES[c].group, []).append(c)
    return FeatureSpec(num, cat, binary, groups)


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each column to quantiles learned on the training data (default 1st / 99th)."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower, self.upper = lower, upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanquantile(X, self.lower, axis=0)
        self.hi_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def preprocessor(spec: FeatureSpec, scale: bool = True) -> ColumnTransformer:
    """Numeric: winsorize (train 1/99 pct) -> median impute -> standardize. Categorical: one-hot."""
    num_steps = [("winsorize", Winsorizer()), ("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), spec.numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), spec.categorical),
        ("bin", "passthrough", spec.binary),
    ], verbose_feature_names_out=False)


def to_model_frame(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Model input frame with consistent dtypes (categoricals as strings, booleans as 0/1)."""
    X = df[spec.columns].copy()
    for c in spec.categorical:
        X[c] = X[c].astype(str)
    for c in spec.binary:
        X[c] = X[c].astype(int)
    return X


def grouped_permutation_importance(
    predict: Callable[[pd.DataFrame], np.ndarray], X: pd.DataFrame, y: np.ndarray,
    score: Callable[[np.ndarray, np.ndarray], float], groups: dict[str, list[str]],
    n_repeats: int = 5, seed: int = 0, block: pd.Series | None = None,
) -> pd.DataFrame:
    """Drop in ``score`` when all columns of a group are shuffled together.

    If ``block`` (e.g. market_date) is given, whole days are shuffled as units, so the
    within-day structure of the shuffled features is preserved.
    """
    rng = np.random.default_rng(seed)
    base = score(y, predict(X))
    n = len(X)
    if block is not None:
        # integer row positions of each day, in time order
        day_rows = [np.flatnonzero(block.to_numpy() == d) for d in pd.unique(block.to_numpy())]

    def permutation() -> np.ndarray:
        if block is None:
            return rng.permutation(n)
        perm = np.empty(n, dtype=int)
        order = rng.permutation(len(day_rows))
        for src_day, dst_day in zip(day_rows, (day_rows[k] for k in order)):
            # row j of a day takes row j of the randomly assigned day (clamped for 23/25-hour days)
            perm[src_day] = dst_day[np.minimum(np.arange(len(src_day)), len(dst_day) - 1)]
        return perm

    rows = []
    for name, cols in groups.items():
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[cols] = X[cols].to_numpy()[permutation()]
            drops.append(base - score(y, predict(Xp)))
        rows.append({"group": name, "n_features": len(cols), "importance": np.mean(drops), "std": np.std(drops)})
    return pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)
