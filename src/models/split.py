"""Chronological data splits. Never random: every evaluation row is later than every training row."""

from __future__ import annotations

import pandas as pd

PERIOD_ORDER = ["lookback", "train", "validation", "test", "holdout"]


def period_frame(df: pd.DataFrame, *periods: str) -> pd.DataFrame:
    """Rows belonging to the given periods (labels assigned in cleaning from config dates)."""
    unknown = set(periods) - set(PERIOD_ORDER)
    if unknown:
        raise ValueError(f"unknown periods {unknown}")
    return df.loc[df["period"].isin(periods)]


def usable(df: pd.DataFrame, features: list[str], target: str = "da_rt_spread") -> pd.Series:
    """Rows with a target and every feature present.

    Rows failing this (mainly hours where PJM published no D-1 load forecast) get no
    prediction; in the backtest that means no position.
    """
    return df[features + [target]].notna().all(axis=1)


def walk_forward_folds(
    df: pd.DataFrame, start: str = "2024-01-01", end: str = "2024-12-31", freq: str = "QS",
) -> list[tuple[pd.Index, pd.Index, str]]:
    """Expanding-window folds: train on everything before each block, evaluate on the block.

    Default: four quarterly folds in 2024, each trained on all earlier training data.
    Only rows of the ``train`` period are used, so validation/test are never touched.
    """
    tr = df.loc[df["period"] == "train"]
    edges = list(pd.date_range(start, end, freq=freq)) + [pd.Timestamp(end) + pd.Timedelta(days=1)]
    folds = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        fit_idx = tr.index[tr["market_date"] < lo]
        eval_idx = tr.index[(tr["market_date"] >= lo) & (tr["market_date"] < hi)]
        if len(fit_idx) and len(eval_idx):
            folds.append((fit_idx, eval_idx, f"{lo:%Y-%m}"))
    return folds
