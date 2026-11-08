"""Tests for chronological splits, preprocessing, metrics and importance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.features import tradable_features
from src.models.baseline import clip_target, regression_metrics, risk_deciles, risk_target
from src.models.classifier import classification_metrics, coverage_table
from src.models.common import Winsorizer, feature_spec, grouped_permutation_importance
from src.models.split import usable, walk_forward_folds


def frame(start="2023-01-01", end="2025-06-30"):
    days = pd.date_range(start, end, freq="D")
    df = pd.DataFrame({"market_date": np.repeat(days, 2)})
    df["period"] = np.where(df["market_date"] <= "2024-12-31", "train", "validation")
    return df


def test_walk_forward_folds_are_chronological_and_train_only():
    df = frame()
    folds = walk_forward_folds(df)
    assert [name for *_, name in folds] == ["2024-01", "2024-04", "2024-07", "2024-10"]
    for fit_idx, eval_idx, _ in folds:
        assert df.loc[fit_idx, "market_date"].max() < df.loc[eval_idx, "market_date"].min()
        assert (df.loc[fit_idx.union(eval_idx), "period"] == "train").all()   # validation never used
    # expanding window: each fold trains on more data than the last
    sizes = [len(f) for f, _, _ in folds]
    assert sizes == sorted(sizes)


def test_usable_requires_all_features_and_target():
    df = pd.DataFrame({"a": [1, np.nan, 3], "b": [1, 2, 3], "da_rt_spread": [1, 1, np.nan]})
    assert usable(df, ["a", "b"]).tolist() == [True, False, False]


def test_feature_spec_uses_only_tradable_features():
    spec = feature_spec()
    assert set(spec.columns) <= set(tradable_features())
    for leak in ("da_lmp", "rt_lmp", "actual_load", "load_forecast_error", "da_rt_spread", "target_rt_above_da"):
        assert leak not in spec.columns
    assert "calendar" not in feature_spec(exclude_groups=("calendar",)).groups


def test_winsorizer_uses_training_quantiles():
    w = Winsorizer(0.1, 0.9).fit(np.arange(101, dtype=float).reshape(-1, 1))
    out = w.transform(np.array([[-1000.0], [50.0], [1e6]]))
    assert out.ravel().tolist() == [10.0, 50.0, 90.0]


def test_coverage_table_positions_and_capture():
    p = np.array([0.9, 0.1, 0.5, 0.65])
    s = np.array([10.0, -20.0, 5.0, -4.0])
    t = coverage_table(p, s, thresholds=(0.6,)).iloc[0]
    # long hour 0 (+10), short hour 1 (+20), flat hour 2, long hour 3 (-4)
    assert t["hours"] == 3
    assert t["mean_capture_per_traded_hour"] == pytest.approx((10 + 20 - 4) / 3)
    assert t["hit_rate"] == pytest.approx(2 / 3)


def test_classification_metrics_baselines():
    y = np.array([0, 0, 0, 1])
    m = classification_metrics(y, np.full(4, 0.25), spread=np.array([-1, -1, -1, 9.0]), base_rate=0.25)
    assert m["brier_skill_score"] == pytest.approx(0.0)
    assert m["majority_class_accuracy"] == 0.75
    assert m["dollar_weighted_hit_rate"] == pytest.approx(3 / 12)   # right on the three $1 hours only


def test_regression_metrics_and_targets():
    y = np.array([-2.0, 0.0, 2.0, 200.0])
    m = regression_metrics(y, np.zeros(4), train_y=np.zeros(10))
    assert m["mae"] == pytest.approx(51.0) and m["mae_skill_vs_median"] == pytest.approx(0.0)
    assert clip_target(y).max() == 100.0
    assert risk_target(np.array([-np.e + 1]))[0] == pytest.approx(1.0)


def test_risk_deciles_rank_ordering():
    rng = np.random.default_rng(0)
    pred = rng.uniform(0, 1, 1000)
    spread = pred * 100 * rng.choice([-1, 1], 1000)
    t = risk_deciles(pred, spread)
    assert t["mean_abs_spread"].is_monotonic_increasing


def test_grouped_permutation_importance_finds_the_informative_group():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"signal": rng.normal(size=500), "noise": rng.normal(size=500)})
    y = X["signal"].to_numpy() * 3
    corr = lambda a, b: np.corrcoef(a, b)[0, 1]
    imp = grouped_permutation_importance(lambda d: d["signal"].to_numpy() * 3, X, y, corr,
                                         {"signal": ["signal"], "noise": ["noise"]}, n_repeats=3)
    assert imp.iloc[0]["group"] == "signal" and imp.iloc[0]["importance"] > 0.5
    assert imp.set_index("group").loc["noise", "importance"] == pytest.approx(0.0)


def test_day_block_permutation_keeps_rows_within_days():
    days = pd.Series(np.repeat(pd.date_range("2024-01-01", periods=50), 24))
    X = pd.DataFrame({"hour": np.tile(np.arange(24), 50).astype(float)})
    # shuffling whole days keeps each row's within-day position -> 'hour' is unchanged -> zero importance
    imp = grouped_permutation_importance(lambda d: d["hour"].to_numpy(), X, X["hour"].to_numpy(),
                                         lambda a, b: -np.abs(a - b).sum(), {"hour": ["hour"]}, block=days)
    assert imp["importance"].iloc[0] == pytest.approx(0.0)
