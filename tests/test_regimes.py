"""Tests for regime labels: leak-free volatility window, train-only thresholds, seasons."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.diagnostics import composition_lift, extreme_flags
from src.analysis.regimes import add_regimes, load_regime, season, trailing_rt_volatility
from src.analysis.spreads import distribution_summary


def daily_frame(days: int, rt_by_day: list[float] | None = None, hours: int = 24) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=days, freq="D")
    rows = []
    for k, d in enumerate(dates):
        for h in range(hours):
            rt = (rt_by_day[k] if rt_by_day is not None else 30.0) + (h % 2)  # small intraday wiggle
            rows.append({"market_date": d, "hour_ending": h + 1, "rt_lmp": rt})
    return pd.DataFrame(rows)


def test_trailing_vol_uses_only_days_up_to_d_minus_2():
    # flat prices for 10 days, then a huge spike on day index 10
    rt = [30.0] * 10 + [5000.0] + [30.0] * 3
    df = daily_frame(len(rt), rt)
    vol = df.assign(v=trailing_rt_volatility(df)).groupby("market_date")["v"].first().to_numpy()
    # the spike (day 10) must not affect its own day or the next day (D-1 is incomplete at decision time) ...
    assert vol[10] == pytest.approx(vol[9])
    assert vol[11] == pytest.approx(vol[9])
    # ... and must first appear two days later
    assert vol[12] > 100


def test_trailing_vol_needs_full_window():
    df = daily_frame(8)
    vol = trailing_rt_volatility(df)
    first_days = df["market_date"] < df["market_date"].min() + pd.Timedelta(days=8)
    assert vol[first_days].isna().all()       # window of 7 complete days ending D-2 not yet available


def test_load_regime_bins():
    thr = np.array([10.0, 20.0, 30.0])
    out = load_regime(pd.Series([5, 10, 25, 30, 99, np.nan]), thr)
    assert out.astype(object).tolist()[:5] == ["Low", "Normal", "High", "Extreme", "Extreme"]   # bins are [lo, hi)
    assert pd.isna(out.iloc[-1])


def test_season_mapping():
    s = season(pd.Series([1, 4, 7, 10, 12]))
    assert list(s) == ["Winter", "Spring", "Summer", "Fall", "Winter"]


def test_thresholds_are_fitted_on_training_rows_only():
    n = 400
    df = pd.DataFrame({
        "market_date": pd.date_range("2024-01-01", periods=n, freq="h").normalize(),
        "month": 1, "day_of_week": 0, "rt_lmp": 30.0,
        "forecast_load": np.r_[np.linspace(50, 100, n // 2), np.full(n // 2, 1000.0)],
        "actual_load": np.r_[np.linspace(50, 100, n // 2), np.full(n // 2, 1000.0)],
    })
    fit = pd.Series([True] * (n // 2) + [False] * (n // 2))
    _, thr = add_regimes(df, fit)
    # the huge later values must not move the cut points
    assert thr["forecast_load_MW"].max() <= 100


def test_extreme_flags_use_fit_rows_only():
    df = pd.DataFrame({"rt_lmp": np.r_[np.arange(100.0), np.full(100, 1e4)],
                       "da_rt_spread": np.r_[np.arange(100.0) - 50, np.full(100, 1e4)]})
    fit = pd.Series([True] * 100 + [False] * 100)
    out, thr = extreme_flags(df, fit, pct=0.01)
    assert thr["rt_high"] < 100
    assert out.loc[~fit, "ev_rt_high"].all()


def test_composition_lift():
    df = pd.DataFrame({"cat": ["a"] * 90 + ["b"] * 10, "ev": [False] * 90 + [True] * 10})
    lift = composition_lift(df, "ev", "cat")
    assert lift.loc["b", "lift"] == pytest.approx(10.0)
    assert lift.loc["a", "lift"] == 0


def test_distribution_summary_shares_and_tail_concentration():
    s = pd.Series([-1.0] * 99 + [100.0])
    out = distribution_summary(s)
    assert out["share_negative"] == pytest.approx(0.99)
    assert out["share_of_positive_spread_from_top1pct_hours"] == pytest.approx(1.0)
