"""Tests for backtest metrics and breakdowns."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.metrics import breakdown, mean_daily_pnl_ci, summary


@pytest.fixture
def toy():
    days = pd.Series(pd.to_datetime(["2025-07-01"] * 3 + ["2025-07-02"] * 3))
    pos = pd.Series([-1.0, -1.0, 0.0, -1.0, 1.0, 0.0])
    spread = pd.Series([-5.0, 10.0, 7.0, -2.0, 4.0, 1.0])
    pnl = pos * spread - 0.5 * pos.abs()
    return pos, spread, pnl, days


def test_summary_counts_and_costs(toy):
    pos, spread, pnl, days = toy
    s = summary(pos, spread, pnl, days)
    assert s["hours_traded"] == 4 and s["mwh_traded"] == 4
    assert s["gross_pnl"] == pytest.approx(5 - 10 + 2 + 4)
    assert s["costs"] == pytest.approx(2.0)
    assert s["net_pnl"] == pytest.approx(-1.0)
    assert s["hit_rate"] == pytest.approx(3 / 4)
    assert s["share_short"] == pytest.approx(3 / 4)
    assert s["profit_factor"] == pytest.approx((4.5 + 1.5 + 3.5) / 10.5)


def test_breakdown_sums_to_total(toy):
    pos, spread, pnl, days = toy
    b = breakdown(pnl, pos, days)
    assert b["net_pnl"].sum() == pytest.approx(pnl.sum())
    assert b["hours_traded"].tolist() == [2, 2]


def test_block_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(0)
    days = pd.Series(np.repeat(pd.date_range("2025-07-01", periods=120), 24))
    pnl = pd.Series(rng.normal(1.0, 5.0, len(days)))
    m, lo, hi = mean_daily_pnl_ci(pnl, days, n_boot=1000)
    assert lo < m < hi
    assert m == pytest.approx(pnl.groupby(days).sum().mean())
