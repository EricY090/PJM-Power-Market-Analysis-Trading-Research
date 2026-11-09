"""Tests for trading signals, the P&L engine and the frozen strategy file."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from src.backtest import signals as sg
from src.backtest.engine import daily_pnl, hourly_pnl, max_drawdown, sharpe_like
from src.backtest.strategy import STRATEGY_PATH, load_strategy
from src.data.features import tradable_features


@pytest.fixture
def rows():
    return pd.DataFrame({
        "hour_ending": [1, 18, 3, 20, 5, 6],
        "fc_load_regime": ["Low", "Low", "Extreme", "Normal", "High", "Normal"],
    })


def test_positions_bounded_and_zero_when_invalid(rows):
    valid = pd.Series([True, True, True, True, False, True])
    for pos in (sg.always_long(rows, valid), sg.always_short(rows, valid), sg.rule_based(rows, valid),
                sg.prob_threshold(pd.Series(np.linspace(0, 1, 6)), valid),
                sg.risk_filtered_short(pd.Series(np.arange(6.0)), valid, cutoff=3.0),
                sg.risk_sized_short(pd.Series(np.arange(6.0)), valid, reference=2.0)):
        assert pos.between(-1, 1).all()
        assert pos.iloc[4] == 0                       # invalid row -> flat
        assert pos.notna().all()


def test_nan_predictions_give_flat(rows):
    valid = pd.Series(True, index=rows.index)
    p = pd.Series([np.nan, 0.1, 0.9, np.nan, 0.5, 0.2])
    pos = sg.prob_threshold(p, valid, 0.6, 0.4)
    assert pos.tolist() == [0, -1, 1, 0, 0, -1]


def test_rule_based_flat_in_spike_prone_hours(rows):
    pos = sg.rule_based(rows, pd.Series(True, index=rows.index))
    # HE18/HE20 (evening) and High/Extreme forecast-load rows are flat; the rest short
    assert pos.tolist() == [-1, 0, 0, 0, 0, -1]


def test_prob_threshold_rejects_bad_thresholds(rows):
    with pytest.raises(ValueError):
        sg.prob_threshold(pd.Series([0.5] * 6), pd.Series(True, index=rows.index), upper=0.4, lower=0.6)


def test_risk_filter_and_sizing():
    valid = pd.Series(True, index=range(4))
    risk = pd.Series(np.log1p([1.0, 5.0, 20.0, 80.0]))
    assert sg.risk_filtered_short(risk, valid, cutoff=np.log1p(10.0)).tolist() == [-1, -1, 0, 0]
    sized = sg.risk_sized_short(risk, valid, reference=5.0)
    assert sized.tolist() == pytest.approx([-1.0, -1.0, -0.25, -0.0625])
    assert sg.risk_cutoff(np.arange(101.0), 40) == pytest.approx(40.0)


def test_engine_pnl_cost_and_metrics():
    pos = pd.Series([1.0, -1.0, 0.0, -0.5])
    spread = pd.Series([10.0, 10.0, 99.0, -4.0])
    pnl = hourly_pnl(pos, spread, cost_per_mwh=0.5)
    assert pnl.tolist() == pytest.approx([9.5, -10.5, 0.0, 1.75])
    days = pd.Series(pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-03", "2024-01-03"]))
    d = daily_pnl(pnl, days)
    assert d.tolist() == pytest.approx([-1.0, 0.0, 1.75])            # missing day filled with 0
    assert max_drawdown(pd.Series([5.0, -10.0, 3.0])) == pytest.approx(-10.0)
    # losses from the very start count: the peak before day 1 is 0
    assert max_drawdown(pd.Series([-4.0, -3.0, 10.0, -1.0])) == pytest.approx(-7.0)
    assert max_drawdown(pd.Series([2.0, 3.0])) == 0.0
    assert np.isnan(sharpe_like(pd.Series([1.0, 1.0])))               # zero variance -> undefined


def test_signals_never_need_ex_post_columns(rows):
    """Signals run on a frame that contains no ex-post column at all."""
    assert set(rows.columns) <= set(tradable_features())
    sg.rule_based(rows, pd.Series(True, index=rows.index))


@pytest.mark.skipif(not STRATEGY_PATH.exists(), reason="strategy not frozen yet")
def test_frozen_strategy_file_is_complete_and_round_trips(tmp_path):
    s = load_strategy()
    assert s["primary_signal"] in s["signals"]
    for key in ("frozen_at", "cost_per_mwh", "selection", "models", "position_convention"):
        assert key in s
    assert "pooled_results" not in s["selection"]                     # no PJM-derived numbers in the committed config
    assert s["models"]["fit_periods"] == ["train", "validation"]
    copy = tmp_path / "s.yaml"
    copy.write_text(yaml.safe_dump(s, sort_keys=False), encoding="utf-8")
    assert load_strategy(copy) == s
