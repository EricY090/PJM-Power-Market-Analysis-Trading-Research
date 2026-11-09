"""End-to-end tests for saved strategy files and the backtest runner, on synthetic data (offline).

Covers the user-facing workflow: any saved strategy file can be evaluated (`run --strategy`), its
parameters (thresholds, cut-offs, cost) drive the positions, and nothing else is required to run it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from src.analysis.regimes import fit_load_thresholds
from src.backtest import run as run_mod
from src.backtest import strategy as st
from src.data.features import build_features
from tests.test_features import synthetic_hourly


@pytest.fixture(scope="module")
def features(tmp_path_factory):
    """Synthetic features with train / validation / test periods, saved as features.parquet."""
    hourly = synthetic_hourly(start="2024-08-01", end="2024-12-31", seed=3)
    md = hourly["market_date"]
    hourly["period"] = np.select(
        [md < "2024-09-01", md < "2024-11-01", md < "2024-12-01"], ["lookback", "train", "validation"], "test")
    thr = fit_load_thresholds(hourly.loc[hourly["period"] == "train", "forecast_load"])
    F = build_features(hourly, thr)
    d = tmp_path_factory.mktemp("processed")
    F.to_parquet(d / "features.parquet", index=False)
    return F, d


@pytest.fixture
def patched(features, monkeypatch):
    F, d = features
    monkeypatch.setattr(run_mod, "load_config", lambda: {"paths": {"processed": d}})
    return F


def strategy_file(path, s4=(0.7, 0.3), s5_cutoff=1.7, s6_ref=2.4, cost=0.5):
    s = {
        "primary_signal": "S4_prob_threshold",
        "primary_params": {"upper": s4[0], "lower": s4[1]},
        "cost_per_mwh": cost,
        "signals": {
            "S0_flat": {}, "S1_always_long": {}, "S2_always_short": {},
            "S3_rule_based": {"flat_load_regimes": ["High", "Extreme"], "flat_hours": [17, 18, 19, 20, 21]},
            "S4_prob_threshold": {"upper": s4[0], "lower": s4[1]},
            "S5_risk_filtered_short": {"percentile": 40.0, "cutoff_log1p_abs_spread": s5_cutoff},
            "S6_risk_sized_short": {"reference_percentile": 10.0, "reference_abs_spread": s6_ref},
        },
    }
    path.write_text(yaml.safe_dump(s, sort_keys=False), encoding="utf-8")
    return path


def test_run_evaluates_a_custom_strategy_file(patched, tmp_path):
    r = run_mod.run(strategy_path=strategy_file(tmp_path / "mine.yaml"))
    assert r.strategy_path.endswith("mine.yaml")
    assert set(r.positions) == set(run_mod.SIGNALS)
    assert (r.rows["period"] == "test").all() and len(r.rows) > 0
    tbl = r.summary_table()
    assert tbl.loc["S0_flat", "net_pnl"] == 0
    # always long and always short are mirror images before costs
    assert tbl.loc["S1_always_long", "gross_pnl"] == pytest.approx(-tbl.loc["S2_always_short", "gross_pnl"])


def test_strategy_file_parameters_drive_positions(patched, tmp_path):
    trade_all = run_mod.run(strategy_path=strategy_file(tmp_path / "a.yaml", s4=(1.0, 1.0), s5_cutoff=1e9))
    trade_none = run_mod.run(strategy_path=strategy_file(tmp_path / "b.yaml", s4=(1.0, 0.0), s5_cutoff=-1e9))
    valid = trade_all.positions["S2_always_short"] != 0
    # S4 with lower = 1.0 shorts every usable hour; with lower = 0.0 it never trades
    assert (trade_all.positions["S4_prob_threshold"][valid] == -1).all()
    assert (trade_none.positions["S4_prob_threshold"] == 0).all()
    # S5 cut-off read from the file: everything below +inf is shorted, nothing is below -inf
    assert (trade_all.positions["S5_risk_filtered_short"][valid] == -1).all()
    assert (trade_none.positions["S5_risk_filtered_short"] == 0).all()


def test_cost_comes_from_the_strategy_file(patched, tmp_path):
    cheap = run_mod.run(strategy_path=strategy_file(tmp_path / "c0.yaml", cost=0.0))
    dear = run_mod.run(strategy_path=strategy_file(tmp_path / "c2.yaml", cost=2.0))
    mwh = cheap.positions["S2_always_short"].abs().sum()
    assert cheap.pnl["S2_always_short"].sum() - dear.pnl["S2_always_short"].sum() == pytest.approx(2.0 * mwh)


def test_positions_for_matches_signal_functions(patched, tmp_path):
    """positions_for applies the file's rule-based settings exactly."""
    s = st.load_strategy(strategy_file(tmp_path / "r.yaml"))
    r = run_mod.run(strategy_path=tmp_path / "r.yaml")
    pos = st.positions_for(s, "S3_rule_based", r.rows, r.models)
    risky = r.rows["fc_load_regime"].astype(str).isin(["High", "Extreme"]) | r.rows["hour_ending"].between(17, 21)
    assert (pos[risky] == 0).all()
    assert set(pos[~risky].unique()) <= {-1.0, 0.0}


def test_only_the_test_period_is_evaluated(patched, tmp_path):
    with pytest.raises(ValueError, match="holdout"):
        run_mod.run(period="holdout", strategy_path=strategy_file(tmp_path / "h.yaml"))


def test_saved_project_strategy_needs_no_extra_metadata():
    """The committed strategy file is plain parameters: no guard/integrity blocks are required."""
    if not st.STRATEGY_PATH.exists():
        pytest.skip("no saved strategy")
    s = st.load_strategy()
    assert "integrity" not in s
    assert set(run_mod.SIGNALS) <= set(s["signals"])
