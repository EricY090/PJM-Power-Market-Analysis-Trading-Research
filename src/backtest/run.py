"""Evaluate a saved (frozen) strategy on the test period.

    python -m src.backtest.run                                  # config/strategy.yaml
    python -m src.backtest.run --strategy config/my_strategy.yaml

Models are refit on train + validation exactly as at freeze time, positions come from the saved
rules and cut-offs, and every signal in the file is evaluated against the same hours.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.backtest import strategy as st
from src.backtest.engine import hourly_pnl
from src.backtest.metrics import mean_daily_pnl_ci, summary
from src.config import load_config

SIGNALS = ["S0_flat", "S1_always_long", "S2_always_short", "S3_rule_based",
           "S4_prob_threshold", "S5_risk_filtered_short", "S6_risk_sized_short"]


@dataclass
class BacktestResult:
    period: str
    strategy_path: str
    strategy: dict
    rows: pd.DataFrame                                  # evaluation hours with features
    positions: dict[str, pd.Series] = field(default_factory=dict)
    pnl: dict[str, pd.Series] = field(default_factory=dict)
    models: dict = field(default_factory=dict)

    def summary_table(self) -> pd.DataFrame:
        out = {}
        for s in self.positions:
            row = summary(self.positions[s], self.rows["da_rt_spread"], self.pnl[s], self.rows["market_date"])
            m, lo, hi = mean_daily_pnl_ci(self.pnl[s], self.rows["market_date"])
            out[s] = pd.concat([row, pd.Series({"mean_daily_pnl": m, "ci95_low": lo, "ci95_high": hi})])
        return pd.DataFrame(out).T


def run(period: str = "test", strategy_path: Path = st.STRATEGY_PATH) -> BacktestResult:
    if period != "test":
        raise ValueError("only the test period is evaluated here; the 2026 holdout has its own step (6.5)")
    strategy = st.load_strategy(strategy_path)
    cost = float(strategy["cost_per_mwh"])
    F = pd.read_parquet(load_config()["paths"]["processed"] / "features.parquet")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        models, _, _ = st.fit_final_models(F)

    rows = F[F["period"] == period].copy()
    res = BacktestResult(period, str(strategy_path), strategy, rows, models=models)
    for s in SIGNALS:
        pos = st.positions_for(strategy, s, rows, models)
        res.positions[s] = pos
        res.pnl[s] = hourly_pnl(pos, rows["da_rt_spread"], cost)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--strategy", type=Path, default=st.STRATEGY_PATH, help="saved strategy file")
    args = ap.parse_args()
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    r = run(strategy_path=args.strategy)
    print(f"period={r.period} strategy={r.strategy_path} rows={len(r.rows)} "
          f"{r.rows['market_date'].min().date()} -> {r.rows['market_date'].max().date()}")
    print(r.summary_table().round(3).to_string())
