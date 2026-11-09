"""Hypothetical P&L for hourly DA/RT spread positions.

    pnl_t = position_t x (RT_t - DA_t) - cost x |position_t|        ($ per hour, positions in MW)

Research simplifications (stated, not hidden):
* price-taker: positions do not move DA or RT prices;
* virtual bids always clear at the DA price (as if offered/bid at the price cap/floor);
* a single flat cost per cleared MWh stands in for PJM transaction fees and uplift/deviation
  charges (base assumption $0.50/MWh; Phase 7 tests $0-2);
* settlement on hourly Western Hub LMPs (real virtuals settle on 5-minute RT intervals);
* no credit/collateral constraints, no position limits beyond 1 MW per hour.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BASE_COST_PER_MWH = 0.50


def hourly_pnl(position: pd.Series, spread: pd.Series, cost_per_mwh: float = BASE_COST_PER_MWH) -> pd.Series:
    pos = position.astype(float)
    return pos * spread.astype(float) - cost_per_mwh * pos.abs()


def daily_pnl(pnl: pd.Series, market_date: pd.Series) -> pd.Series:
    """Sum hourly P&L into operating days (every day in the range appears, 0 if nothing traded)."""
    d = pnl.groupby(market_date.to_numpy()).sum()
    return d.reindex(pd.date_range(d.index.min(), d.index.max(), freq="D"), fill_value=0.0)


def sharpe_like(daily: pd.Series, periods_per_year: int = 365) -> float:
    """Annualized mean / std of daily P&L (no risk-free rate; a ranking metric, not a fund Sharpe)."""
    sd = daily.std(ddof=1)
    return float(daily.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else np.nan


def drawdown_series(daily: pd.Series) -> pd.Series:
    """Cumulative P&L minus its running peak, where the running peak starts at 0 (P&L before day 1)."""
    cum = daily.cumsum()
    return cum - cum.cummax().clip(lower=0.0)


def max_drawdown(daily: pd.Series) -> float:
    """Largest peak-to-trough fall of cumulative P&L ($, reported as a negative number).

    The starting point (cumulative P&L = 0 before the first day) counts as a peak, so losses on
    the very first days are included.
    """
    return float(min(0.0, drawdown_series(daily).min())) if len(daily) else 0.0
