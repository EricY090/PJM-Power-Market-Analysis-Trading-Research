"""Backtest metrics: summary statistics, breakdowns and uncertainty for hourly spread positions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.engine import daily_pnl, max_drawdown, sharpe_like


def summary(pos: pd.Series, spread: pd.Series, pnl: pd.Series, market_date: pd.Series) -> pd.Series:
    """Headline statistics for one signal. P&L in $ for 1 MW per traded hour."""
    traded = pos != 0
    daily = daily_pnl(pnl, market_date)
    gross = (pos * spread)[traded]
    wins, losses = pnl[traded & (pnl > 0)].sum(), -pnl[traded & (pnl < 0)].sum()
    return pd.Series({
        "net_pnl": pnl.sum(),
        "gross_pnl": gross.sum(),
        "costs": gross.sum() - pnl.sum(),
        "mwh_traded": pos.abs().sum(),
        "hours_traded": int(traded.sum()),
        "coverage": traded.mean(),
        "share_short": (pos < 0).sum() / max(1, traded.sum()),
        "pnl_per_mwh": pnl.sum() / pos.abs().sum() if pos.abs().sum() > 0 else np.nan,
        "hit_rate": (pnl[traded] > 0).mean() if traded.any() else np.nan,
        "avg_win": pnl[traded & (pnl > 0)].mean() if (traded & (pnl > 0)).any() else np.nan,
        "avg_loss": pnl[traded & (pnl < 0)].mean() if (traded & (pnl < 0)).any() else np.nan,
        "profit_factor": wins / losses if losses > 0 else np.nan,
        "hourly_std": pnl[traded].std() if traded.sum() > 1 else np.nan,
        "sharpe_like": sharpe_like(daily),
        "max_drawdown": max_drawdown(daily),
        "best_day": daily.max(),
        "worst_day": daily.min(),
        "share_positive_days": (daily[daily != 0] > 0).mean() if (daily != 0).any() else np.nan,
        "worst_5_days_share_of_losses": (daily.nsmallest(5).clip(upper=0).sum() / daily.clip(upper=0).sum()
                                          if (daily < 0).any() else np.nan),
    })


def mean_daily_pnl_ci(pnl: pd.Series, market_date: pd.Series, n_boot: int = 5000, seed: int = 0,
                      block_days: int = 5) -> tuple[float, float, float]:
    """Mean daily P&L with a 95% moving-block bootstrap CI (blocks of consecutive days)."""
    daily = daily_pnl(pnl, market_date).to_numpy()
    n = len(daily)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_days))
    starts = rng.integers(0, n - block_days + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block_days)).reshape(n_boot, -1)[:, :n]
    boots = daily[idx].mean(axis=1)
    return float(daily.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def breakdown(pnl: pd.Series, pos: pd.Series, by: pd.Series) -> pd.DataFrame:
    """Net P&L, traded MWh, P&L per MWh and hit rate per category."""
    traded = pos != 0
    df = pd.DataFrame({"pnl": pnl, "mwh": pos.abs(), "traded": traded,
                       "win": traded & (pnl > 0), "key": by.to_numpy()})
    g = df.groupby("key", observed=True)
    out = pd.DataFrame({"net_pnl": g["pnl"].sum(), "mwh_traded": g["mwh"].sum(),
                        "hours": g.size(), "hours_traded": g["traded"].sum(), "wins": g["win"].sum()})
    out["pnl_per_mwh"] = out["net_pnl"] / out["mwh_traded"].replace(0, np.nan)
    out["hit_rate"] = out["wins"] / out["hours_traded"].replace(0, np.nan)
    return out.drop(columns="wins")
