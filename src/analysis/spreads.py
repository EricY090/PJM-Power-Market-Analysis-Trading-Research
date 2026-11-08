"""DA/RT spread statistics: distribution summary and grouped statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd

PCTS = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


def distribution_summary(s: pd.Series, tail_thresholds: tuple[float, ...] = (50, 100, 250)) -> pd.Series:
    """Mean, median, dispersion, shape, percentiles, sign frequencies and tail counts."""
    s = s.dropna()
    out = {
        "hours": len(s),
        "mean": s.mean(),
        "median": s.median(),
        "std": s.std(),
        "mad_from_median": (s - s.median()).abs().median(),
        "skew": s.skew(),
        "excess_kurtosis": s.kurt(),
        "min": s.min(),
        "max": s.max(),
        **{f"p{int(p * 100):02d}": s.quantile(p) for p in PCTS},
        "share_positive": (s > 0).mean(),
        "share_negative": (s < 0).mean(),
        "mean_if_positive": s[s > 0].mean(),
        "mean_if_negative": s[s < 0].mean(),
    }
    for t in tail_thresholds:
        out[f"hours_above_+{t}"] = int((s > t).sum())
        out[f"hours_below_-{t}"] = int((s < -t).sum())
    # how concentrated is the total spread? share of the summed positive spread from the top 1% of hours
    pos = s.clip(lower=0).sort_values(ascending=False)
    out["share_of_positive_spread_from_top1pct_hours"] = pos.iloc[: max(1, len(s) // 100)].sum() / pos.sum()
    return pd.Series(out)


def group_stats(df: pd.DataFrame, by, col: str = "da_rt_spread") -> pd.DataFrame:
    """Per-group spread statistics (level, dispersion, sign frequency, tails)."""
    g = df.groupby(by, observed=True)[col]
    out = pd.DataFrame({
        "hours": g.size(),
        "mean": g.mean(),
        "median": g.median(),
        "std": g.std(),
        "p05": g.quantile(0.05),
        "p95": g.quantile(0.95),
        "mean_abs": g.apply(lambda s: s.abs().mean()),
        "share_positive": g.apply(lambda s: (s > 0).mean()),
    })
    return out


def mean_ci(s: pd.Series, z: float = 1.96) -> tuple[float, float, float]:
    """Mean and a normal-approximation 95% confidence interval."""
    s = s.dropna()
    m, se = s.mean(), s.std(ddof=1) / np.sqrt(len(s))
    return m, m - z * se, m + z * se


def daily_block_bootstrap_mean_ci(
    df: pd.DataFrame, col: str = "da_rt_spread", n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean with a 95% CI that resamples whole days (hours within a day are correlated)."""
    daily = df.groupby("market_date")[col].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(daily), size=(n_boot, len(daily)))
    sums, counts = daily["sum"].to_numpy()[idx].sum(1), daily["count"].to_numpy()[idx].sum(1)
    boots = sums / counts
    return daily["sum"].sum() / daily["count"].sum(), np.quantile(boots, 0.025), np.quantile(boots, 0.975)
