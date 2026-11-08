"""Relationships between fundamentals and the spread: forecast error, extremes, event profiles."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def correlations(df: pd.DataFrame, x: str, y: str) -> pd.Series:
    """Pearson and Spearman correlation with p-values, on rows where both exist.

    Note: hourly observations are autocorrelated, so p-values overstate significance;
    the bucketed and day-level views are the more honest evidence.
    """
    d = df[[x, y]].dropna()
    pr, pp = stats.pearsonr(d[x], d[y])
    sr, sp = stats.spearmanr(d[x], d[y])
    return pd.Series({"n": len(d), "pearson_r": pr, "pearson_p": pp, "spearman_rho": sr, "spearman_p": sp})


def bucket_table(
    df: pd.DataFrame, x: str, y: str = "da_rt_spread", q: int = 10,
    extreme_flag: str | None = None, edges: np.ndarray | None = None,
) -> pd.DataFrame:
    """Bucket ``x`` into quantiles (or given ``edges``) and summarise ``y`` per bucket."""
    d = df.dropna(subset=[x, y]).copy()
    if edges is None:
        d["bucket"] = pd.qcut(d[x], q=q, duplicates="drop")
    else:
        d["bucket"] = pd.cut(d[x], bins=edges, include_lowest=True)
    g = d.groupby("bucket", observed=True)
    out = pd.DataFrame({
        "hours": g.size(),
        f"{x}_mid": g[x].median(),
        "mean_spread": g[y].mean(),
        "median_spread": g[y].median(),
        "se_mean": g[y].std() / np.sqrt(g.size()),
        "share_rt_above_da": g[y].apply(lambda s: (s > 0).mean()),
        "mean_abs_spread": g[y].apply(lambda s: s.abs().mean()),
    })
    if extreme_flag is not None:
        out["share_extreme"] = g[extreme_flag].mean()
    return out


def extreme_flags(df: pd.DataFrame, fit_mask: pd.Series, pct: float = 0.01) -> tuple[pd.DataFrame, dict]:
    """Flag top/bottom ``pct`` RT LMP and top ``pct`` |spread| hours using thresholds from ``fit_mask`` rows."""
    out = df.copy()
    fit = out.loc[fit_mask]
    thr = {
        "rt_high": fit["rt_lmp"].quantile(1 - pct),
        "rt_low": fit["rt_lmp"].quantile(pct),
        "abs_spread_high": fit["da_rt_spread"].abs().quantile(1 - pct),
    }
    out["ev_rt_high"] = out["rt_lmp"] >= thr["rt_high"]
    out["ev_rt_low"] = out["rt_lmp"] <= thr["rt_low"]
    out["ev_abs_spread"] = out["da_rt_spread"].abs() >= thr["abs_spread_high"]
    out["ev_spread_pos"] = out["ev_abs_spread"] & (out["da_rt_spread"] > 0)
    out["ev_spread_neg"] = out["ev_abs_spread"] & (out["da_rt_spread"] < 0)
    return out, thr


def composition_lift(df: pd.DataFrame, event: str, by: str) -> pd.DataFrame:
    """Share of event hours in each category vs the share of all hours ("lift" = ratio)."""
    base = df[by].value_counts(normalize=True).rename("share_all_hours")
    ev = df.loc[df[event], by].value_counts(normalize=True).rename("share_event_hours")
    counts = df.loc[df[event], by].value_counts().rename("event_hours")
    out = pd.concat([counts, ev, base], axis=1).fillna(0)
    out["lift"] = out["share_event_hours"] / out["share_all_hours"]
    return out.sort_index()


def event_profile(df: pd.DataFrame, events: dict[str, str], cols: list[str]) -> pd.DataFrame:
    """Median of selected columns for each event type next to the all-hours baseline."""
    rows = {"All hours": df[cols].median(numeric_only=True)}
    rows["All hours"]["hours"] = len(df)
    for label, flag in events.items():
        sub = df.loc[df[flag]]
        r = sub[cols].median(numeric_only=True)
        r["hours"] = len(sub)
        rows[label] = r
    return pd.DataFrame(rows).T[["hours", *cols]]


def event_days(df: pd.DataFrame, flag: str) -> pd.DataFrame:
    """Group event hours into operating days, to see whether events cluster in a few episodes."""
    ev = df.loc[df[flag]]
    days = ev.groupby("market_date").agg(
        hours=("timestamp_utc", "size"),
        max_rt=("rt_lmp", "max"),
        max_spread=("da_rt_spread", "max"),
        min_spread=("da_rt_spread", "min"),
        peak_load=("actual_load", "max"),
        mean_fc_error=("load_forecast_error", "mean"),
    )
    return days.sort_values("hours", ascending=False)
