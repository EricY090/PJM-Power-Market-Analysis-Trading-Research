"""Regime labels: season, weekday/weekend, load regime, trailing-volatility regime.

Thresholds are always *fitted* on one slice of data (the training period) and then
*applied* everywhere, so regime labels never use information from later periods.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEASONS = ["Winter", "Spring", "Summer", "Fall"]
LOAD_REGIMES = ["Low", "Normal", "High", "Extreme"]
# percentile edges for the load regimes: <25th, 25-75th, 75-95th, >95th
LOAD_REGIME_PCTS = (0.25, 0.75, 0.95)

EDA_PERIODS = ("train", "validation")   # periods used for exploratory analysis


def eda_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Rows used for exploratory analysis: train + validation only (test/holdout untouched)."""
    return df.loc[df["period"].isin(EDA_PERIODS)].copy()


def season(month: pd.Series) -> pd.Series:
    """Meteorological seasons: Winter = DJF, Spring = MAM, Summer = JJA, Fall = SON."""
    lookup = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
              6: "Summer", 7: "Summer", 8: "Summer", 9: "Fall", 10: "Fall", 11: "Fall"}
    return pd.Categorical(month.map(lookup), categories=SEASONS, ordered=True)


def is_weekend(day_of_week: pd.Series) -> pd.Series:
    return day_of_week >= 5


def fit_load_thresholds(load: pd.Series, pcts: tuple[float, ...] = LOAD_REGIME_PCTS) -> np.ndarray:
    """Percentile cut points (MW) estimated on the fitting sample."""
    return load.dropna().quantile(list(pcts)).to_numpy()


def load_regime(load: pd.Series, thresholds: np.ndarray) -> pd.Series:
    """Low / Normal / High / Extreme from fitted percentile thresholds."""
    bins = [-np.inf, *thresholds, np.inf]
    return pd.cut(load, bins=bins, labels=LOAD_REGIMES, right=False)


def trailing_rt_volatility(df: pd.DataFrame, window_days: int = 7) -> pd.Series:
    """Std of hourly RT LMP over the ``window_days`` complete days ending on D-2.

    For operating day D the position is fixed on the morning of D-1, when day D-1 is
    still in progress, so the last complete day known is D-2. Every hour of day D gets
    the same value. Requires the full hourly frame (including the lookback month).
    """
    daily = df.groupby("market_date")["rt_lmp"].agg(["sum", "count"])
    daily["sumsq"] = df.assign(sq=df["rt_lmp"] ** 2).groupby("market_date")["sq"].sum()
    # shift by 2 days: the window for D ends at D-2
    roll = daily.rolling(window_days, min_periods=window_days).sum().shift(2)
    mean = roll["sum"] / roll["count"]
    var = roll["sumsq"] / roll["count"] - mean**2
    vol = np.sqrt(var.clip(lower=0) * roll["count"] / (roll["count"] - 1))
    return df["market_date"].map(vol)


def add_regimes(df: pd.DataFrame, fit_mask: pd.Series) -> tuple[pd.DataFrame, dict]:
    """Add season, weekend, load-regime (forecast and actual load) and volatility columns.

    ``fit_mask`` selects the rows used to fit thresholds (normally the training period).
    Returns the frame and the fitted thresholds.
    """
    out = df.copy()
    out["season"] = season(out["month"])
    out["is_weekend"] = is_weekend(out["day_of_week"])
    fc_thr = fit_load_thresholds(out.loc[fit_mask, "forecast_load"])
    act_thr = fit_load_thresholds(out.loc[fit_mask, "actual_load"])
    out["load_regime"] = load_regime(out["forecast_load"], fc_thr)              # known ex-ante
    out["load_regime_actual"] = load_regime(out["actual_load"], act_thr)        # ex-post
    out["rt_vol_7d"] = trailing_rt_volatility(out)
    vol_thr = out.loc[fit_mask, "rt_vol_7d"].dropna().median()
    out["vol_regime"] = pd.Categorical(
        np.where(out["rt_vol_7d"].isna(), None, np.where(out["rt_vol_7d"] >= vol_thr, "High vol", "Low vol")),
        categories=["Low vol", "High vol"], ordered=True,
    )
    thresholds = {"forecast_load_MW": fc_thr, "actual_load_MW": act_thr, "rt_vol_7d_median": vol_thr}
    return out, thresholds
