"""Feature engineering for the DA/RT spread models.

Decision time: **10:30 EPT on D-1** (before the day-ahead auction closes). A spread position
for operating day D (e.g. a virtual INC/DEC) is submitted then, so a *tradable* feature may only
use information published before that moment:

=====================  ===================================  =================================
Data                   Published                            Usable for day D
=====================  ===================================  =================================
Load forecast (D)      09:45 EPT on D-1                     forecast for D (and earlier days)
DA LMP (day X)         ~13:30 EPT on X-1                    DA up to D-1
RT LMP, actual load    shortly after each hour              complete days up to **D-2**
Forecast error (X)     once day X is over                   up to D-2
=====================  ===================================  =================================

Real-time data for the morning of D-1 is technically observable (unverified) by 10:30, but we
conservatively stop at the last *complete* day, D-2. The DA LMP for D itself is **ex-post**
here: it is set by the auction the position is bid into.

Every output column is registered in ``FEATURES`` with its group and availability
(``tradable`` vs ``ex_post``); models must only use ``tradable_features()``.

Run:  python -m src.data.features   -> data/processed/features.parquet
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, USLaborDay, USMemorialDay, USThanksgivingDay

from src.analysis.regimes import fit_load_thresholds, load_regime, season

log = logging.getLogger(__name__)

SPREAD_CLIP = 100.0   # $/MWh; trailing spread means use clipped values so one storm (e.g. Elliott) can't dominate


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Feature:
    group: str          # calendar | fundamental | forecast_bias | market | target | ex_post
    availability: str   # tradable | ex_post
    description: str


FEATURES: dict[str, Feature] = {
    # calendar (known in advance)
    "hour_ending": Feature("calendar", "tradable", "Hour ending 1-24 (EPT)"),
    "day_of_week": Feature("calendar", "tradable", "Day of week of D, Monday = 0"),
    "is_weekend": Feature("calendar", "tradable", "D is Saturday or Sunday"),
    "is_holiday": Feature("calendar", "tradable", "D is a NERC holiday"),
    "month": Feature("calendar", "tradable", "Calendar month of D"),
    "season": Feature("calendar", "tradable", "Winter/Spring/Summer/Fall (meteorological)"),
    # fundamentals from the D-1 09:45 load forecast
    "fc_load": Feature("fundamental", "tradable", "D-1 09:45 RTO load forecast for this hour (MW)"),
    "fc_load_regime": Feature("fundamental", "tradable", "Low/Normal/High/Extreme from fc_load (train percentiles)"),
    "fc_daily_peak": Feature("fundamental", "tradable", "Max forecast load over day D (MW)"),
    "fc_load_vs_peak": Feature("fundamental", "tradable", "fc_load / fc_daily_peak"),
    "fc_ramp_1h": Feature("fundamental", "tradable", "fc_load minus previous hour's forecast (HE1 uses D-1 HE24) (MW)"),
    "fc_change_vs_prev_day": Feature("fundamental", "tradable", "fc_load(D,h) - fc_load(D-1,h); D-1 forecast was made on D-2"),
    "fc_peak_change_vs_prev_day": Feature("fundamental", "tradable", "fc_daily_peak(D) - fc_daily_peak(D-1)"),
    "fc_vs_actual_d2": Feature("fundamental", "tradable", "fc_load(D,h) - actual load(D-2,h) (MW)"),
    # recent forecast bias (proxy for the ex-post forecast error)
    "fc_err_d2_same_hour": Feature("forecast_bias", "tradable", "Actual - forecast at the same hour on D-2 (MW)"),
    "fc_err_d2_daily_mean": Feature("forecast_bias", "tradable", "Mean (actual - forecast) over D-2 (MW)"),
    "fc_err_7d_mean": Feature("forecast_bias", "tradable", "Mean (actual - forecast) over the 7 days ending D-2 (MW)"),
    "fc_err_pct_7d_mean": Feature("forecast_bias", "tradable", "Mean percentage forecast error over the 7 days ending D-2"),
    # market history
    "da_lmp_d1_same_hour": Feature("market", "tradable", "DA LMP at the same hour on D-1 (published on D-2)"),
    "da_lmp_d1_daily_mean": Feature("market", "tradable", "Mean DA LMP over D-1"),
    "rt_lmp_d2_same_hour": Feature("market", "tradable", "RT LMP at the same hour on D-2"),
    "spread_d2_same_hour": Feature("market", "tradable", "RT - DA at the same hour on D-2"),
    "spread_d2_daily_mean": Feature("market", "tradable", "Mean RT - DA over D-2"),
    "spread_same_hour_7d_mean": Feature("market", "tradable", f"Mean same-hour spread over 7 days ending D-2 (clipped ±${SPREAD_CLIP:.0f})"),
    "spread_7d_mean": Feature("market", "tradable", f"Mean spread over 7 days ending D-2 (clipped ±${SPREAD_CLIP:.0f})"),
    "spread_30d_mean": Feature("market", "tradable", f"Mean spread over 30 days ending D-2 (clipped ±${SPREAD_CLIP:.0f})"),
    "spread_7d_median": Feature("market", "tradable", "Median spread over 7 days ending D-2"),
    "share_rt_above_da_7d": Feature("market", "tradable", "Share of hours with RT > DA over 7 days ending D-2"),
    "rt_vol_7d": Feature("market", "tradable", "Std of hourly RT LMP over 7 days ending D-2"),
    "rt_vol_30d": Feature("market", "tradable", "Std of hourly RT LMP over 30 days ending D-2"),
    "spread_vol_7d": Feature("market", "tradable", "Std of hourly spread over 7 days ending D-2"),
    # target and ex-post columns (never model inputs)
    "da_rt_spread": Feature("target", "ex_post", "RT - DA for this hour (regression target)"),
    "target_rt_above_da": Feature("target", "ex_post", "1 if RT > DA (classification target)"),
    "da_lmp": Feature("ex_post", "ex_post", "DA LMP for D - set by the auction the position is bid into"),
    "rt_lmp": Feature("ex_post", "ex_post", "RT LMP for D"),
    "actual_load": Feature("ex_post", "ex_post", "Metered load for D"),
    "load_forecast_error": Feature("ex_post", "ex_post", "Actual - forecast for D"),
    "load_regime_actual": Feature("ex_post", "ex_post", "Load regime from actual load"),
}

KEY_COLUMNS = ["timestamp_utc", "timestamp_ept", "market_date", "hour_ending", "is_dst_repeat", "period"]


def tradable_features(groups: tuple[str, ...] | None = None) -> list[str]:
    """Names of features known at decision time, optionally restricted to some groups."""
    return [n for n, f in FEATURES.items() if f.availability == "tradable" and (groups is None or f.group in groups)]


def ex_post_columns() -> list[str]:
    return [n for n, f in FEATURES.items() if f.availability == "ex_post"]


# --------------------------------------------------------------------------- #
# Calendar
# --------------------------------------------------------------------------- #
class NERCHolidayCalendar(AbstractHolidayCalendar):
    """NERC off-peak holidays (Sunday holidays observed on Monday)."""

    def _sunday_to_monday(d):  # noqa: N805 - used as observance function
        return d + pd.Timedelta(days=1) if d.weekday() == 6 else d

    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=_sunday_to_monday),
        USMemorialDay,
        Holiday("Independence Day", month=7, day=4, observance=_sunday_to_monday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas Day", month=12, day=25, observance=_sunday_to_monday),
    ]


def nerc_holidays(start, end) -> pd.DatetimeIndex:
    return NERCHolidayCalendar().holidays(start=start, end=end)


# --------------------------------------------------------------------------- #
# Hour-aligned helpers (DST-safe)
# --------------------------------------------------------------------------- #
def _hour_matrix(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """market_date x hour_ending matrix on a complete daily index.

    The repeated HE2 of a fall-back day is averaged into one cell; the missing HE3 of a
    spring-forward day is NaN. Lags therefore always match the same *clock hour*.
    """
    m = df.pivot_table(index="market_date", columns="hour_ending", values=col, aggfunc="mean", dropna=False)
    full = pd.date_range(df["market_date"].min(), df["market_date"].max(), freq="D", name="market_date")
    return m.reindex(index=full, columns=range(1, 25))


def _lookup(df: pd.DataFrame, matrix: pd.DataFrame) -> np.ndarray:
    """Read matrix[market_date, hour_ending] for every row of ``df``."""
    stacked = matrix.stack(future_stack=True)
    idx = pd.MultiIndex.from_arrays([df["market_date"], df["hour_ending"]])
    return stacked.reindex(idx).to_numpy()


def _daily(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Daily sum, sum of squares and count of ``col`` on a complete daily index."""
    g = df.groupby("market_date")[col]
    d = pd.DataFrame({"sum": g.sum(min_count=1), "sumsq": (df[col] ** 2).groupby(df["market_date"]).sum(min_count=1),
                      "n": g.count()})
    full = pd.date_range(df["market_date"].min(), df["market_date"].max(), freq="D", name="market_date")
    return d.reindex(full)


def _trailing_mean_std(daily: pd.DataFrame, window: int, lag_days: int) -> tuple[pd.Series, pd.Series]:
    """Mean and std over ``window`` days ending ``lag_days`` before D (all hours pooled)."""
    r = daily[["sum", "sumsq", "n"]].fillna(0).rolling(window, min_periods=window).sum().shift(lag_days)
    full_days = daily["n"].notna().astype(int).rolling(window, min_periods=window).sum().shift(lag_days)
    mean = r["sum"] / r["n"]
    var = (r["sumsq"] - r["n"] * mean**2) / (r["n"] - 1)
    ok = full_days >= window * 0.8            # require most of the window to have data
    return mean.where(ok), np.sqrt(var.clip(lower=0)).where(ok)


def _by_date(df: pd.DataFrame, series: pd.Series) -> np.ndarray:
    return df["market_date"].map(series).to_numpy()


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #
def build_features(hourly: pd.DataFrame, load_thresholds: np.ndarray) -> pd.DataFrame:
    """Build all registered features for every hour of ``hourly``.

    ``hourly`` must be the cleaned hourly table *including* enough history before the first
    modelled day (the lookback month). ``load_thresholds`` come from the training period.
    """
    df = hourly.sort_values("timestamp_utc").reset_index(drop=True).copy()
    out = df[KEY_COLUMNS].copy()

    # ---- calendar ---------------------------------------------------------
    out["hour_ending"] = df["hour_ending"]
    out["day_of_week"] = df["day_of_week"]
    out["is_weekend"] = df["day_of_week"] >= 5
    hol = nerc_holidays(df["market_date"].min(), df["market_date"].max())
    out["is_holiday"] = df["market_date"].isin(hol)
    out["month"] = df["month"]
    out["season"] = season(df["month"])

    # ---- fundamentals (D-1 09:45 forecast) --------------------------------
    out["fc_load"] = df["forecast_load"]
    out["fc_load_regime"] = load_regime(df["forecast_load"], load_thresholds)
    peak = df.groupby("market_date")["forecast_load"].max()
    out["fc_daily_peak"] = _by_date(df, peak)
    out["fc_load_vs_peak"] = out["fc_load"] / out["fc_daily_peak"]
    # rows are in UTC order, so the previous row is the previous hour; for HE1 that is HE24 of D-1,
    # whose forecast was published on D-2
    out["fc_ramp_1h"] = df["forecast_load"].diff()

    fc = _hour_matrix(df, "forecast_load")
    act = _hour_matrix(df, "actual_load")
    out["fc_change_vs_prev_day"] = out["fc_load"] - _lookup(df, fc.shift(1))
    full_peak = peak.reindex(fc.index)
    out["fc_peak_change_vs_prev_day"] = out["fc_daily_peak"] - _by_date(df, full_peak.shift(1))
    out["fc_vs_actual_d2"] = out["fc_load"] - _lookup(df, act.shift(2))

    # ---- recent forecast bias (complete days up to D-2) -------------------
    err = df["actual_load"] - df["forecast_load"]
    err_pct = err / df["forecast_load"] * 100
    tmp = df[["market_date", "hour_ending"]].assign(err=err, err_pct=err_pct)
    out["fc_err_d2_same_hour"] = _lookup(df, _hour_matrix(tmp, "err").shift(2))
    e_daily = _daily(tmp, "err")
    out["fc_err_d2_daily_mean"] = _by_date(df, (e_daily["sum"] / e_daily["n"]).shift(2))
    out["fc_err_7d_mean"] = _by_date(df, _trailing_mean_std(e_daily, 7, 2)[0])
    out["fc_err_pct_7d_mean"] = _by_date(df, _trailing_mean_std(_daily(tmp, "err_pct"), 7, 2)[0])

    # ---- market history ---------------------------------------------------
    da = _hour_matrix(df, "da_lmp")
    out["da_lmp_d1_same_hour"] = _lookup(df, da.shift(1))
    da_daily = _daily(df, "da_lmp")
    out["da_lmp_d1_daily_mean"] = _by_date(df, (da_daily["sum"] / da_daily["n"]).shift(1))

    out["rt_lmp_d2_same_hour"] = _lookup(df, _hour_matrix(df, "rt_lmp").shift(2))
    sp = _hour_matrix(df, "da_rt_spread")
    out["spread_d2_same_hour"] = _lookup(df, sp.shift(2))
    sp_daily = _daily(df, "da_rt_spread")
    out["spread_d2_daily_mean"] = _by_date(df, (sp_daily["sum"] / sp_daily["n"]).shift(2))

    clipped = df[["market_date", "hour_ending"]].assign(s=df["da_rt_spread"].clip(-SPREAD_CLIP, SPREAD_CLIP))
    out["spread_same_hour_7d_mean"] = _lookup(df, _hour_matrix(clipped, "s").shift(2).rolling(7, min_periods=5).mean())
    c_daily = _daily(clipped, "s")
    out["spread_7d_mean"] = _by_date(df, _trailing_mean_std(c_daily, 7, 2)[0])
    out["spread_30d_mean"] = _by_date(df, _trailing_mean_std(c_daily, 30, 2)[0])
    out["spread_vol_7d"] = _by_date(df, _trailing_mean_std(sp_daily, 7, 2)[1])

    # rolling median / share over pooled hours: compute per day on the trailing window of hourly values
    day_index = sp.index
    med, share = {}, {}
    vals = df.groupby("market_date")["da_rt_spread"].apply(lambda s: s.to_numpy()).reindex(day_index)
    for i, day in enumerate(day_index):
        window = vals.iloc[max(0, i - 8): i - 1] if i >= 2 else []   # days D-8 .. D-2
        arrs = [a for a in window if isinstance(a, np.ndarray)]
        if len(arrs) >= 6:
            pooled = np.concatenate(arrs)
            med[day], share[day] = np.nanmedian(pooled), np.nanmean(pooled > 0)
    out["spread_7d_median"] = df["market_date"].map(pd.Series(med)).to_numpy()
    out["share_rt_above_da_7d"] = df["market_date"].map(pd.Series(share)).to_numpy()

    rt_daily = _daily(df, "rt_lmp")
    out["rt_vol_7d"] = _by_date(df, _trailing_mean_std(rt_daily, 7, 2)[1])
    out["rt_vol_30d"] = _by_date(df, _trailing_mean_std(rt_daily, 30, 2)[1])

    # ---- targets and ex-post columns --------------------------------------
    out["da_rt_spread"] = df["da_rt_spread"]
    out["target_rt_above_da"] = (df["da_rt_spread"] > 0).astype(int).where(df["da_rt_spread"].notna())
    out["da_lmp"] = df["da_lmp"]
    out["rt_lmp"] = df["rt_lmp"]
    out["actual_load"] = df["actual_load"]
    out["load_forecast_error"] = err
    out["load_regime_actual"] = load_regime(df["actual_load"], load_thresholds)   # same cut points, ex-post

    missing = set(out.columns) - set(KEY_COLUMNS) - set(FEATURES)
    if missing:
        raise RuntimeError(f"unregistered feature columns: {sorted(missing)}")
    return out


def run(cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    from src.config import load_config

    cfg = cfg or load_config()
    hourly = pd.read_parquet(cfg["paths"]["processed"] / "hourly.parquet")     # holdout is NOT read here
    thresholds = fit_load_thresholds(hourly.loc[hourly["period"] == "train", "forecast_load"])
    feats = build_features(hourly, thresholds)
    path = Path(cfg["paths"]["processed"]) / "features.parquet"
    feats.to_parquet(path, index=False)
    log.info("saved %s: %d rows, %d tradable features", path.name, len(feats), len(tradable_features()))
    return feats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    f = run()
    model_rows = f[f["period"] != "lookback"]
    print("\nmissing share per tradable feature (excluding lookback):")
    print(model_rows[tradable_features()].isna().mean().sort_values(ascending=False).round(4).to_string())
