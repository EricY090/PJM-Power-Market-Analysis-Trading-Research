"""Tests for feature engineering, centred on the look-ahead guarantee.

The look-ahead test scrambles every value that was *not yet published* at the decision time
for day D (10:30 EPT on D-1) and checks that no tradable feature for day D changes:

* RT LMP and actual load: we only allow complete days up to D-2 -> scramble days >= D-1
* DA LMP: day X is published on X-1           -> scramble days >= D
* Load forecast: day X's forecast is made X-1 -> scramble days >= D+1
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.clean import add_market_time
from src.data.features import (
    FEATURES, KEY_COLUMNS, build_features, ex_post_columns, nerc_holidays, tradable_features,
)

EPT = "America/New_York"
THRESHOLDS = np.array([90.0, 110.0, 130.0])


def synthetic_hourly(start="2024-09-01", end="2024-12-15", seed=0) -> pd.DataFrame:
    """Random hourly frame with the cleaned-table schema, spanning the Nov 3 2024 fall-back."""
    s = pd.Timestamp(start).tz_localize(EPT).tz_convert("UTC")
    e = (pd.Timestamp(end) + pd.Timedelta(days=1)).tz_localize(EPT).tz_convert("UTC") - pd.Timedelta(hours=1)
    df = add_market_time(pd.DataFrame({"timestamp_utc": pd.date_range(s, e, freq="h")}))
    rng = np.random.default_rng(seed)
    n = len(df)
    df["forecast_load"] = rng.uniform(80, 140, n)
    df["actual_load"] = df["forecast_load"] + rng.normal(0, 3, n)
    df["da_lmp"] = rng.uniform(20, 60, n)
    df["rt_lmp"] = df["da_lmp"] + rng.normal(0, 10, n)
    return finalize(df)


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["da_rt_spread"] = df["rt_lmp"] - df["da_lmp"]
    df["load_forecast_error"] = df["actual_load"] - df["forecast_load"]
    df["day_of_week"] = df["market_date"].dt.day_of_week
    df["month"] = df["market_date"].dt.month
    df["period"] = "train"
    return df


def scramble_unpublished(df: pd.DataFrame, day: pd.Timestamp, seed: int = 99) -> pd.DataFrame:
    """Replace everything not yet published at 10:30 EPT on day-1 with new random numbers."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    rt_mask = out["market_date"] >= day - pd.Timedelta(days=1)
    da_mask = out["market_date"] >= day
    fc_mask = out["market_date"] >= day + pd.Timedelta(days=1)
    out.loc[rt_mask, "rt_lmp"] = rng.uniform(-500, 3000, rt_mask.sum())
    out.loc[rt_mask, "actual_load"] = rng.uniform(50, 200, rt_mask.sum())
    out.loc[da_mask, "da_lmp"] = rng.uniform(-50, 1000, da_mask.sum())
    out.loc[fc_mask, "forecast_load"] = rng.uniform(50, 200, fc_mask.sum())
    return finalize(out)


@pytest.fixture(scope="module")
def base():
    df = synthetic_hourly()
    return df, build_features(df, THRESHOLDS)


def tradable_rows(feats: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    return feats.loc[feats["market_date"] == day, tradable_features()].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Look-ahead
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("day", ["2024-10-15", "2024-11-03", "2024-11-04", "2024-11-05", "2024-12-10"])
def test_no_lookahead_tradable_features_ignore_unpublished_data(base, day):
    df, feats = base
    day = pd.Timestamp(day)
    rebuilt = build_features(scramble_unpublished(df, day), THRESHOLDS)
    before, after = tradable_rows(feats, day), tradable_rows(rebuilt, day)
    assert before.notna().any().all(), "every tradable feature should be defined on this day"
    pd.testing.assert_frame_equal(before, after, check_dtype=False)


def test_lookahead_test_is_not_vacuous(base):
    """Changing *published* data (day D-2 RT, day D-1 DA) must change the features."""
    df, feats = base
    day = pd.Timestamp("2024-10-15")
    changed = df.copy()
    d2 = changed["market_date"] == day - pd.Timedelta(days=2)
    d1 = changed["market_date"] == day - pd.Timedelta(days=1)
    changed.loc[d2, "rt_lmp"] += 50
    changed.loc[d1, "da_lmp"] += 50
    rebuilt = build_features(finalize(changed), THRESHOLDS)
    before, after = tradable_rows(feats, day), tradable_rows(rebuilt, day)
    for col in ("spread_d2_same_hour", "rt_lmp_d2_same_hour", "da_lmp_d1_same_hour", "spread_7d_mean"):
        assert not np.allclose(before[col], after[col]), col


def test_ex_post_columns_do_change_with_same_day_data(base):
    df, feats = base
    day = pd.Timestamp("2024-10-15")
    rebuilt = build_features(scramble_unpublished(df, day), THRESHOLDS)
    a = feats.loc[feats["market_date"] == day, "da_rt_spread"].to_numpy()
    b = rebuilt.loc[rebuilt["market_date"] == day, "da_rt_spread"].to_numpy()
    assert not np.allclose(a, b)


# --------------------------------------------------------------------------- #
# Alignment and definitions
# --------------------------------------------------------------------------- #
def test_same_hour_lags_align_by_clock_hour(base):
    df, feats = base
    day, d2, d1 = pd.Timestamp("2024-10-15"), pd.Timestamp("2024-10-13"), pd.Timestamp("2024-10-14")
    f = feats.loc[feats["market_date"] == day].set_index("hour_ending")
    src = df.set_index(["market_date", "hour_ending"])
    for h in (1, 8, 18, 24):
        assert f.loc[h, "spread_d2_same_hour"] == pytest.approx(src.loc[(d2, h), "da_rt_spread"])
        assert f.loc[h, "da_lmp_d1_same_hour"] == pytest.approx(src.loc[(d1, h), "da_lmp"])
        assert f.loc[h, "fc_change_vs_prev_day"] == pytest.approx(
            src.loc[(day, h), "forecast_load"] - src.loc[(d1, h), "forecast_load"])


def test_fall_back_day_lag_uses_mean_of_repeated_hour(base):
    """Two days after the 25-hour day, HE2's D-2 lag is the average of the two HE2 hours."""
    df, feats = base
    fb = df[df["market_date"] == pd.Timestamp("2024-11-03")]
    he2 = fb.loc[fb["hour_ending"] == 2, "da_rt_spread"]
    assert len(he2) == 2
    f = feats[(feats["market_date"] == pd.Timestamp("2024-11-05")) & (feats["hour_ending"] == 2)]
    assert f["spread_d2_same_hour"].iloc[0] == pytest.approx(he2.mean())


def test_spread_trailing_mean_is_clipped():
    df = synthetic_hourly(start="2024-09-01", end="2024-10-10")
    spike_day = pd.Timestamp("2024-09-20")
    df.loc[df["market_date"] == spike_day, "rt_lmp"] += 5000
    feats = build_features(finalize(df), THRESHOLDS)
    later = feats.loc[feats["market_date"] == spike_day + pd.Timedelta(days=3), "spread_7d_mean"].iloc[0]
    assert later < 100 / 7 + 10        # one clipped day of +$100 cannot push a 7-day mean far


def test_nerc_holidays():
    hol = set(nerc_holidays("2022-12-01", "2025-12-31").date)
    for d in ("2022-12-26", "2023-01-02", "2023-07-04", "2024-05-27", "2024-09-02", "2024-11-28", "2025-12-25"):
        assert pd.Timestamp(d).date() in hol, d
    assert pd.Timestamp("2023-01-01").date() not in hol     # Sunday holiday observed Monday


def test_registry_covers_every_column_and_separates_targets(base):
    _, feats = base
    assert set(feats.columns) == set(KEY_COLUMNS) | set(FEATURES)   # hour_ending is both key and feature
    trad, post = set(tradable_features()), set(ex_post_columns())
    assert not trad & post
    for col in ("da_lmp", "rt_lmp", "actual_load", "load_forecast_error", "da_rt_spread", "target_rt_above_da"):
        assert col in post and col not in trad
