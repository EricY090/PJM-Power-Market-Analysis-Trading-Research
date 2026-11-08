"""Tests for timestamp/DST handling, dedupe, forecast-vintage selection and the hourly join."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.clean import (
    add_market_time,
    build_hourly,
    clean_lmp,
    clean_metered_load,
    decision_time_utc,
    expected_hours,
    select_forecast_vintage,
    validate,
)

EPT = "America/New_York"
FMT = "%Y-%m-%dT%H:%M:%S"


def utc_hours(start_ept: str, end_ept: str) -> pd.DatetimeIndex:
    """All UTC hours covering EPT market days start_ept..end_ept inclusive."""
    s = pd.Timestamp(start_ept).tz_localize(EPT).tz_convert("UTC")
    e = (pd.Timestamp(end_ept) + pd.Timedelta(days=1)).tz_localize(EPT).tz_convert("UTC") - pd.Timedelta(hours=1)
    return pd.date_range(s, e, freq="h")


def raw_lmp(hours: pd.DatetimeIndex, price_col: str, pnode_id: int = 51288, price=None) -> pd.DataFrame:
    return pd.DataFrame({
        "datetime_beginning_utc": hours.tz_localize(None).strftime(FMT),
        "datetime_beginning_ept": hours.tz_convert(EPT).tz_localize(None).strftime(FMT),
        "pnode_id": pnode_id,
        price_col: price if price is not None else range(len(hours)),
        "row_is_current": True,
        "version_nbr": 1,
    })


# --------------------------------------------------------------------------- #
# DST handling
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("day,hours", [("2025-03-09", 23), ("2025-11-02", 25), ("2025-07-15", 24)])
def test_expected_hours_on_dst_days(day, hours):
    assert expected_hours(pd.Timestamp(day)) == hours


def test_market_time_on_spring_forward_day():
    hrs = utc_hours("2025-03-09", "2025-03-09")
    df = add_market_time(pd.DataFrame({"timestamp_utc": hrs}))
    assert len(df) == 23
    assert (df["hours_in_day"] == 23).all()
    assert 3 not in df["hour_ending"].tolist()        # local 02:00 does not exist -> no HE3
    assert not df["is_dst_repeat"].any()


def test_market_time_on_fall_back_day():
    hrs = utc_hours("2025-11-02", "2025-11-02")
    df = add_market_time(pd.DataFrame({"timestamp_utc": hrs}))
    assert len(df) == 25
    assert df["hour_ending"].tolist().count(2) == 2  # HE2 occurs twice
    repeat = df.loc[df["is_dst_repeat"]]
    assert len(repeat) == 1 and repeat["hour_ending"].iloc[0] == 2
    assert df["timestamp_utc"].is_unique


# --------------------------------------------------------------------------- #
# Per-feed cleaning
# --------------------------------------------------------------------------- #
def test_clean_lmp_selects_node_and_keeps_latest_version():
    hrs = utc_hours("2025-07-15", "2025-07-15")
    raw = pd.concat([raw_lmp(hrs, "total_lmp_da"), raw_lmp(hrs, "total_lmp_da", pnode_id=999)])
    dup = raw_lmp(hrs[:1], "total_lmp_da", price=[123.0])
    dup["version_nbr"] = 2
    out = clean_lmp(pd.concat([raw, dup]), 51288, "total_lmp_da", "da_lmp")
    assert len(out) == 24 and out.index.is_unique
    assert out["da_lmp"].iloc[0] == 123.0               # version 2 wins


def test_clean_lmp_rejects_inconsistent_ept_labels():
    hrs = utc_hours("2025-07-15", "2025-07-15")
    raw = raw_lmp(hrs, "total_lmp_da")
    raw.loc[0, "datetime_beginning_ept"] = "2025-07-15T05:00:00"
    with pytest.raises(ValueError, match="EPT"):
        clean_lmp(raw, 51288, "total_lmp_da", "da_lmp")


def test_clean_lmp_handles_fall_back_repeated_ept_hour():
    hrs = utc_hours("2025-11-02", "2025-11-02")
    out = clean_lmp(raw_lmp(hrs, "total_lmp_rt"), 51288, "total_lmp_rt", "rt_lmp")
    assert len(out) == 25                              # repeated 01:00 EPT hour kept, not deduped


def test_clean_metered_load_prefers_verified_row():
    hrs = utc_hours("2025-07-15", "2025-07-15")
    base = pd.DataFrame({
        "datetime_beginning_utc": hrs.tz_localize(None).strftime(FMT),
        "datetime_beginning_ept": hrs.tz_convert(EPT).tz_localize(None).strftime(FMT),
        "load_area": "RTO", "mw": 100_000.0, "is_verified": False,
    })
    verified = base.iloc[:1].assign(mw=101_000.0, is_verified=True)
    out = clean_metered_load(pd.concat([verified, base]), "RTO")
    assert len(out) == 24 and out["actual_load"].iloc[0] == 101_000.0


# --------------------------------------------------------------------------- #
# Forecast vintage (look-ahead protection)
# --------------------------------------------------------------------------- #
def test_decision_time_is_d_minus_1_close_in_utc():
    d = pd.Series(pd.to_datetime(["2025-07-15", "2025-01-15"]))
    got = decision_time_utc(d, "10:30")
    assert got.tolist() == [pd.Timestamp("2025-07-14 14:30", tz="UTC"), pd.Timestamp("2025-01-14 15:30", tz="UTC")]


def forecast_raw(day: str, vintages_ept: list[str]) -> pd.DataFrame:
    hrs = utc_hours(day, day)
    rows = []
    for k, ev in enumerate(vintages_ept):
        ev_utc = pd.Timestamp(ev).tz_localize(EPT).tz_convert("UTC")
        for h in hrs:
            rows.append({
                "evaluated_at_utc": ev_utc.tz_localize(None).strftime(FMT),
                "forecast_hour_beginning_utc": h.tz_localize(None).strftime(FMT),
                "forecast_hour_beginning_ept": h.tz_convert(EPT).tz_localize(None).strftime(FMT),
                "forecast_area": "RTO",
                "forecast_load_mw": 1000 * (k + 1),       # value identifies the vintage
            })
    return pd.DataFrame(rows)


def test_forecast_uses_latest_vintage_before_decision_time_only():
    raw = forecast_raw("2025-07-15", [
        "2025-07-14 05:45", "2025-07-14 09:45", "2025-07-14 11:45",   # 11:45 is after the 10:30 close
        "2025-07-15 05:45",                                            # same-day: clearly look-ahead
    ])
    out = select_forecast_vintage(raw, "RTO", "10:30")
    assert len(out) == 24
    assert (out["forecast_load"] == 2000).all()                         # the 09:45 D-1 vintage
    assert (out["forecast_evaluated_utc"].dt.tz_convert(EPT).dt.strftime("%H:%M") == "09:45").all()
    assert (out["forecast_lead_hours"] > 0).all()


def test_forecast_missing_if_no_vintage_before_decision_time():
    raw = forecast_raw("2025-07-15", ["2025-07-14 11:45"])
    assert select_forecast_vintage(raw, "RTO", "10:30").empty


# --------------------------------------------------------------------------- #
# Join + validation
# --------------------------------------------------------------------------- #
def test_build_hourly_joins_and_validates_across_dst():
    hrs = utc_hours("2025-11-01", "2025-11-03")        # includes 25-hour fall-back day
    da = clean_lmp(raw_lmp(hrs, "total_lmp_da", price=[30.0] * len(hrs)), 51288, "total_lmp_da", "da_lmp")
    rt = clean_lmp(raw_lmp(hrs, "total_lmp_rt", price=[35.0] * len(hrs)), 51288, "total_lmp_rt", "rt_lmp")
    fc = pd.concat([
        select_forecast_vintage(forecast_raw(d, [f"{(pd.Timestamp(d) - pd.Timedelta(days=1)).date()} 09:45"]), "RTO", "10:30")
        for d in ("2025-11-01", "2025-11-02", "2025-11-03")
    ])
    load = pd.DataFrame({"actual_load": 1500.0}, index=pd.Index(hrs, name="timestamp_utc"))
    df = build_hourly(da, rt, fc, load, hrs[0].tz_localize(None), hrs[-1].tz_localize(None))

    assert len(df) == 24 + 25 + 24
    assert (df["da_rt_spread"] == 5.0).all()
    assert (df["load_forecast_error"] == 500.0).all()
    rep = validate(df)
    assert rep.ok()
    assert rep.dst_days == {"2025-11-02": 25}
    assert all(v == 0 for v in rep.missing_by_column.values())


def test_validate_flags_missing_hour():
    hrs = utc_hours("2025-07-14", "2025-07-16")
    da = clean_lmp(raw_lmp(hrs, "total_lmp_da"), 51288, "total_lmp_da", "da_lmp").drop(hrs[30])
    rt = clean_lmp(raw_lmp(hrs, "total_lmp_rt"), 51288, "total_lmp_rt", "rt_lmp")
    empty_fc = pd.DataFrame(columns=["forecast_load", "forecast_evaluated_utc", "forecast_lead_hours"],
                            index=pd.DatetimeIndex([], tz="UTC"))
    empty_fc["forecast_evaluated_utc"] = pd.to_datetime(empty_fc["forecast_evaluated_utc"], utc=True)
    load = pd.DataFrame({"actual_load": 1.0}, index=hrs)
    df = build_hourly(da, rt, empty_fc, load, hrs[0].tz_localize(None), hrs[-1].tz_localize(None))
    rep = validate(df)
    assert rep.missing_by_column["da_lmp"] == 1
    assert rep.duplicate_timestamps == 0 and rep.ok()   # index stays complete; the gap is a NaN
