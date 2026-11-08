"""Clean, align and validate the raw PJM feeds into one hourly table.

Timestamp policy (see README "Timestamps and DST"):

* The join key is ``timestamp_utc`` = PJM's ``datetime_beginning_utc`` (hour beginning, UTC).
  UTC has no gaps or repeats, so every hour is unique.
* Market-time columns are derived from it in America/New_York (EPT):
  ``market_date`` (EPT calendar date), ``hour_ending`` (1-24, PJM convention = local
  beginning hour + 1) and ``is_dst_repeat`` (True for the second 01:00 hour on the
  November fall-back day, which PJM labels HE02*).
* Spring-forward days therefore have 23 rows and fall-back days 25 rows.
* We cross-check PJM's own ``datetime_beginning_ept`` against our conversion.

Decision-time policy for the load forecast:

* A DA/RT spread position for operating day D must be fixed before the day-ahead market
  closes on D-1. We use 10:30 EPT on D-1 as the decision time (PJM's deadline is
  10:30 or 11:00 depending on the rule in force; 10:30 is the conservative choice).
* ``forecast_load`` is the most recent RTO forecast *evaluated at or before* that
  decision time (in practice the 09:45 EPT D-1 vintage).

Run:  python -m src.data.clean
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_config
from src.data.pjm_client import default_feeds

log = logging.getLogger(__name__)

EPT = "America/New_York"


# --------------------------------------------------------------------------- #
# Timestamp helpers
# --------------------------------------------------------------------------- #
def to_utc(values: pd.Series) -> pd.Series:
    """Parse PJM's naive UTC strings into tz-aware UTC timestamps."""
    return pd.to_datetime(values).dt.tz_localize("UTC")


def ept_naive_to_utc(values: pd.Series, tz: str = EPT) -> pd.Series:
    """Convert naive EPT wall-clock strings (no repeated hours) to tz-aware UTC."""
    return pd.to_datetime(values).dt.tz_localize(tz, nonexistent="raise", ambiguous="raise").dt.tz_convert("UTC")


def add_market_time(df: pd.DataFrame, tz: str = EPT) -> pd.DataFrame:
    """Add EPT market-time columns derived from the ``timestamp_utc`` column."""
    out = df.copy()
    local = out["timestamp_utc"].dt.tz_convert(tz)
    out["timestamp_ept"] = local
    out["market_date"] = local.dt.tz_localize(None).dt.normalize()
    out["hour_ending"] = local.dt.hour + 1
    # Second occurrence of a repeated local hour (fall-back day): same local wall time as previous hour.
    wall = local.dt.tz_localize(None)
    out["is_dst_repeat"] = wall.duplicated(keep="first")
    out["hours_in_day"] = out.groupby("market_date")["timestamp_utc"].transform("size")
    return out


def check_ept_consistency(df: pd.DataFrame, utc_col: str, ept_col: str, tz: str = EPT) -> int:
    """Count rows where PJM's EPT wall-clock disagrees with our UTC->EPT conversion."""
    ours = df[utc_col].dt.tz_convert(tz).dt.tz_localize(None)
    theirs = pd.to_datetime(df[ept_col])
    return int((ours != theirs).sum())


# --------------------------------------------------------------------------- #
# Per-feed cleaning
# --------------------------------------------------------------------------- #
def _dedupe(df: pd.DataFrame, keys: list[str], order_col: str | None = None) -> tuple[pd.DataFrame, int]:
    """Drop duplicate keys, keeping the highest ``order_col`` (e.g. version_nbr) if given."""
    if order_col is not None and order_col in df:
        df = df.sort_values(order_col)
    before = len(df)
    df = df.drop_duplicates(keys, keep="last")
    return df, before - len(df)


def clean_lmp(raw: pd.DataFrame, pnode_id: int, price_col: str, out_col: str) -> pd.DataFrame:
    """Hourly total LMP for one pricing node, indexed by ``timestamp_utc``."""
    df = raw.loc[raw["pnode_id"] == pnode_id].copy()
    if df.empty:
        raise ValueError(f"pnode_id {pnode_id} not found in raw LMP data")
    if "row_is_current" in df:
        df = df.loc[df["row_is_current"].astype(str).str.lower() == "true"]
    df["timestamp_utc"] = to_utc(df["datetime_beginning_utc"])
    bad = check_ept_consistency(df, "timestamp_utc", "datetime_beginning_ept")
    if bad:
        raise ValueError(f"{out_col}: {bad} rows where PJM EPT time != UTC converted to EPT")
    df, dropped = _dedupe(df, ["timestamp_utc"], "version_nbr")
    if dropped:
        log.warning("%s: dropped %d duplicate hours", out_col, dropped)
    return df.set_index("timestamp_utc")[[price_col]].rename(columns={price_col: out_col}).sort_index()


def clean_metered_load(raw: pd.DataFrame, area: str) -> pd.DataFrame:
    """Hourly metered load (MW) for one load area, indexed by ``timestamp_utc``."""
    df = raw.loc[raw["load_area"] == area].copy()
    df["timestamp_utc"] = to_utc(df["datetime_beginning_utc"])
    bad = check_ept_consistency(df, "timestamp_utc", "datetime_beginning_ept")
    if bad:
        raise ValueError(f"actual_load: {bad} rows where PJM EPT time != UTC converted to EPT")
    # prefer company-verified values when both exist
    df["_verified"] = df["is_verified"].astype(str).str.lower() == "true"
    df, dropped = _dedupe(df, ["timestamp_utc"], "_verified")
    if dropped:
        log.warning("actual_load: dropped %d duplicate hours", dropped)
    return df.set_index("timestamp_utc")[["mw"]].rename(columns={"mw": "actual_load"}).sort_index()


def decision_time_utc(market_date: pd.Series, close_ept: str, days_before: int = 1, tz: str = EPT) -> pd.Series:
    """UTC decision time for each operating day: ``close_ept`` local time on D - days_before."""
    hh, mm = (int(x) for x in close_ept.split(":"))
    local = market_date - pd.Timedelta(days=days_before) + pd.Timedelta(hours=hh, minutes=mm)
    return local.dt.tz_localize(tz).dt.tz_convert("UTC")


def select_forecast_vintage(raw: pd.DataFrame, area: str, close_ept: str, tz: str = EPT) -> pd.DataFrame:
    """For each forecast hour, the latest forecast evaluated at or before the decision time.

    Returns ``forecast_load`` plus the vintage used (``forecast_evaluated_utc``) and the
    forecast lead time in hours, indexed by ``timestamp_utc``.
    """
    df = raw.loc[raw["forecast_area"] == area].copy()
    df["timestamp_utc"] = to_utc(df["forecast_hour_beginning_utc"])
    df["evaluated_utc"] = to_utc(df["evaluated_at_utc"])
    bad = check_ept_consistency(df, "timestamp_utc", "forecast_hour_beginning_ept")
    if bad:
        raise ValueError(f"forecast_load: {bad} rows where PJM EPT time != UTC converted to EPT")

    market_date = df["timestamp_utc"].dt.tz_convert(tz).dt.tz_localize(None).dt.normalize()
    df["decision_utc"] = decision_time_utc(market_date, close_ept, tz=tz)
    eligible = df.loc[df["evaluated_utc"] <= df["decision_utc"]]
    chosen = eligible.sort_values("evaluated_utc").drop_duplicates("timestamp_utc", keep="last")
    out = chosen.set_index("timestamp_utc").sort_index()
    out = out.rename(columns={"forecast_load_mw": "forecast_load", "evaluated_utc": "forecast_evaluated_utc"})
    out["forecast_lead_hours"] = (out.index - out["forecast_evaluated_utc"]) / pd.Timedelta(hours=1)
    return out[["forecast_load", "forecast_evaluated_utc", "forecast_lead_hours"]]


# --------------------------------------------------------------------------- #
# Join + validation
# --------------------------------------------------------------------------- #
def build_hourly(
    da: pd.DataFrame, rt: pd.DataFrame, forecast: pd.DataFrame, load: pd.DataFrame,
    start_utc: pd.Timestamp, end_utc: pd.Timestamp, tz: str = EPT,
) -> pd.DataFrame:
    """Outer-join the cleaned feeds on a complete UTC hourly index and add derived columns."""
    index = pd.date_range(start_utc, end_utc, freq="h", tz="UTC", name="timestamp_utc")
    df = pd.concat([da, rt, forecast, load], axis=1, sort=True).reindex(index)
    df = df.reset_index()
    df["da_rt_spread"] = df["rt_lmp"] - df["da_lmp"]
    df["load_forecast_error"] = df["actual_load"] - df["forecast_load"]
    df = add_market_time(df, tz)
    df["day_of_week"] = df["market_date"].dt.day_of_week         # Monday = 0
    df["month"] = df["market_date"].dt.month
    cols = [
        "timestamp_utc", "timestamp_ept", "market_date", "hour_ending", "is_dst_repeat", "hours_in_day",
        "day_of_week", "month", "da_lmp", "rt_lmp", "da_rt_spread", "forecast_load", "actual_load",
        "load_forecast_error", "forecast_evaluated_utc", "forecast_lead_hours",
    ]
    return df[cols]


@dataclass
class ValidationReport:
    rows: int
    start_ept: str
    end_ept: str
    duplicate_timestamps: int
    missing_by_column: dict[str, int]
    dst_days: dict[str, int]                   # market_date -> hours, for days != 24 hours
    unexpected_day_lengths: dict[str, int]     # days whose length is not explained by DST
    price_range: dict[str, tuple[float, float]]
    load_range: dict[str, tuple[float, float]]
    forecast_vintages: dict[str, int]          # EPT evaluation time-of-day -> count
    warnings: list[str] = field(default_factory=list)

    def ok(self) -> bool:
        return self.duplicate_timestamps == 0 and not self.unexpected_day_lengths


def expected_hours(day: pd.Timestamp, tz: str = EPT) -> int:
    start = pd.Timestamp(day).tz_localize(tz)
    end = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize(tz)
    return int((end - start) / pd.Timedelta(hours=1))


def validate(df: pd.DataFrame, tz: str = EPT) -> ValidationReport:
    value_cols = ["da_lmp", "rt_lmp", "forecast_load", "actual_load"]
    day_len = df.groupby("market_date").size()
    full_days = day_len.iloc[1:-1] if len(day_len) > 2 else day_len   # edges may be partial
    non24 = {str(d.date()): int(n) for d, n in day_len.items() if n != 24}
    unexpected = {
        str(d.date()): int(n) for d, n in full_days.items() if n != expected_hours(d, tz)
    }
    warnings = []
    for col in ("da_lmp", "rt_lmp"):
        s = df[col].dropna()
        if (s < -500).any() or (s > 5000).any():
            warnings.append(f"{col} outside [-500, 5000] $/MWh in {int(((s < -500) | (s > 5000)).sum())} hours")
    for col in ("forecast_load", "actual_load"):
        s = df[col].dropna()
        if (s < 50_000).any() or (s > 200_000).any():
            warnings.append(f"{col} outside [50, 200] GW in {int(((s < 50_000) | (s > 200_000)).sum())} hours")
    vint = df["forecast_evaluated_utc"].dropna().dt.tz_convert(tz).dt.strftime("%H:%M").value_counts()
    rng = lambda c: (float(df[c].min()), float(df[c].max()))  # noqa: E731
    return ValidationReport(
        rows=len(df),
        start_ept=str(df["timestamp_ept"].iloc[0]),
        end_ept=str(df["timestamp_ept"].iloc[-1]),
        duplicate_timestamps=int(df["timestamp_utc"].duplicated().sum()),
        missing_by_column={c: int(df[c].isna().sum()) for c in value_cols},
        dst_days=non24,
        unexpected_day_lengths=unexpected,
        price_range={c: rng(c) for c in ("da_lmp", "rt_lmp", "da_rt_spread")},
        load_range={c: rng(c) for c in ("forecast_load", "actual_load", "load_forecast_error")},
        forecast_vintages=vint.to_dict(),
        warnings=warnings,
    )


def assign_period(market_date: pd.Series, cfg: dict[str, Any]) -> pd.Series:
    """Label each row lookback / train / validation / test / holdout from the config dates."""
    d = market_date
    p, s, h = cfg["period"], cfg["split"], cfg["final_holdout"]
    conditions = [
        d < pd.Timestamp(p["start"]),
        d <= pd.Timestamp(s["train_end"]),
        d <= pd.Timestamp(s["validation_end"]),
        d <= pd.Timestamp(p["end"]),
        (d >= pd.Timestamp(h["start"])) & (d <= pd.Timestamp(h["end"])),
    ]
    labels = ["lookback", "train", "validation", "test", "holdout"]
    return pd.Series(np.select(conditions, labels, default="gap"), index=market_date.index)


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def _read_feed(raw_dir: Path, feed: str) -> pd.DataFrame:
    files = sorted((Path(raw_dir) / feed).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no raw files for {feed} in {raw_dir}; run python -m src.data.download")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def run(cfg: dict[str, Any] | None = None) -> tuple[pd.DataFrame, ValidationReport]:
    from src.data.download import download_range

    cfg = cfg or load_config()
    tz, raw_dir = cfg["market"]["timezone"], cfg["paths"]["raw"]
    feeds = default_feeds(cfg)
    hub, area = int(cfg["market"]["pricing_node_id"]), cfg["market"]["load_area"]
    close = cfg["market"].get("decision_time_ept", "10:30")

    da = clean_lmp(_read_feed(raw_dir, feeds["da_lmp"].feed), hub, "total_lmp_da", "da_lmp")
    rt = clean_lmp(_read_feed(raw_dir, feeds["rt_lmp"].feed), hub, "total_lmp_rt", "rt_lmp")
    fc = select_forecast_vintage(_read_feed(raw_dir, feeds["load_forecast"].feed), area, close, tz)
    load = clean_metered_load(_read_feed(raw_dir, feeds["load_metered"].feed), area)

    # Full EPT operating days only: lookback start 00:00 EPT -> holdout end 23:00 EPT.
    start_utc, end_utc = download_range(cfg)
    hourly = build_hourly(da, rt, fc, load, start_utc, end_utc, tz)
    hourly["period"] = assign_period(hourly["market_date"], cfg)
    report = validate(hourly, tz)

    out = Path(cfg["paths"]["processed"])
    out.mkdir(parents=True, exist_ok=True)
    research = hourly.loc[hourly["period"] != "holdout"]
    holdout = hourly.loc[hourly["period"] == "holdout"]
    research.to_parquet(out / "hourly.parquet", index=False)
    holdout.to_parquet(out / "hourly_holdout.parquet", index=False)
    log.info("saved hourly.parquet (%d rows) and hourly_holdout.parquet (%d rows)", len(research), len(holdout))
    return hourly, report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    hourly, report = run()
    print("\n=== Validation report ===")
    for k, v in report.__dict__.items():
        print(f"{k}: {v}")
    print("periods:", hourly["period"].value_counts().to_dict())
    print("STATUS:", "OK" if report.ok() else "CHECK FAILURES ABOVE")


if __name__ == "__main__":
    main()
