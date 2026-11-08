"""Client for the PJM Data Miner 2 API.

Handles authentication, pagination (50,000-row pages), request throttling for the
non-member limit (6 requests/minute), retries, and the Data Miner archive rules:

* Hourly DA/RT LMPs older than ``archiveCutoffDays`` (731) are "historic". Historic
  queries must stay inside one UTC calendar year, cannot sort, and can only filter on
  dates, ``type``, ``row_is_current`` and ``version_nbr`` -- not ``pnode_id``.
* A single query may not span the archive cutoff.
* Standard queries must cover < 366 days.

We therefore request one UTC calendar year at a time, split at the archive cutoff,
and filter LMPs by ``type=HUB`` (all hubs), selecting Western Hub later in cleaning.
Every completed chunk is cached as Parquet under ``data/raw/<feed>/`` so nothing is
downloaded twice.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import requests

log = logging.getLogger(__name__)

SPAN_ERROR_TEXT = "spans over archived and standard"


class PJMAPIError(RuntimeError):
    """Non-retryable error returned by the Data Miner API."""

    def __init__(self, status: int, message: str):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.message = message


class ArchiveSpanError(PJMAPIError):
    """The requested date range crosses the archive cutoff and must be split."""


# --------------------------------------------------------------------------- #
# Feed definitions
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FeedSpec:
    """How to download one Data Miner feed."""

    feed: str                       # API feed name, e.g. "da_hrl_lmps"
    date_field: str                 # UTC datetime field used for the range filter
    filters: dict[str, str] = field(default_factory=dict)
    tag: str = ""                   # short label used in cache file names

    def cache_name(self, start: pd.Timestamp, end: pd.Timestamp) -> str:
        tag = f"_{self.tag}" if self.tag else ""
        return f"{self.feed}{tag}_{start:%Y%m%dT%H}_{end:%Y%m%dT%H}.parquet"


def default_feeds(cfg: dict[str, Any]) -> dict[str, FeedSpec]:
    """The four MVP feeds, keyed by the short names used throughout the project."""
    area = cfg["market"]["load_area"]
    lmp_filters = {"type": "HUB", "row_is_current": "TRUE"}
    return {
        "da_lmp": FeedSpec("da_hrl_lmps", "datetime_beginning_utc", lmp_filters, "hubs"),
        "rt_lmp": FeedSpec("rt_hrl_lmps", "datetime_beginning_utc", lmp_filters, "hubs"),
        "load_forecast": FeedSpec(
            "load_frcstd_hist", "forecast_hour_beginning_utc", {"forecast_area": area}, area.lower()
        ),
        "load_metered": FeedSpec(
            "hrl_load_metered", "datetime_beginning_utc", {"load_area": area}, area.lower()
        ),
    }


# --------------------------------------------------------------------------- #
# Date-range helpers (pure functions, unit-tested)
# --------------------------------------------------------------------------- #
def year_chunks(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Split the inclusive hourly range [start, end] at UTC calendar-year boundaries.

    Each returned chunk is an inclusive (first_hour, last_hour) pair.
    """
    start, end = pd.Timestamp(start).floor("h"), pd.Timestamp(end).floor("h")
    if end < start:
        raise ValueError("end must not be before start")
    chunks = []
    cur = start
    while cur <= end:
        next_year = pd.Timestamp(year=cur.year + 1, month=1, day=1)
        chunk_end = min(end, next_year - pd.Timedelta(hours=1))
        chunks.append((cur, chunk_end))
        cur = chunk_end + pd.Timedelta(hours=1)
    return chunks


def split_at(
    chunk: tuple[pd.Timestamp, pd.Timestamp], boundary: pd.Timestamp | None
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Split an inclusive chunk so that ``boundary`` (an hour) starts a new piece."""
    start, end = chunk
    if boundary is None or not (start < boundary <= end):
        return [chunk]
    return [(start, boundary - pd.Timedelta(hours=1)), (boundary, end)]


def bisect_chunk(
    chunk: tuple[pd.Timestamp, pd.Timestamp],
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Split an inclusive chunk roughly in half on a day boundary."""
    start, end = chunk
    midpoint = start + (end - start) / 2
    mid = midpoint.floor("D")
    if mid <= start:  # chunk shorter than ~2 days: split on the hour instead
        mid = midpoint.ceil("h")
    return split_at(chunk, mid)


def format_range(start: pd.Timestamp, end: pd.Timestamp) -> str:
    """Data Miner range syntax, inclusive on both ends."""
    return f"{start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M}"


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #
class PJMClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.pjm.com/api/v1",
        max_requests_per_minute: int = 6,
        rows_per_request: int = 50_000,
        timeout: tuple[float, float] = (15, 300),
        max_retries: int = 5,
        session: requests.Session | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.base_url = base_url.rstrip("/")
        self.rows_per_request = rows_per_request
        self.timeout = timeout
        self.max_retries = max_retries
        # 0.5 s safety margin on top of the published limit
        self.min_interval = 60.0 / max_requests_per_minute + 0.5
        self.session = session or requests.Session()
        self.session.headers.update({"Ocp-Apim-Subscription-Key": api_key})
        self._clock, self._sleep = clock, sleep
        self._last_request: float | None = None
        self._metadata: dict[str, dict] = {}
        self.request_count = 0

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "PJMClient":
        from src.config import get_api_key

        p = cfg["pjm"]
        return cls(
            api_key=get_api_key(cfg),
            base_url=p["base_url"],
            max_requests_per_minute=p["max_requests_per_minute"],
            rows_per_request=p["rows_per_request"],
        )

    # -- low level ---------------------------------------------------------- #
    def _throttle(self) -> None:
        if self._last_request is not None:
            wait = self.min_interval - (self._clock() - self._last_request)
            if wait > 0:
                self._sleep(wait)
        self._last_request = self._clock()

    def _get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}/{path}"
        for attempt in range(self.max_retries + 1):
            self._throttle()
            self.request_count += 1
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout) as exc:
                if attempt == self.max_retries:
                    raise
                backoff = 15 * 2**attempt
                log.warning("%s on %s; retrying in %ss", type(exc).__name__, path, backoff)
                self._sleep(backoff)
                continue

            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt == self.max_retries:
                    break
                backoff = float(resp.headers.get("Retry-After") or 15 * 2**attempt)
                log.warning("HTTP %s on %s; retrying in %.0fs", resp.status_code, path, backoff)
                self._sleep(backoff)
                continue
            self._raise_api_error(resp)
        raise PJMAPIError(resp.status_code, f"gave up after {self.max_retries} retries")

    @staticmethod
    def _raise_api_error(resp: requests.Response) -> None:
        try:
            body = resp.json()
            msgs = [e.get("message", "") for e in body.get("errors", [])] or [str(body)]
            message = " | ".join(msgs)
        except ValueError:
            message = resp.text[:500]
        if SPAN_ERROR_TEXT in message:
            raise ArchiveSpanError(resp.status_code, message)
        raise PJMAPIError(resp.status_code, message)

    # -- public API ---------------------------------------------------------- #
    def metadata(self, feed: str) -> dict:
        if feed not in self._metadata:
            self._metadata[feed] = self._get_json(f"{feed}/metadata")
        return self._metadata[feed]

    def archive_cutoff(self, feed: str, now: pd.Timestamp | None = None) -> pd.Timestamp | None:
        """Approximate UTC start of 'standard' (non-archived) data, or None if not archived."""
        meta = self.metadata(feed)
        if not meta.get("enableArchiving"):
            return None
        now = pd.Timestamp.now(tz="UTC").tz_localize(None) if now is None else now
        return (now - pd.Timedelta(days=int(meta["archiveCutoffDays"]))).floor("h")

    def fetch(self, feed: str, params: dict[str, Any]) -> pd.DataFrame:
        """Fetch every page of a query and return the rows as a DataFrame."""
        rows: list[dict] = []
        start_row = 1
        while True:
            page = self._get_json(
                feed, {**params, "rowCount": self.rows_per_request, "startRow": start_row}
            )
            items = page.get("items", []) if isinstance(page, dict) else page
            total = page.get("totalRows") if isinstance(page, dict) else None
            rows.extend(items)
            start_row += self.rows_per_request
            if len(items) < self.rows_per_request or (total is not None and len(rows) >= total):
                break
        if total is not None and len(rows) != total:
            raise PJMAPIError(200, f"{feed}: expected {total} rows, received {len(rows)}")
        return pd.DataFrame(rows)

    def fetch_range(
        self,
        spec: FeedSpec,
        start: pd.Timestamp,
        end: pd.Timestamp,
        span_retry_wait: float = 300.0,
        span_max_retries: int = 12,
    ) -> pd.DataFrame:
        """Fetch an inclusive UTC hour range, splitting at the archive cutoff as needed.

        PJM's archive cutoff moves with the clock, and for a few hours around it the API
        rejects even single-hour requests as "spanning" archived and standard data. Ranges
        are bisected down to one hour; a single hour that is still rejected is retried after
        ``span_retry_wait`` seconds (the cutoff moves past it within about an hour).
        """
        pending = split_at((start, end), self.archive_cutoff(spec.feed))
        frames = []
        span_retries = 0
        while pending:
            chunk = pending.pop(0)
            params = {**spec.filters, spec.date_field: format_range(*chunk)}
            try:
                frames.append(self.fetch(spec.feed, params))
            except ArchiveSpanError:
                if chunk[1] > chunk[0]:
                    log.info("%s %s spans the archive cutoff; splitting", spec.feed, chunk)
                    pending[:0] = bisect_chunk(chunk)
                    continue
                if span_retries >= span_max_retries:
                    raise
                span_retries += 1
                log.info("%s hour %s is at the moving archive cutoff; retry %d/%d in %.0fs",
                         spec.feed, chunk[0], span_retries, span_max_retries, span_retry_wait)
                self._sleep(span_retry_wait)
                pending.insert(0, chunk)
        frames = [f for f in frames if not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --------------------------------------------------------------------------- #
# Cached download
# --------------------------------------------------------------------------- #
def download_feed(
    client: PJMClient,
    spec: FeedSpec,
    start: pd.Timestamp,
    end: pd.Timestamp,
    raw_dir: Path,
    refresh: bool = False,
) -> list[Path]:
    """Download [start, end] (UTC, inclusive hours) one calendar year at a time.

    Each year is written to ``raw_dir/<feed>/<cache_name>`` and skipped on later runs.
    Returns the list of cache files covering the range.
    """
    out_dir = Path(raw_dir) / spec.feed
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for chunk in year_chunks(start, end):
        path = out_dir / spec.cache_name(*chunk)
        paths.append(path)
        if path.exists() and not refresh:
            log.info("cached: %s", path.name)
            continue
        t0 = time.time()
        df = client.fetch_range(spec, *chunk)
        df.to_parquet(path, index=False)
        log.info("saved %s: %d rows in %.0fs", path.name, len(df), time.time() - t0)
    return paths


def load_raw(paths: list[Path]) -> pd.DataFrame:
    """Concatenate cached raw Parquet files."""
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
