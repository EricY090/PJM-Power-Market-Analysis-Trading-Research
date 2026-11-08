"""Offline tests for the PJM client: date chunking, pagination, throttling, archive splits."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.pjm_client import (
    ArchiveSpanError,
    FeedSpec,
    PJMAPIError,
    PJMClient,
    bisect_chunk,
    format_range,
    split_at,
    year_chunks,
)

T = pd.Timestamp


# --------------------------------------------------------------------------- #
# Date helpers
# --------------------------------------------------------------------------- #
def test_year_chunks_split_on_calendar_years_and_cover_range():
    chunks = year_chunks(T("2022-12-01 05:00"), T("2024-03-01 03:00"))
    assert chunks == [
        (T("2022-12-01 05:00"), T("2022-12-31 23:00")),
        (T("2023-01-01 00:00"), T("2023-12-31 23:00")),
        (T("2024-01-01 00:00"), T("2024-03-01 03:00")),
    ]
    # contiguous: every chunk starts one hour after the previous one ends
    for (_, prev_end), (nxt_start, _) in zip(chunks, chunks[1:]):
        assert nxt_start - prev_end == pd.Timedelta(hours=1)


def test_full_leap_year_chunk_is_under_366_days():
    (start, end), = year_chunks(T("2024-01-01"), T("2024-12-31 23:00"))
    assert end - start < pd.Timedelta(days=366)


def test_year_chunks_rejects_reversed_range():
    with pytest.raises(ValueError):
        year_chunks(T("2024-01-02"), T("2024-01-01"))


def test_split_at_boundary_inside_and_outside():
    chunk = (T("2024-01-01"), T("2024-12-31 23:00"))
    assert split_at(chunk, T("2024-11-06")) == [
        (T("2024-01-01"), T("2024-11-05 23:00")),
        (T("2024-11-06"), T("2024-12-31 23:00")),
    ]
    assert split_at(chunk, T("2025-03-01")) == [chunk]
    assert split_at(chunk, T("2024-01-01")) == [chunk]   # boundary at start: nothing to split
    assert split_at(chunk, None) == [chunk]


@pytest.mark.parametrize(
    "chunk",
    [
        (T("2024-01-01"), T("2024-12-31 23:00")),
        (T("2024-03-01 05:00"), T("2024-03-02 02:00")),
        (T("2024-03-01 00:00"), T("2024-03-01 01:00")),
    ],
)
def test_bisect_chunk_produces_two_contiguous_halves(chunk):
    left, right = bisect_chunk(chunk)
    assert left[0] == chunk[0] and right[1] == chunk[1]
    assert right[0] - left[1] == pd.Timedelta(hours=1)
    assert left[0] <= left[1] and right[0] <= right[1]


def test_format_range_is_inclusive_data_miner_syntax():
    assert format_range(T("2023-07-01"), T("2023-07-01 23:00")) == "2023-07-01 00:00 to 2023-07-01 23:00"


# --------------------------------------------------------------------------- #
# Client behaviour with a fake HTTP session
# --------------------------------------------------------------------------- #
class FakeResponse:
    def __init__(self, status=200, body=None, headers=None):
        self.status_code = status
        self._body = body
        self.headers = headers or {}
        self.text = str(body)

    def json(self):
        return self._body


class FakeSession:
    """Returns queued responses (or calls a responder) and records each request."""

    def __init__(self, responder):
        self.responder = responder
        self.headers = {}
        self.calls: list[tuple[str, dict]] = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        return self.responder(url, params or {})


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def make_client(responder, rows_per_request=3, rpm=6):
    clock = FakeClock()
    session = FakeSession(responder)
    client = PJMClient(
        "test-key", max_requests_per_minute=rpm, rows_per_request=rows_per_request,
        session=session, clock=clock, sleep=clock.sleep,
    )
    return client, session, clock


def paged(total_rows):
    def responder(url, params):
        start, n = params["startRow"], params["rowCount"]
        items = [{"i": i} for i in range(start, min(start + n, total_rows + 1))]
        return FakeResponse(body={"items": items, "totalRows": total_rows})
    return responder


def test_api_key_sent_as_subscription_header():
    client, session, _ = make_client(paged(1))
    assert session.headers["Ocp-Apim-Subscription-Key"] == "test-key"


def test_fetch_paginates_until_all_rows_collected():
    client, session, _ = make_client(paged(7), rows_per_request=3)
    df = client.fetch("feed", {"x": 1})
    assert df["i"].tolist() == list(range(1, 8))
    assert [p["startRow"] for _, p in session.calls] == [1, 4, 7]
    assert all(p["x"] == 1 and p["rowCount"] == 3 for _, p in session.calls)


def test_fetch_exact_multiple_of_page_size_stops_without_extra_call():
    client, session, _ = make_client(paged(6), rows_per_request=3)
    assert len(client.fetch("feed", {})) == 6
    assert len(session.calls) == 2


def test_throttle_spaces_requests_by_rate_limit():
    client, _, clock = make_client(paged(9), rows_per_request=3, rpm=6)
    client.fetch("feed", {})
    # 3 requests -> 2 waits of 60/6 + 0.5 s
    assert clock.sleeps == pytest.approx([10.5, 10.5])


def test_retries_on_server_error_then_succeeds():
    responses = iter([FakeResponse(503), FakeResponse(body={"items": [{"a": 1}], "totalRows": 1})])
    client, session, _ = make_client(lambda u, p: next(responses))
    assert len(client.fetch("feed", {})) == 1
    assert len(session.calls) == 2


def test_client_error_raises_with_api_message():
    body = {"errors": [{"field": "Filters", "message": "A datetime is missing."}]}
    client, _, _ = make_client(lambda u, p: FakeResponse(400, body))
    with pytest.raises(PJMAPIError, match="A datetime is missing"):
        client.fetch("feed", {})


def test_fetch_range_bisects_on_archive_span_error():
    span = {"errors": [{"message": "Date range in the API request spans over archived and standard data."}]}
    cutoff = T("2024-11-06")

    def responder(url, params):
        if url.endswith("/metadata"):
            return FakeResponse(body={"enableArchiving": False})
        lo, hi = (T(s) for s in params["dt"].split(" to "))
        if lo < cutoff <= hi:
            return FakeResponse(400, span)
        return FakeResponse(body={"items": [{"lo": str(lo), "hi": str(hi)}], "totalRows": 1})

    client, _, _ = make_client(responder, rows_per_request=50_000)
    spec = FeedSpec("feed", "dt")
    df = client.fetch_range(spec, T("2024-01-01"), T("2024-12-31 23:00"))
    # pieces are contiguous, none spans the cutoff, and together they cover the year
    pieces = sorted((T(a), T(b)) for a, b in zip(df["lo"], df["hi"]))
    assert pieces[0][0] == T("2024-01-01") and pieces[-1][1] == T("2024-12-31 23:00")
    for (_, e), (s, _) in zip(pieces, pieces[1:]):
        assert s - e == pd.Timedelta(hours=1)
    assert not any(lo < cutoff <= hi for lo, hi in pieces)


def test_single_hour_at_moving_cutoff_is_retried_after_wait():
    span = {"errors": [{"message": "Date range in the API request spans over archived and standard data."}]}
    attempts = {"n": 0}

    def responder(url, params):
        if url.endswith("/metadata"):
            return FakeResponse(body={"enableArchiving": False})
        attempts["n"] += 1
        if attempts["n"] <= 2:                       # rejected twice, then the cutoff has moved on
            return FakeResponse(400, span)
        return FakeResponse(body={"items": [{"ok": 1}], "totalRows": 1})

    client, _, clock = make_client(responder, rows_per_request=50_000)
    df = client.fetch_range(FeedSpec("feed", "dt"), T("2024-11-05 03:00"), T("2024-11-05 03:00"), span_retry_wait=300)
    assert len(df) == 1
    assert clock.sleeps.count(300) == 2


def test_single_hour_span_error_gives_up_after_max_retries():
    span = {"errors": [{"message": "spans over archived and standard data"}]}

    def responder(url, params):
        if url.endswith("/metadata"):
            return FakeResponse(body={"enableArchiving": False})
        return FakeResponse(400, span)

    client, _, _ = make_client(responder, rows_per_request=50_000)
    with pytest.raises(ArchiveSpanError):
        client.fetch_range(FeedSpec("feed", "dt"), T("2024-11-05 03:00"), T("2024-11-05 03:00"),
                           span_retry_wait=1, span_max_retries=2)


def test_archive_span_error_is_subclass_for_callers():
    assert issubclass(ArchiveSpanError, PJMAPIError)
