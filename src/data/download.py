"""Download the raw PJM feeds used by the project into data/raw/.

Usage (from the project root):
    python -m src.data.download                      # all four feeds
    python -m src.data.download da_lmp rt_lmp        # selected feeds
    python -m src.data.download --refresh load_metered

The range covers the research window, the 2026 final holdout, and a lookback buffer
before the start so that rolling features are defined from the first training day.
Downloaded files are cached per UTC calendar year; re-running only fetches what is missing.
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from src.config import load_config
from src.data.pjm_client import PJMClient, default_feeds, download_feed


def download_range(cfg: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Inclusive UTC hour range to download (naive UTC timestamps)."""
    tz = cfg["market"]["timezone"]
    lookback = pd.Timedelta(days=cfg["period"].get("lookback_days", 31))
    start_ept = pd.Timestamp(cfg["period"]["start"]).tz_localize(tz) - lookback
    end_ept = pd.Timestamp(cfg["final_holdout"]["end"]).tz_localize(tz) + pd.Timedelta(hours=23)
    to_utc = lambda ts: ts.tz_convert("UTC").tz_localize(None)  # noqa: E731
    return to_utc(start_ept), to_utc(end_ept)


def main(argv: list[str] | None = None) -> None:
    cfg = load_config()
    feeds = default_feeds(cfg)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("feeds", nargs="*", help=f"feeds to download (default: all of {', '.join(feeds)})")
    parser.add_argument("--refresh", action="store_true", help="re-download even if cached")
    args = parser.parse_args(argv)
    unknown = set(args.feeds) - set(feeds)
    if unknown:
        parser.error(f"unknown feed(s): {', '.join(sorted(unknown))}; choose from {', '.join(feeds)}")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    start, end = download_range(cfg)
    logging.info("download range (UTC): %s -> %s", start, end)

    client = PJMClient.from_config(cfg)
    for name in args.feeds or list(feeds):
        logging.info("== %s (%s)", name, feeds[name].feed)
        download_feed(client, feeds[name], start, end, cfg["paths"]["raw"], refresh=args.refresh)
    logging.info("done: %d API requests", client.request_count)


if __name__ == "__main__":
    main()
