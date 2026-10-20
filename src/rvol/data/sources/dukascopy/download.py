"""Download a date range of Dukascopy ticks through the client and cache."""

from __future__ import annotations

import concurrent.futures as cf
import datetime as dt

import pandas as pd

from rvol.data.sources.dukascopy.decoding import EMPTY_COLUMNS, decode_hour, point_divider
from rvol.data.sources.dukascopy.dukascopy_client import DukascopyClient
from rvol.data.sources.dukascopy.exceptions import FetchError
from rvol.data.sources.dukascopy.raw_hour_cache import RawHourCache


def trading_hours(start: dt.date, end: dt.date) -> list[dt.datetime]:
    """Every hour in [start, end] that the FX market can be open.

    Saturday is skipped. Sunday is kept: the week opens on Sunday evening UTC,
    and the empty earlier hours cost one cheap 404 each, cached thereafter.
    """
    hours: list[dt.datetime] = []
    day = start
    while day <= end:
        if day.weekday() != 5:
            hours.extend(
                dt.datetime(day.year, day.month, day.day, h) for h in range(24)
            )
        day += dt.timedelta(days=1)
    return hours


def fetch_range(
    symbol: str,
    start: dt.date,
    end: dt.date,
    *,
    client: DukascopyClient | None = None,
    cache: RawHourCache | None = None,
    max_workers: int = 6,
    progress: bool = True,
) -> tuple[pd.DataFrame, list[dt.datetime]]:
    """Fetch every hour in [start, end] inclusive.

    Downloads run in a small thread pool, because the server throttles each
    connection far below the available bandwidth. Hours that fail every attempt
    are collected and returned rather than raised, so one bad patch does not
    discard the rest of the run; re-running picks them up from the cache.

    Returns (ticks, failed_hours).
    """
    downloader = DukascopyClient() if client is None else client
    divisor = point_divider(symbol)
    hours = trading_hours(start, end)
    frames: dict[dt.datetime, pd.DataFrame] = {}
    failed: list[dt.datetime] = []

    def one(when: dt.datetime) -> tuple[dt.datetime, bytes | None]:
        if cache is not None:
            hit, raw = cache.read(symbol, when)
            if hit:
                return when, raw
        raw = downloader.fetch_hour(symbol, when)
        if cache is not None:
            cache.write(symbol, when, raw)
        return when, raw

    done = 0
    with cf.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(one, when): when for when in hours}
        for future in cf.as_completed(futures):
            when = futures[future]
            try:
                _, raw = future.result()
            except FetchError as e:
                failed.append(when)
                if progress:
                    print(f"  ! {when}  {e}", flush=True)
            else:
                frames[when] = decode_hour(raw, when, divisor)
            done += 1
            if progress and done % 24 == 0:
                print(f"  {done}/{len(hours)} hours", flush=True)

    non_empty = [frames[when] for when in sorted(frames) if not frames[when].empty]
    if not non_empty:
        return pd.DataFrame(columns=[*EMPTY_COLUMNS, "mid"]), sorted(failed)

    ticks = pd.concat(non_empty, ignore_index=True)
    ticks = ticks[ticks["ts"].notna()].sort_values("ts").reset_index(drop=True)
    ticks["mid"] = 0.5 * (ticks["bid"] + ticks["ask"])
    return ticks, sorted(failed)
