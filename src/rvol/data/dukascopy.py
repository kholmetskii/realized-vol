"""
Download Dukascopy tick data.

Format: https://datafeed.dukascopy.com/datafeed/{SYMBOL}/{YYYY}/{MM}/{DD}/{HH}h_ticks.bi5
  - MM is zero-indexed: January = 00, December = 11. DD and HH are ordinary,
    zero-padded.
  - The body is LZMA-compressed (the .bi5 extension tells you nothing).
  - Decompressed, it is a stream of 20-byte records, big-endian:
        uint32   milliseconds since the start of the hour
        uint32   ask, integer (divide by the point divider)
        uint32   bid, integer
        float32  ask volume
        float32  bid volume
  - An empty hour (weekend, holiday) is a zero-length body or a 404.
  - Timestamps are UTC.

The server throttles and answers 503 under load, so downloads retry with
exponential backoff, a whole run is never lost to one failing hour, and the
cache lets an interrupted run continue where it stopped.

Usage:
    python -m rvol.data.dukascopy EURUSD 2024-01-15 2024-01-19 --out data/
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import http.client
import lzma
import pathlib
import random
import time
import urllib.error
import urllib.request
from collections.abc import Callable

import numpy as np
import pandas as pd

BASE = "https://datafeed.dukascopy.com/datafeed"

#: Statuses worth another attempt: throttling and transient server faults.
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

EMPTY_COLUMNS = ["ts", "bid", "ask", "bid_vol", "ask_vol"]


class FetchError(RuntimeError):
    """One hour could not be downloaded, after every attempt."""


def point_divider(symbol: str) -> float:
    """Integer price divider. JPY pairs and most CFDs quote to 3 decimals,
    the rest of FX to 5."""
    return 1e3 if "JPY" in symbol.upper() else 1e5


RECORD = np.dtype([
    ("ms", ">u4"),
    ("ask", ">u4"),
    ("bid", ">u4"),
    ("ask_vol", ">f4"),
    ("bid_vol", ">f4"),
])


def hour_url(symbol: str, when: dt.datetime) -> str:
    return (
        f"{BASE}/{symbol.upper()}/{when.year:04d}/{when.month - 1:02d}/"
        f"{when.day:02d}/{when.hour:02d}h_ticks.bi5"
    )


def fetch_hour(
    symbol: str,
    when: dt.datetime,
    retries: int = 6,
    base_delay: float = 1.0,
    timeout: float = 60.0,
    sleep: Callable[[float], None] = time.sleep,
) -> bytes | None:
    """Fetch one hourly file.

    Returns the raw (still compressed) body, or None when the server says the
    hour does not exist — a weekend, a holiday, or before history starts.

    Throttling (429) and transient server faults (5xx) are retried with
    exponential backoff plus jitter. Anything still failing after `retries`
    attempts raises FetchError, so the caller can record the hour and carry on.
    """
    url = hour_url(symbol, when)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    last: Exception | None = None

    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body: bytes = resp.read()
                return body
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            last = e
            if e.code not in RETRYABLE_STATUS:
                raise FetchError(f"{url}: HTTP {e.code} {e.reason}") from e
        except (urllib.error.URLError, TimeoutError, http.client.HTTPException) as e:
            last = e

        if attempt < retries - 1:
            # 1, 2, 4, 8, 16 ... seconds, jittered so parallel workers do not
            # retry in lockstep and trip the rate limit again.
            sleep(base_delay * 2**attempt * (1.0 + 0.25 * random.random()))

    raise FetchError(f"{url}: giving up after {retries} attempts ({last})")


def decode_hour(
    raw: bytes | None, hour_start: dt.datetime, divisor: float
) -> pd.DataFrame:
    """Decompress and decode one hourly body into a DataFrame."""
    if not raw:
        return pd.DataFrame(columns=EMPTY_COLUMNS)

    body = lzma.decompress(raw)
    if len(body) == 0:
        return pd.DataFrame(columns=EMPTY_COLUMNS)
    if len(body) % RECORD.itemsize:
        raise ValueError(
            f"{hour_start}: body length {len(body)} is not a multiple of {RECORD.itemsize}"
        )

    arr = np.frombuffer(body, dtype=RECORD)
    ts = pd.Timestamp(hour_start, tz="UTC") + pd.to_timedelta(
        arr["ms"].astype("int64"), unit="ms"
    )
    return pd.DataFrame({
        "ts": ts,
        "bid": arr["bid"].astype("float64") / divisor,
        "ask": arr["ask"].astype("float64") / divisor,
        "bid_vol": arr["bid_vol"].astype("float64"),
        "ask_vol": arr["ask_vol"].astype("float64"),
    })


def cache_paths(cache: pathlib.Path, symbol: str, when: dt.datetime) -> tuple[
    pathlib.Path, pathlib.Path
]:
    """Where an hour's body and its "no data here" marker live."""
    body = cache / symbol.upper() / f"{when:%Y/%m/%d/%H}h_ticks.bi5"
    return body, body.with_suffix(".empty")


def read_cache(
    cache: pathlib.Path, symbol: str, when: dt.datetime
) -> tuple[bool, bytes | None]:
    """Look one hour up in the cache.

    Returns (hit, body). A marker file means the hour is known to hold no data;
    a failed download leaves nothing behind, so it is retried next run.
    """
    body_path, marker = cache_paths(cache, symbol, when)
    if marker.exists():
        return True, None
    if body_path.exists() and body_path.stat().st_size > 0:
        return True, body_path.read_bytes()
    return False, None


def write_cache(
    cache: pathlib.Path, symbol: str, when: dt.datetime, raw: bytes | None
) -> None:
    body_path, marker = cache_paths(cache, symbol, when)
    body_path.parent.mkdir(parents=True, exist_ok=True)
    if raw:
        body_path.write_bytes(raw)
    else:
        marker.touch()


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
    cache: pathlib.Path | None = None,
    max_workers: int = 6,
    retries: int = 6,
    progress: bool = True,
) -> tuple[pd.DataFrame, list[dt.datetime]]:
    """Fetch every hour in [start, end] inclusive.

    Downloads run in a small thread pool, because the server throttles each
    connection far below the available bandwidth. Hours that fail every attempt
    are collected and returned rather than raised, so one bad patch does not
    discard the rest of the run; re-running picks them up from the cache.

    Returns (ticks, failed_hours).
    """
    divisor = point_divider(symbol)
    hours = trading_hours(start, end)
    frames: dict[dt.datetime, pd.DataFrame] = {}
    failed: list[dt.datetime] = []

    def one(when: dt.datetime) -> tuple[dt.datetime, bytes | None]:
        if cache is not None:
            hit, raw = read_cache(cache, symbol, when)
            if hit:
                return when, raw
        raw = fetch_hour(symbol, when, retries=retries)
        if cache is not None:
            write_cache(cache, symbol, when, raw)
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


def sanity_check(ticks: pd.DataFrame) -> None:
    """Minimal checks. Accepting data silently is a bad habit."""
    if not ticks["ts"].is_monotonic_increasing:
        raise ValueError("ticks are not sorted")
    crossed = (ticks["ask"] < ticks["bid"]).sum()
    print(f"ticks: {len(ticks):,}")
    print(f"span: {ticks['ts'].iloc[0]} .. {ticks['ts'].iloc[-1]}")
    print(f"crossed quotes (ask < bid): {crossed}")
    spread = (ticks["ask"] - ticks["bid"]) / ticks["mid"] * 1e4
    print(f"spread, bps: median {spread.median():.2f}, "
          f"99th pct {spread.quantile(0.99):.2f}")
    print(f"non-positive prices: {(ticks['mid'] <= 0).sum()}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("symbol", help="e.g. EURUSD")
    p.add_argument("start", help="YYYY-MM-DD")
    p.add_argument("end", help="YYYY-MM-DD")
    p.add_argument("--out", default="data", help="output directory")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--workers", type=int, default=6,
                   help="parallel downloads (the server throttles each one)")
    p.add_argument("--retries", type=int, default=6, help="attempts per hour")
    args = p.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = None if args.no_cache else out / "raw"

    ticks, failed = fetch_range(
        args.symbol,
        dt.date.fromisoformat(args.start),
        dt.date.fromisoformat(args.end),
        cache=cache,
        max_workers=args.workers,
        retries=args.retries,
    )

    if ticks.empty:
        raise SystemExit("no ticks downloaded")
    sanity_check(ticks)

    if failed:
        print(f"\nfailed hours: {len(failed)} (re-run to retry them)")
        for when in failed[:10]:
            print(f"  {when}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")

    dest = out / f"{args.symbol.upper()}_{args.start}_{args.end}.parquet"
    ticks.to_parquet(dest, index=False)
    print(f"written: {dest}")


if __name__ == "__main__":
    main()
