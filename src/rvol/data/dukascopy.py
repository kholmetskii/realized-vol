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
  - An empty hour (weekend, holiday) is a zero-length body or a 404. Skip it.
  - Timestamps are UTC.

Usage:
    python dukascopy.py EURUSD 2024-01-15 2024-01-19 --out data/
"""

import argparse
import datetime as dt
import lzma
import pathlib
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

BASE = "https://datafeed.dukascopy.com/datafeed"


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


def fetch_hour(symbol: str, when: dt.datetime, retries: int = 3) -> bytes:
    """Fetch one hourly file. Returns the raw (still compressed) body."""
    url = hour_url(symbol, when)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return b""          # no such hour: weekend, or before history starts
            if attempt == retries - 1:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries - 1:
                raise
        time.sleep(2 ** attempt)
    return b""


def decode_hour(raw: bytes, hour_start: dt.datetime, divisor: float) -> pd.DataFrame:
    """Decompress and decode one hourly body into a DataFrame."""
    if not raw:
        return pd.DataFrame(columns=["ts", "bid", "ask", "bid_vol", "ask_vol"])

    body = lzma.decompress(raw)
    if len(body) == 0:
        return pd.DataFrame(columns=["ts", "bid", "ask", "bid_vol", "ask_vol"])
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


def fetch_range(symbol: str, start: dt.date, end: dt.date,
                cache: pathlib.Path | None = None,
                pause: float = 0.05) -> pd.DataFrame:
    """Fetch every hour in [start, end] inclusive. Caches raw bodies on disk."""
    divisor = point_divider(symbol)
    frames = []
    day = start
    while day <= end:
        if day.weekday() >= 5:                  # Saturday/Sunday: market closed
            day += dt.timedelta(days=1)
            continue
        for hour in range(24):
            when = dt.datetime(day.year, day.month, day.day, hour)
            raw = None
            path = None
            if cache is not None:
                path = cache / symbol.upper() / f"{when:%Y/%m/%d/%H}h_ticks.bi5"
                if path.exists():
                    raw = path.read_bytes()
            if raw is None:
                raw = fetch_hour(symbol, when)
                if path is not None:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(raw)
                time.sleep(pause)               # be polite to the server
            frames.append(decode_hour(raw, when, divisor))
        print(f"  {day}  done", flush=True)
        day += dt.timedelta(days=1)

    ticks = pd.concat(frames, ignore_index=True)
    ticks = ticks[ticks["ts"].notna()].sort_values("ts").reset_index(drop=True)
    ticks["mid"] = 0.5 * (ticks["bid"] + ticks["ask"])
    return ticks


def sanity_check(ticks: pd.DataFrame) -> None:
    """Minimal checks. Accepting data silently is a bad habit."""
    assert ticks["ts"].is_monotonic_increasing, "ticks are not sorted"
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
    args = p.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = None if args.no_cache else out / "raw"

    ticks = fetch_range(
        args.symbol,
        dt.date.fromisoformat(args.start),
        dt.date.fromisoformat(args.end),
        cache=cache,
    )
    sanity_check(ticks)

    dest = out / f"{args.symbol.upper()}_{args.start}_{args.end}.parquet"
    ticks.to_parquet(dest, index=False)
    print(f"written: {dest}")


if __name__ == "__main__":
    main()
