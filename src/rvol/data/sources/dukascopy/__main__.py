"""Command line: python -m rvol.data.sources.dukascopy SYMBOL START END."""

from __future__ import annotations

import argparse
import datetime as dt
import pathlib

import pandas as pd

from rvol.data.sources.dukascopy.download import fetch_range
from rvol.data.sources.dukascopy.dukascopy_client import DukascopyClient
from rvol.data.sources.dukascopy.raw_hour_cache import RawHourCache


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
    p = argparse.ArgumentParser(prog="python -m rvol.data.sources.dukascopy")
    p.add_argument("symbol", help="e.g. EURUSD")
    p.add_argument("start", help="YYYY-MM-DD")
    p.add_argument("end", help="YYYY-MM-DD")
    p.add_argument("--out", help="tick output directory (default: data/ticks/SYMBOL)")
    p.add_argument(
        "--cache", default="data/raw/dukascopy", help="raw download cache (default: %(default)s)"
    )
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--workers", type=int, default=6,
                   help="parallel downloads (the server throttles each one)")
    p.add_argument("--retries", type=int, default=6, help="attempts per hour")
    args = p.parse_args()

    out = pathlib.Path(args.out or f"data/ticks/{args.symbol.upper()}")
    out.mkdir(parents=True, exist_ok=True)
    cache = None if args.no_cache else RawHourCache(args.cache)

    ticks, failed = fetch_range(
        args.symbol,
        dt.date.fromisoformat(args.start),
        dt.date.fromisoformat(args.end),
        client=DukascopyClient(retries=args.retries),
        cache=cache,
        max_workers=args.workers,
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
