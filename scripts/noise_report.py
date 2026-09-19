"""Paired noise test: does fine sampling inflate RV, day by day?

Usage:
    python scripts/noise_report.py data/EURUSD_2024-01-01_2024-03-31.parquet
"""

import argparse
import pathlib

import pandas as pd

from rvol.estimators.signature import noise_test


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("parquet", help="tick file produced by rvol.data.dukascopy")
    p.add_argument("--fine", default="1s")
    p.add_argument("--coarse", default="5min")
    args = p.parse_args()

    ticks = pd.read_parquet(args.parquet)
    print(f"{pathlib.Path(args.parquet).stem}, {args.fine} vs {args.coarse}\n")

    for col in ("mid", "bid", "ask"):
        print(f"--- {col}")
        print(noise_test(ticks, col, fine=args.fine, coarse=args.coarse))
        print()


if __name__ == "__main__":
    main()
