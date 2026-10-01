"""Build a compact daily realized-volatility dataset from tick Parquet files.

Usage:
    python scripts/build_daily_dataset.py "data/EURUSD_*.parquet" \
        --frequency 5min \
        --out data/derived/EURUSD_daily_rv_5min.parquet
"""

from __future__ import annotations

import argparse
import glob
import pathlib

from rvol.features.dataset import build_daily_dataset
from rvol.infrastructure import ParquetDatasetRepository


def expand_inputs(patterns: list[str]) -> list[pathlib.Path]:
    """Expand shell-style patterns even when the caller quotes them."""
    paths: set[pathlib.Path] = set()
    for pattern in patterns:
        matches = [pathlib.Path(match) for match in glob.glob(pattern)]
        if matches:
            paths.update(matches)
        else:
            candidate = pathlib.Path(pattern)
            if candidate.is_file():
                paths.add(candidate)
    return sorted(paths)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="tick Parquet files or quoted glob patterns")
    parser.add_argument("--frequency", default="5min")
    parser.add_argument("--min-session-hours", type=float, default=12.0)
    parser.add_argument("--out", required=True, help="derived Parquet destination")
    args = parser.parse_args()

    output = pathlib.Path(args.out).resolve()
    inputs = [path for path in expand_inputs(args.inputs) if path.resolve() != output]
    if not inputs:
        parser.error("no input Parquet files matched")

    dataset = build_daily_dataset(
        inputs,
        frequency=args.frequency,
        min_session_hours=args.min_session_hours,
    )
    if dataset.empty:
        raise SystemExit("no complete sessions found in the input files")
    ParquetDatasetRepository(output).save(dataset)

    print(f"input files:      {len(inputs)}")
    print(f"daily sessions:   {len(dataset)}")
    print(f"first session:    {dataset['date'].iloc[0].date()}")
    print(f"last session:     {dataset['date'].iloc[-1].date()}")
    print(f"written:          {output}")


if __name__ == "__main__":
    main()
