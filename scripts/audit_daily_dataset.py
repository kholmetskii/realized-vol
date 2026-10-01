"""Audit coverage and quality of a derived daily volatility dataset.

Usage:
    python scripts/audit_daily_dataset.py \
        data/derived/EURUSD_daily_rv_5min.parquet
"""

from __future__ import annotations

import argparse

import pandas as pd

from rvol.features.coverage import audit_daily_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("daily", help="daily realized-variance Parquet dataset")
    parser.add_argument("--rv-column", default="rv_5min")
    parser.add_argument("--expected-observations", type=int, default=288)
    parser.add_argument("--observation-tolerance", type=float, default=0.10)
    parser.add_argument("--max-gap-days", type=int, default=7)
    args = parser.parse_args()

    daily = pd.read_parquet(args.daily)
    audit = audit_daily_dataset(
        daily,
        rv_col=args.rv_column,
        expected_observations=args.expected_observations,
        observation_tolerance=args.observation_tolerance,
        max_gap_days=args.max_gap_days,
    )

    period = (
        "unavailable"
        if audit.start_date is None or audit.end_date is None
        else f"{audit.start_date.date()} .. {audit.end_date.date()}"
    )
    print(f"status:                  {'PASS' if audit.passed else 'FAIL'}")
    print(f"rows:                    {audit.n_rows}")
    print(f"unique sessions:         {audit.n_sessions}")
    print(f"period:                  {period}")
    print(f"median observations:     {audit.median_observation_count:.0f}")
    print(f"duplicate dates:         {audit.duplicate_dates}")
    print(f"dates out of order:      {audit.dates_out_of_order}")
    print(f"invalid dates:           {audit.invalid_dates}")
    print(f"invalid RV values:       {audit.invalid_rv}")
    print(f"invalid counts:          {audit.invalid_observation_counts}")
    print(f"unusual count sessions:  {audit.unusual_observation_sessions}")
    print(f"large gaps:              {len(audit.large_gaps)}")
    print("sessions by year:")
    for year in audit.sessions_by_year:
        print(f"  {year.year}: {year.sessions}")
    if audit.large_gaps:
        print("gap details:")
        for gap in audit.large_gaps[:10]:
            print(
                f"  {gap.previous_date.date()} .. {gap.next_date.date()} "
                f"({gap.calendar_days} days)"
            )
        if len(audit.large_gaps) > 10:
            print(f"  ... and {len(audit.large_gaps) - 10} more")

    if not audit.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
