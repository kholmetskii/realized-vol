"""Run walk-forward HAR-RV evaluation against the naïve benchmark.

Usage:
    python scripts/evaluate_forecasts.py \
        data/derived/EURUSD_daily_rv_5min.parquet \
        --forecast-start 2023-01-01
"""

from __future__ import annotations

import argparse

import pandas as pd

from rvol.evaluation.comparison import compare_forecasts
from rvol.evaluation.walk_forward import walk_forward_forecasts
from rvol.features.forecasting import build_har_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("daily", help="daily realized-variance Parquet dataset")
    parser.add_argument("--min-train-size", type=int, default=252)
    parser.add_argument("--forecast-start", help="first target date to evaluate")
    parser.add_argument("--hac-lags", type=int)
    args = parser.parse_args()

    daily = pd.read_parquet(args.daily)
    features = build_har_features(daily)
    forecasts = walk_forward_forecasts(
        features,
        min_train_size=args.min_train_size,
        forecast_start=args.forecast_start,
    )
    if forecasts.empty:
        raise SystemExit("no forecasts: add more data or lower --min-train-size")

    comparisons = compare_forecasts(forecasts, hac_lags=args.hac_lags)
    print(f"forecasts: {len(forecasts)}")
    print(f"period:    {forecasts['target_date'].iloc[0].date()} .. "
          f"{forecasts['target_date'].iloc[-1].date()}")
    print("alternative: HAR has lower expected loss than naïve\n")
    print(f"{'metric':<12} {'naive':>12} {'HAR':>12} {'difference':>12} {'DM':>9} {'p':>9}")
    for result in comparisons:
        print(
            f"{result.metric:<12} "
            f"{result.naive_mean_loss:>12.6g} "
            f"{result.har_mean_loss:>12.6g} "
            f"{result.mean_loss_difference:>12.6g} "
            f"{result.dm_statistic:>9.3f} "
            f"{result.p_value:>9.4f}"
        )
    print(f"\nNewey-West lags: {comparisons[0].hac_lags}")


if __name__ == "__main__":
    main()
