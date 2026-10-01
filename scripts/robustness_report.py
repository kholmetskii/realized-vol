"""Run predefined robustness checks for a fixed forecast experiment.

Usage:
    python scripts/robustness_report.py \
        data/derived/EURUSD_daily_rv_5min.parquet \
        --min-train-size 200 \
        --forecast-start 2024-01-01 \
        --forecast-end 2024-03-31 \
        --output-dir outputs/final_evaluation/robustness
"""

from __future__ import annotations

import argparse
import datetime as dt

from rvol.application import WalkForwardExperiment
from rvol.domain import ExperimentConfig
from rvol.evaluation import ForecastRobustnessAnalyzer
from rvol.features.forecasting import build_har_features
from rvol.infrastructure import ParquetDatasetRepository
from rvol.models import (
    AR1Forecaster,
    HARForecaster,
    HistoricalMeanForecaster,
    NaiveForecaster,
)
from rvol.reporting import RobustnessArtifactWriter


def iso_date(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected date in YYYY-MM-DD format") from error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("daily", help="daily realized-variance Parquet dataset")
    parser.add_argument("--min-train-size", type=int, default=252)
    parser.add_argument("--forecast-start", type=iso_date)
    parser.add_argument("--forecast-end", type=iso_date)
    parser.add_argument("--max-hac-lag", type=int, default=5)
    parser.add_argument("--top-errors", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        default="outputs/final_evaluation/robustness",
    )
    args = parser.parse_args()
    if args.max_hac_lag < 0:
        parser.error("--max-hac-lag must be non-negative")
    if args.top_errors < 1:
        parser.error("--top-errors must be positive")

    daily = ParquetDatasetRepository(args.daily).load()
    features = build_har_features(daily)
    experiment = WalkForwardExperiment(
        models=(
            HistoricalMeanForecaster(),
            NaiveForecaster(),
            AR1Forecaster(),
            HARForecaster(),
        ),
        config=ExperimentConfig(
            min_train_size=args.min_train_size,
            forecast_start=args.forecast_start,
            forecast_end=args.forecast_end,
        ),
    ).run(features)
    if not experiment.records:
        raise SystemExit("no forecasts: add more data or lower --min-train-size")

    robustness = ForecastRobustnessAnalyzer(
        hac_lags=tuple(range(args.max_hac_lag + 1)),
        top_errors=args.top_errors,
    ).analyze(experiment)
    paths = RobustnessArtifactWriter(args.output_dir).write(robustness)

    target_dates = {record.target_date for record in experiment.records}
    print(f"targets:          {len(target_dates)}")
    print(f"HAC lag settings: {args.max_hac_lag + 1}")
    print(f"monthly rows:     {len(robustness.monthly_losses)}")
    print(f"win-rate rows:    {len(robustness.win_rates)}")
    print("artifacts:")
    for path in paths.all():
        print(f"  {path}")


if __name__ == "__main__":
    main()
