"""Compose and run the out-of-sample volatility forecast experiment.

Usage:
    python scripts/evaluate_forecasts.py \
        data/derived/EURUSD_daily_rv_5min.parquet \
        --forecast-start 2023-01-01 \
        --forecast-end 2023-12-31
"""

from __future__ import annotations

import argparse
import datetime as dt

import pandas as pd

from rvol.application import WalkForwardExperiment
from rvol.domain import ExperimentConfig
from rvol.evaluation import ForecastEvaluator
from rvol.features.forecasting import build_har_features
from rvol.models import (
    AR1Forecaster,
    HARForecaster,
    HistoricalMeanForecaster,
    NaiveForecaster,
)


def iso_date(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected date in YYYY-MM-DD format") from error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("daily", help="daily realized-variance Parquet dataset")
    parser.add_argument("--min-train-size", type=int, default=252)
    parser.add_argument("--forecast-start", type=iso_date, help="first target date to evaluate")
    parser.add_argument("--forecast-end", type=iso_date, help="last target date to evaluate")
    parser.add_argument("--hac-lags", type=int)
    parser.add_argument(
        "--compare",
        action="append",
        nargs=2,
        metavar=("BASELINE", "CANDIDATE"),
        help="one-sided comparison that CANDIDATE has lower loss; may be repeated",
    )
    args = parser.parse_args()

    daily = pd.read_parquet(args.daily)
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
    )
    experiment_result = experiment.run(features)
    if not experiment_result.records:
        raise SystemExit("no forecasts: add more data or lower --min-train-size")

    comparison_pairs = tuple(args.compare or (("naive", "HAR"),))
    evaluation = ForecastEvaluator(
        comparison_pairs=comparison_pairs,
        hac_lags=args.hac_lags,
    ).evaluate(experiment_result)
    target_dates = sorted({record.target_date for record in experiment_result.records})

    print(f"targets: {len(target_dates)}")
    print(f"period:  {target_dates[0]} .. {target_dates[-1]}\n")
    print(f"{'metric':<12} {'model':<18} {'n':>6} {'mean loss':>14}")
    for summary in evaluation.summaries:
        print(
            f"{summary.metric:<12} "
            f"{summary.model:<18} "
            f"{summary.n_obs:>6} "
            f"{summary.mean_loss:>14.6g}"
        )

    print("\npairwise alternative: candidate has lower expected loss than baseline\n")
    print(
        f"{'metric':<12} {'baseline':<16} {'candidate':<12} "
        f"{'difference':>12} {'DM':>9} {'p':>9}"
    )
    for comparison in evaluation.comparisons:
        print(
            f"{comparison.metric:<12} "
            f"{comparison.baseline_model:<16} "
            f"{comparison.candidate_model:<12} "
            f"{comparison.mean_loss_difference:>12.6g} "
            f"{comparison.dm_statistic:>9.3f} "
            f"{comparison.p_value:>9.4f}"
        )
    if evaluation.comparisons:
        print(f"\nNewey-West lags: {evaluation.comparisons[0].hac_lags}")


if __name__ == "__main__":
    main()
