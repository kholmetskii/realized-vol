"""Plot out-of-sample volatility forecasts and relative model performance.

Usage:
    python scripts/forecast_plots.py \
        data/derived/EURUSD_daily_rv_5min.parquet \
        --min-train-size 200 \
        --forecast-start 2024-01-01 \
        --forecast-end 2024-03-31
"""

from __future__ import annotations

import argparse
import datetime as dt
import pathlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from rvol.application import WalkForwardExperiment  # noqa: E402
from rvol.domain import ExperimentConfig  # noqa: E402
from rvol.evaluation import ForecastEvaluator  # noqa: E402
from rvol.features.forecasting import build_har_features  # noqa: E402
from rvol.infrastructure import ParquetDatasetRepository  # noqa: E402
from rvol.models import (  # noqa: E402
    AR1Forecaster,
    HARForecaster,
    HistoricalMeanForecaster,
    NaiveForecaster,
)
from rvol.plotting.forecasting import plot_forecast_evaluation  # noqa: E402


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
    parser.add_argument("--metric", choices=("QLIKE", "log-RV MSE"), default="QLIKE")
    parser.add_argument("--out", default="figures/forecast_evaluation.png")
    args = parser.parse_args()

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

    comparisons = tuple((model, "HAR") for model in experiment.models if model != "HAR")
    evaluation = ForecastEvaluator(comparison_pairs=comparisons).evaluate(experiment)
    target_dates = sorted({record.target_date for record in experiment.records})
    figure = plot_forecast_evaluation(
        experiment,
        evaluation,
        metric=args.metric,
        title=(
            "EUR/USD out-of-sample volatility forecasts "
            f"({target_dates[0]} to {target_dates[-1]})"
        ),
    )

    output = pathlib.Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(f"targets: {len(target_dates)}")
    print(f"metric:  {args.metric}")
    print(f"figure:  {output}")


if __name__ == "__main__":
    main()
