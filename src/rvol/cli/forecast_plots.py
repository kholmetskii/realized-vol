"""Create the standard out-of-sample forecast evaluation plot."""

from __future__ import annotations

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from rvol.cli._arguments import iso_date  # noqa: E402
from rvol.composition import run_standard_forecast_experiment  # noqa: E402
from rvol.domain import ExperimentConfig  # noqa: E402
from rvol.evaluation import ForecastEvaluator  # noqa: E402
from rvol.infrastructure import ParquetDatasetRepository  # noqa: E402
from rvol.plotting.forecasting import plot_forecast_evaluation  # noqa: E402


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
    experiment = run_standard_forecast_experiment(
        daily,
        ExperimentConfig(
            min_train_size=args.min_train_size,
            forecast_start=args.forecast_start,
            forecast_end=args.forecast_end,
        ),
    )
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
