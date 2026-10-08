"""Create the standard out-of-sample forecast evaluation plot."""

from __future__ import annotations

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from rvol.cli._arguments import iso_date  # noqa: E402
from rvol.composition import standard_experiment_definition  # noqa: E402
from rvol.domain import ExperimentConfig  # noqa: E402
from rvol.infrastructure import ParquetDatasetRepository  # noqa: E402
from rvol.plotting.forecasting import plot_forecast_evaluation  # noqa: E402


def main() -> None:
    definition = standard_experiment_definition()
    parser = argparse.ArgumentParser()
    parser.add_argument("daily", help="daily realized-variance Parquet dataset")
    parser.add_argument("--min-train-size", type=int, default=ExperimentConfig().min_train_size)
    parser.add_argument("--forecast-start", type=iso_date)
    parser.add_argument("--forecast-end", type=iso_date)
    parser.add_argument(
        "--metric",
        choices=tuple(metric.name for metric in definition.metrics),
        default=definition.metrics[0].name,
    )
    parser.add_argument("--out", default="figures/forecast_evaluation.png")
    args = parser.parse_args()

    daily = ParquetDatasetRepository(args.daily).load()
    experiment = definition.run(
        daily,
        ExperimentConfig(
            min_train_size=args.min_train_size,
            forecast_start=args.forecast_start,
            forecast_end=args.forecast_end,
        ),
    )
    if not experiment.records:
        raise SystemExit("no forecasts: add more data or lower --min-train-size")

    evaluation = definition.evaluator(
        comparison_pairs=definition.all_comparison_pairs,
    ).evaluate(experiment)
    target_dates = sorted({record.target_date for record in experiment.records})
    figure = plot_forecast_evaluation(
        experiment,
        evaluation,
        candidate_model=definition.candidate_model,
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
