"""Run and report the standard out-of-sample forecast experiment."""

from __future__ import annotations

import argparse

from rvol.cli._arguments import iso_date
from rvol.composition import standard_experiment_definition
from rvol.domain import ExperimentConfig
from rvol.infrastructure import ParquetDatasetRepository
from rvol.reporting import DatasetSnapshot, ForecastArtifactWriter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("daily", help="daily realized-variance Parquet dataset")
    parser.add_argument("--min-train-size", type=int, default=ExperimentConfig().min_train_size)
    parser.add_argument("--forecast-start", type=iso_date, help="first target date to evaluate")
    parser.add_argument("--forecast-end", type=iso_date, help="last target date to evaluate")
    parser.add_argument("--hac-lags", type=int)
    parser.add_argument(
        "--output-dir",
        help="optional directory for deterministic CSV and JSON result artifacts",
    )
    parser.add_argument(
        "--compare",
        action="append",
        nargs=2,
        metavar=("BASELINE", "CANDIDATE"),
        help="one-sided comparison that CANDIDATE has lower loss; may be repeated",
    )
    args = parser.parse_args()

    definition = standard_experiment_definition()
    daily = ParquetDatasetRepository(args.daily).load()
    config = ExperimentConfig(
        min_train_size=args.min_train_size,
        forecast_start=args.forecast_start,
        forecast_end=args.forecast_end,
    )
    specification = definition.specification
    experiment_result = definition.run(daily, config)
    if not experiment_result.records:
        raise SystemExit("no forecasts: add more data or lower --min-train-size")

    evaluation = definition.evaluator(
        comparison_pairs=args.compare,
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
    if args.output_dir:
        snapshot = DatasetSnapshot.from_frame(args.daily, daily)
        artifact_paths = ForecastArtifactWriter(args.output_dir).write(
            experiment_result,
            evaluation,
            config=config,
            dataset=snapshot,
            specification=specification,
        )
        print("\nartifacts:")
        for path in artifact_paths.all():
            print(f"  {path}")
