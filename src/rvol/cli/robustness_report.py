"""Run predefined robustness checks for the standard forecast experiment."""

from __future__ import annotations

import argparse

from rvol.cli._arguments import add_forecast_arguments, experiment_run_from_args
from rvol.composition import ROBUSTNESS_CHECK_NAMES, standard_diagnostic_checks
from rvol.infrastructure import ParquetDatasetRepository
from rvol.reporting import RobustnessArtifactWriter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("daily", help="daily realized-variance Parquet dataset")
    add_forecast_arguments(parser)
    parser.add_argument("--max-hac-lag", type=int, default=5)
    parser.add_argument("--top-errors", type=int, default=5)
    parser.add_argument(
        "--check", action="append", choices=ROBUSTNESS_CHECK_NAMES,
        help="diagnostic to include; may be repeated (default: all checks)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/final_evaluation/robustness",
    )
    args = parser.parse_args()
    if args.max_hac_lag < 0:
        parser.error("--max-hac-lag must be non-negative")
    if args.top_errors < 1:
        parser.error("--top-errors must be positive")
    run = experiment_run_from_args(parser, args)

    definition = run.definition
    try:
        checks = standard_diagnostic_checks(
            args.check, hac_lags=tuple(range(args.max_hac_lag + 1)), top_errors=args.top_errors,
        )
    except ValueError as error:
        parser.error(str(error))
    analyzer = definition.robustness_analyzer(checks=checks)
    daily = ParquetDatasetRepository(args.daily).load()
    experiment = run.run(daily)
    if not experiment.records:
        raise SystemExit("no forecasts: add more data or lower --min-train-size")

    robustness = analyzer.analyze(experiment)
    paths = RobustnessArtifactWriter(args.output_dir).write(robustness)

    target_dates = {record.target_date for record in experiment.records}
    print(f"targets:          {len(target_dates)}")
    for check in robustness.checks:
        print(f"{check.name}: {len(check.rows)} rows")
    print("artifacts:")
    for path in paths.all():
        print(f"  {path}")
