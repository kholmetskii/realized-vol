"""Run one forecast experiment and render its selected outputs."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from rvol.cli._arguments import (
    add_forecast_arguments,
    experiment_from_args,
    forecast_dataset_snapshot,
    nonnegative_integer,
    positive_integer,
)
from rvol.domain import ExperimentResult
from rvol.forecasting.composition import ROBUSTNESS_CHECK_NAMES, Experiment
from rvol.forecasting.evaluation import EvaluationResult
from rvol.infrastructure import ParquetDatasetRepository
from rvol.presentation.reporting import (
    DatasetSnapshot,
    ForecastArtifactWriter,
    RobustnessArtifactWriter,
    write_forecast_plot_metadata,
)


def _print_evaluation(experiment: ExperimentResult, evaluation: EvaluationResult) -> None:
    target_dates = sorted({record.target_date for record in experiment.records})
    print(f"targets: {len(target_dates)}")
    if evaluation.instrument:
        print(f"instrument: {evaluation.instrument}")
    horizon = experiment.target_horizon
    print(f"target:  {horizon.kind}, horizon {horizon.sessions} observed sessions")
    print(f"period:  {target_dates[0]} .. {target_dates[-1]}\n")
    print(f"{'metric':<12} {'model':<18} {'n':>6} {'mean loss':>14}")
    for summary in evaluation.summaries:
        print(
            f"{summary.metric:<12} {summary.model:<18} "
            f"{summary.n_obs:>6} {summary.mean_loss:>14.6g}"
        )
    if not evaluation.comparisons:
        return
    assert evaluation.comparison_test is not None
    test_name = evaluation.comparison_test.name
    print("\npairwise alternative: candidate has lower expected loss than baseline\n")
    print(
        f"{'metric':<12} {'baseline':<16} {'candidate':<12} "
        f"{'difference':>12} {test_name:>9} {'p':>9}"
    )
    for comparison in evaluation.comparisons:
        print(
            f"{comparison.metric:<12} {comparison.baseline_model:<16} "
            f"{comparison.candidate_model:<12} {comparison.mean_loss_difference:>12.6g} "
            f"{comparison.statistic:>9.3f} {comparison.p_value:>9.4f}"
        )
    parameters = dict(evaluation.comparisons[0].test_parameters)
    print(f"\ncomparison test: {test_name}; parameters: {parameters}")


def _write_plot(
    run: Experiment,
    experiment: ExperimentResult,
    evaluation: EvaluationResult,
    output_dir: Path,
    dataset: DatasetSnapshot,
    configuration: Mapping[str, object],
) -> tuple[Path, Path]:
    # Numerical runs do not need to initialize a plotting backend or font cache.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from rvol.presentation.plotting.forecasting import plot_forecast_evaluation

    assert run.plot_metric is not None
    target_dates = sorted({record.target_date for record in experiment.records})
    figure = plot_forecast_evaluation(
        experiment, evaluation,
        candidate_model=run.candidate_model,
        baseline_models=run.baseline_models,
        metric=run.plot_metric,
        title=(
            (f"{run.config.instrument} " if run.config.instrument else "")
            + "Out-of-sample volatility forecasts "
            f"({target_dates[0]} to {target_dates[-1]})"
        ),
    )
    output = output_dir / "forecasts.png"
    try:
        figure.savefig(output, dpi=160, bbox_inches="tight")
    finally:
        plt.close(figure)
    metadata = write_forecast_plot_metadata(
        output, experiment, dataset=dataset, configuration=configuration,
    )
    return output, metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a forecast experiment once and save its results.", allow_abbrev=False,
    )
    parser.add_argument("daily", help="daily realized-variance Parquet dataset")
    add_forecast_arguments(parser)
    parser.add_argument(
        "--output-dir", default="outputs/experiment", help="directory for all experiment outputs",
    )
    parser.add_argument("--plot", action="store_true", help="also save forecasts.png and metadata")
    parser.add_argument(
        "--robustness", action="store_true", help="also save diagnostics under robustness/",
    )
    parser.add_argument(
        "--compare", action="append", nargs=2, metavar=("BASELINE", "CANDIDATE"),
        help="evaluation comparison; may be repeated and replaces saved comparisons",
    )
    parser.add_argument("--hac-lags", type=nonnegative_integer, help="explicit DM lag count")
    parser.add_argument("--metric", help="metric to plot; requires --plot")
    parser.add_argument(
        "--check", action="append", choices=ROBUSTNESS_CHECK_NAMES,
        help="diagnostic to include; may be repeated and requires --robustness",
    )
    parser.add_argument("--max-hac-lag", type=nonnegative_integer, help="requires --robustness")
    parser.add_argument("--top-errors", type=positive_integer, help="requires --robustness")
    args = parser.parse_args()
    if args.metric is not None and not args.plot:
        parser.error("--metric requires --plot")
    for option in ("check", "max_hac_lag", "top_errors"):
        if getattr(args, option) is not None and not args.robustness:
            parser.error(f"--{option.replace('_', '-')} requires --robustness")
    run = experiment_from_args(parser, args)

    daily = ParquetDatasetRepository(args.daily).load()
    experiment = run.run(daily)
    if not experiment.records:
        raise SystemExit("no forecasts: add more data or lower --min-train-size")
    evaluation = run.evaluator().evaluate(experiment)
    robustness = (
        run.robustness_analyzer().analyze(
            experiment, evaluation=evaluation,
        ) if args.robustness else None
    )
    dataset = forecast_dataset_snapshot(args.daily, daily, run)
    configuration = run.resolved_configuration
    output_dir = Path(args.output_dir)
    paths = list(ForecastArtifactWriter(output_dir).write(
        experiment, evaluation, config=run.config, dataset=dataset,
        specification=run.specification, configuration=configuration,
    ).all())
    if args.plot:
        paths.extend(_write_plot(run, experiment, evaluation, output_dir, dataset, configuration))
    if robustness is not None:
        paths.extend(RobustnessArtifactWriter(output_dir / "robustness").write(
            robustness, configuration=configuration,
        ).all())

    _print_evaluation(experiment, evaluation)
    if args.plot:
        print(f"\nplot metric: {run.plot_metric}")
    if robustness is not None:
        print("\nrobustness checks:")
        for check in robustness.checks:
            print(f"  {check.name}: {len(check.rows)} rows")
    print("\nartifacts:")
    for path in paths:
        print(f"  {path}")
