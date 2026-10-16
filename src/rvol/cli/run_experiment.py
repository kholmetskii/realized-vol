"""Run one forecast experiment and render its selected outputs."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

from rvol.cli._arguments import (
    add_forecast_arguments,
    experiment_from_args,
    forecast_dataset_snapshot,
)
from rvol.domain import ExperimentResult
from rvol.forecasting.composition import Experiment
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


def main(experiment: Experiment | None = None, argv: Sequence[str] | None = None) -> None:
    """Run a Python-defined setup with paths, model selections and output options."""
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
    args = parser.parse_args(argv)
    run = experiment_from_args(parser, args, experiment)

    daily = ParquetDatasetRepository(args.daily).load()
    dataset = forecast_dataset_snapshot(args.daily, daily, run)
    result = run.run(daily)
    if not result.records:
        raise SystemExit("no forecasts: check data, reporting dates and Python training settings")
    evaluation = run.evaluator().evaluate(result)
    robustness = (
        run.robustness_analyzer().analyze(
            result, evaluation=evaluation,
        ) if args.robustness else None
    )
    configuration = run.resolved_configuration
    output_dir = Path(args.output_dir)
    paths = list(ForecastArtifactWriter(output_dir).write(
        result, evaluation, config=run.config, dataset=dataset,
        specification=run.specification, configuration=configuration,
    ).all())
    if args.plot:
        paths.extend(_write_plot(run, result, evaluation, output_dir, dataset, configuration))
    if robustness is not None:
        paths.extend(RobustnessArtifactWriter(output_dir / "robustness").write(
            robustness, configuration=configuration,
        ).all())

    _print_evaluation(result, evaluation)
    if args.plot:
        print(f"\nplot metric: {run.plot_metric}")
    if robustness is not None:
        print("\nrobustness checks:")
        for check in robustness.checks:
            print(f"  {check.name}: {len(check.rows)} rows")
    print("\nartifacts:")
    for path in paths:
        print(f"  {path}")
