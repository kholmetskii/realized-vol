"""Small command-line selections applied to an experiment defined in Python."""

from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import replace

import pandas as pd

from rvol.forecasting.composition import Experiment, standard_experiment
from rvol.forecasting.evaluation import (
    LagSensitivityCheck,
    LargestErrorsCheck,
    MonthlyLossCheck,
    WinRateCheck,
)
from rvol.presentation.reporting import DatasetSnapshot


def iso_date(value: str) -> dt.date:
    """Parse an ISO calendar date for argparse."""
    try:
        return dt.date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected date in YYYY-MM-DD format") from error


def add_forecast_arguments(parser: argparse.ArgumentParser) -> None:
    """Register operational choices; detailed research settings live in Python."""
    parser.add_argument("--instrument", help="instrument label, for example EUR/USD")
    parser.add_argument("--models", nargs="+", help="select models configured in the Python setup")
    parser.add_argument("--candidate", help="candidate model; defaults to a sole selected model")
    parser.add_argument(
        "--baselines", nargs="*", help="baseline models; an empty list disables them",
    )
    parser.add_argument("--forecast-start", type=iso_date, help="first target date to evaluate")
    parser.add_argument("--forecast-end", type=iso_date, help="last target date to evaluate")
    parser.add_argument("--run-id", help="identity of this experiment run")
    parser.add_argument("--fold-id", help="optional validation fold identity")


def experiment_from_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    experiment: Experiment | None = None,
) -> Experiment:
    """Select configured components and apply explicit reporting overrides."""
    try:
        base = standard_experiment() if experiment is None else experiment
        models = base.models
        if args.models is not None:
            if len(args.models) != len(set(args.models)):
                raise ValueError("selected model names must be unique")
            configured = {model.name: model for model in base.models}
            missing = set(args.models).difference(configured)
            if missing:
                raise ValueError(f"unknown models in Python setup: {', '.join(sorted(missing))}")
            models = tuple(configured[name] for name in args.models)
        names = {model.name for model in models}
        candidate = args.candidate if args.candidate is not None else (
            models[0].name if len(models) == 1 else base.candidate_model
        )
        if candidate not in names:
            raise ValueError(
                f"candidate {candidate!r} must name a selected model; choose it with --candidate"
            )

        if args.baselines is not None:
            baselines = tuple(args.baselines)
        elif candidate != base.candidate_model:
            order = dict.fromkeys((
                *base.baseline_models, base.candidate_model, *(model.name for model in base.models),
            ))
            baselines = tuple(name for name in order if name in names and name != candidate)
        else:
            baselines = tuple(name for name in base.baseline_models if name in names)

        pairs = base.comparison_pairs
        default_pairs = ((baselines[0], candidate),) if baselines else ()
        if pairs:
            if args.candidate is not None or args.baselines is not None:
                pairs = default_pairs
            elif args.models is not None:
                pairs = tuple(pair for pair in pairs if all(name in names for name in pair))
                pairs = pairs or default_pairs

        checks = base.diagnostic_checks
        if not baselines:
            checks = tuple(
                check for check in checks
                if not isinstance(check, (LagSensitivityCheck, WinRateCheck))
            ) or (MonthlyLossCheck(), LargestErrorsCheck())

        config = base.config
        identity = replace(
            config.identity,
            run_id=config.identity.run_id if args.run_id is None else args.run_id,
            fold_id=config.identity.fold_id if args.fold_id is None else args.fold_id,
        )
        return replace(
            base,
            config=replace(
                config, identity=identity,
                instrument=config.instrument if args.instrument is None else args.instrument,
                forecast_start=(
                    config.forecast_start if args.forecast_start is None else args.forecast_start
                ),
                forecast_end=(
                    config.forecast_end if args.forecast_end is None else args.forecast_end
                ),
            ),
            models=models, candidate_model=candidate, baseline_models=baselines,
            comparison_pairs=pairs, diagnostic_checks=checks,
        )
    except ValueError as error:
        parser.error(str(error))


def forecast_dataset_snapshot(
    source: str, daily: pd.DataFrame, run: Experiment,
) -> DatasetSnapshot:
    """Fingerprint the input using the configured target's date column."""
    date_column = dict(run.target_builder.specification.parameters).get("date_col", "date")
    if not isinstance(date_column, str) or not date_column.strip():
        raise ValueError("target date_col specification must be a non-empty string")
    return DatasetSnapshot.from_frame(source, daily, date_column=date_column)
