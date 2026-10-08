"""Argument parsers shared by command-line entry points."""

from __future__ import annotations

import argparse
import datetime as dt

from rvol.composition import (
    ExperimentRun,
    holdout_experiment_definition,
    standard_experiment_definition,
)
from rvol.domain import (
    EveryNSessions,
    ExpandingWindow,
    ExperimentConfig,
    HoldoutSettings,
    RollingWindow,
    RunIdentity,
    WalkForwardSettings,
)


def iso_date(value: str) -> dt.date:
    """Parse an ISO calendar date for argparse."""
    try:
        return dt.date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected date in YYYY-MM-DD format") from error


def positive_integer(value: str) -> int:
    """Parse a strictly positive observation count or session interval."""
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a positive integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def add_forecast_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the shared training, date and execution options."""
    parser.add_argument(
        "--min-train-size", type=positive_integer,
        help="minimum eligible training observations (default: selected executor's minimum)",
    )
    parser.add_argument("--forecast-start", type=iso_date, help="first target date to evaluate")
    parser.add_argument("--forecast-end", type=iso_date, help="last target date to evaluate")
    parser.add_argument("--run-id", default="default", help="identity of this experiment run")
    parser.add_argument("--fold-id", help="optional validation fold identity")
    parser.add_argument(
        "--executor", choices=("walk-forward", "holdout"), default="walk-forward",
        help="experiment execution method (default: %(default)s)",
    )
    parser.add_argument(
        "--training-window", choices=("expanding", "rolling"),
        help="walk-forward training history (default: expanding)",
    )
    parser.add_argument(
        "--window-size", type=positive_integer,
        help="maximum training observations; required only for a rolling window",
    )
    parser.add_argument(
        "--retrain-every", type=positive_integer,
        help=(
            "refit every N eligible forecast sessions, anchored to the full dataset "
            "(walk-forward only; default: 1)"
        ),
        metavar="N",
    )
    parser.add_argument(
        "--train-end", type=iso_date,
        help="holdout training cutoff; required for --executor holdout",
    )
    parser.add_argument(
        "--update-observations", action="store_true",
        help="holdout only: update stateful predictors with newly observed outcomes",
    )


def experiment_run_from_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace,
) -> ExperimentRun[WalkForwardSettings] | ExperimentRun[HoldoutSettings]:
    """Bind executor and typed settings; reject incompatible options before I/O."""
    try:
        identity = RunIdentity(args.run_id, args.fold_id)
        if args.executor == "holdout":
            for option in ("training_window", "window_size", "retrain_every"):
                if getattr(args, option) is not None:
                    parser.error(f"--{option.replace('_', '-')} requires --executor walk-forward")
            if args.train_end is None:
                parser.error("--train-end is required when --executor is holdout")
            holdout = HoldoutSettings(
                train_end=args.train_end,
                min_train_size=(
                    HoldoutSettings(args.train_end).min_train_size
                    if args.min_train_size is None else args.min_train_size
                ),
                update_observations=args.update_observations,
            )
            return ExperimentRun(
                definition=holdout_experiment_definition(),
                config=ExperimentConfig(
                    execution=holdout, forecast_start=args.forecast_start,
                    forecast_end=args.forecast_end,
                    identity=identity,
                ),
            )
        if args.train_end is not None:
            parser.error("--train-end requires --executor holdout")
        if args.update_observations:
            parser.error("--update-observations requires --executor holdout")
        if args.training_window == "rolling" and args.window_size is None:
            parser.error("--window-size is required when --training-window is rolling")
        if args.training_window != "rolling" and args.window_size is not None:
            parser.error("--window-size requires --training-window rolling")
        window = (
            RollingWindow(args.window_size)
            if args.training_window == "rolling" else ExpandingWindow()
        )
        walk_forward = WalkForwardSettings(
            training_window=window,
            retrain=(EveryNSessions() if args.retrain_every is None
                     else EveryNSessions(args.retrain_every)),
            min_train_size=(
                WalkForwardSettings().min_train_size
                if args.min_train_size is None else args.min_train_size
            ),
        )
        return ExperimentRun(
            definition=standard_experiment_definition(),
            config=ExperimentConfig(
                execution=walk_forward, forecast_start=args.forecast_start,
                forecast_end=args.forecast_end,
                identity=identity,
            ),
        )
    except ValueError as error:
        parser.error(str(error))
