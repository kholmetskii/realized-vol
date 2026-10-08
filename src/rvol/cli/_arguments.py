"""Argument parsers shared by command-line entry points."""

from __future__ import annotations

import argparse
import datetime as dt

from rvol.domain import (
    EveryNSessions,
    ExpandingWindow,
    ExperimentConfig,
    RollingWindow,
    WalkForwardStrategy,
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
        "--min-train-size", type=positive_integer, default=ExperimentConfig().min_train_size,
        help="minimum eligible training observations before forecasting (default: %(default)s)",
    )
    parser.add_argument("--forecast-start", type=iso_date, help="first target date to evaluate")
    parser.add_argument("--forecast-end", type=iso_date, help="last target date to evaluate")
    parser.add_argument(
        "--training-window", choices=("expanding", "rolling"), default="expanding",
        help="training history used at each fit (default: %(default)s)",
    )
    parser.add_argument(
        "--window-size", type=positive_integer,
        help="maximum training observations; required only for a rolling window",
    )
    parser.add_argument(
        "--retrain-every", type=positive_integer, default=EveryNSessions().interval,
        help=(
            "refit every N eligible forecast sessions, anchored to the full dataset "
            "(default: %(default)s)"
        ),
        metavar="N",
    )


def experiment_config_from_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace,
) -> ExperimentConfig:
    """Build the domain configuration, reporting invalid combinations as usage errors."""
    if args.training_window == "rolling" and args.window_size is None:
        parser.error("--window-size is required when --training-window is rolling")
    if args.training_window == "expanding" and args.window_size is not None:
        parser.error("--window-size requires --training-window rolling")
    window = (
        RollingWindow(args.window_size) if args.training_window == "rolling" else ExpandingWindow()
    )
    try:
        return ExperimentConfig(
            min_train_size=args.min_train_size,
            forecast_start=args.forecast_start,
            forecast_end=args.forecast_end,
            strategy=WalkForwardStrategy(window, EveryNSessions(args.retrain_every)),
        )
    except ValueError as error:
        parser.error(str(error))
