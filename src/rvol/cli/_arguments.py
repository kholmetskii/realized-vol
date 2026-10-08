"""Experiment configuration arguments and explicit command-line overrides."""

from __future__ import annotations

import argparse
import datetime as dt
from typing import Any

import pandas as pd

from rvol.composition import ExperimentRun
from rvol.composition.configuration import experiment_run_from_mapping
from rvol.domain import HoldoutSettings, WalkForwardSettings
from rvol.infrastructure.configuration import load_experiment_configuration
from rvol.reporting import DatasetSnapshot


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


def nonnegative_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a non-negative integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("expected a non-negative integer")
    return parsed


def add_forecast_arguments(parser: argparse.ArgumentParser) -> None:
    """Register shared options; None distinguishes absence from explicit overrides."""
    parser.add_argument("--config", help="saved experiment configuration (TOML)")
    parser.add_argument("--instrument", help="instrument label, for example EUR/USD")
    parser.add_argument("--models", nargs="+", help="models to run; retains saved model parameters")
    parser.add_argument("--metrics", nargs="+", help="evaluation metrics to include")
    parser.add_argument("--candidate", help="candidate model; rebuilds inherited comparisons")
    parser.add_argument(
        "--baselines", nargs="*", help="baseline models; rebuilds inherited comparisons",
    )
    parser.add_argument(
        "--min-train-size", type=positive_integer,
        help="minimum eligible training observations (default: selected executor's minimum)",
    )
    parser.add_argument("--forecast-start", type=iso_date, help="first target date to evaluate")
    parser.add_argument("--forecast-end", type=iso_date, help="last target date to evaluate")
    parser.add_argument("--run-id", help="identity of this experiment run (default: default)")
    parser.add_argument("--fold-id", help="optional validation fold identity")
    parser.add_argument(
        "--horizon", type=positive_integer,
        help="forecast horizon in observed sessions (default: 1)",
    )
    parser.add_argument(
        "--target-kind", choices=("session", "cumulative"),
        help="future-session variance or sum over the next N sessions (default: session)",
    )
    parser.add_argument(
        "--executor", choices=("walk-forward", "holdout"),
        help="experiment execution method (default: walk-forward)",
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
        "--retrain-every", type=positive_integer, metavar="N",
        help="refit every N eligible forecast sessions (walk-forward only; default: 1)",
    )
    parser.add_argument(
        "--train-end", type=iso_date,
        help="holdout training cutoff; required for --executor holdout",
    )
    parser.add_argument(
        "--update-observations", action=argparse.BooleanOptionalAction, default=None,
        help="holdout only: update stateful predictors with newly observed outcomes",
    )


def experiment_run_from_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace,
) -> ExperimentRun[WalkForwardSettings] | ExperimentRun[HoldoutSettings]:
    """Resolve defaults < TOML < explicit flags, validating everything before data I/O."""
    try:
        document = {} if args.config is None else load_experiment_configuration(args.config)

        def table(name: str) -> dict[str, Any]:
            values = document.setdefault(name, {})
            if not isinstance(values, dict):
                raise ValueError(f"{name} must be a table")
            return values

        execution = table("execution")
        if args.executor is not None and args.executor != execution.get("name", "walk-forward"):
            # Changing the selected executor discards the previous executor's private options.
            for key in ("training_window", "window_size", "retrain_every",
                        "train_end", "update_observations"):
                execution.pop(key, None)
        if args.training_window == "expanding" and execution.get("training_window") == "rolling":
            execution.pop("window_size", None)
        overrides = {
            "dataset": {"instrument": "instrument"},
            "run": {key: key for key in ("run_id", "fold_id", "forecast_start", "forecast_end")},
            "target": {"horizon": "sessions", "target_kind": "kind"},
            "execution": {
                "executor": "name", **{key: key for key in (
                    "min_train_size", "training_window", "window_size", "retrain_every",
                    "train_end", "update_observations",
                )},
            },
            "evaluation": {
                "metrics": "metrics", "candidate": "candidate", "baselines": "baselines",
                "compare": "comparison_pairs",
            },
            "robustness": {"check": "checks", "top_errors": "top_errors"},
            "plot": {"metric": "metric"},
        }
        for section, names in overrides.items():
            for argument, key in names.items():
                value = getattr(args, argument, None)
                if value is not None:
                    table(section)[key] = value
        hac_lags = getattr(args, "hac_lags", None)
        if hac_lags is not None:
            comparison = table("evaluation").setdefault("test", {})
            if not isinstance(comparison, dict):
                raise ValueError("evaluation.test must be a table")
            comparison["hac_lags"] = hac_lags
        max_lag = getattr(args, "max_hac_lag", None)
        if max_lag is not None:
            table("robustness")["hac_lags"] = list(range(max_lag + 1))
        if args.models is not None:
            saved = document.get("models", [])
            existing = {value["name"]: value for value in saved
                        if isinstance(value, dict) and isinstance(value.get("name"), str)} \
                if isinstance(saved, list) else {}
            document["models"] = [existing.get(name, {"name": name}) for name in args.models]
        if args.models is not None or args.candidate is not None or args.baselines is not None:
            _reconcile_comparisons(table("evaluation"), args)
        if args.metrics is not None and getattr(args, "metric", None) is None:
            plot = table("plot")
            if plot.get("metric") not in args.metrics:
                plot.pop("metric", None)
        return experiment_run_from_mapping(document)
    except ValueError as error:
        parser.error(str(error))


def _reconcile_comparisons(evaluation: dict[str, Any], args: argparse.Namespace) -> None:
    """Adapt inherited references without discarding explicit invalid overrides."""
    if args.candidate is not None and args.baselines is None:
        evaluation.pop("baselines", None)
    elif args.models is not None and args.baselines is None:
        baselines = evaluation.get("baselines")
        if isinstance(baselines, list) and all(isinstance(name, str) for name in baselines):
            evaluation["baselines"] = [name for name in baselines if name in args.models]

    if getattr(args, "compare", None) is not None:
        return
    pairs = evaluation.get("comparison_pairs")
    if not isinstance(pairs, list) or not pairs:
        return  # Preserve an explicit empty list; malformed lists are validated by composition.
    if args.candidate is not None or args.baselines is not None:
        evaluation.pop("comparison_pairs")
    elif args.models is not None and all(
        isinstance(pair, list) and len(pair) == 2
        and all(isinstance(name, str) for name in pair) for pair in pairs
    ):
        retained = [pair for pair in pairs if all(name in args.models for name in pair)]
        if retained:
            evaluation["comparison_pairs"] = retained
        else:
            evaluation.pop("comparison_pairs")


def forecast_dataset_snapshot(
    source: str, daily: pd.DataFrame,
    run: ExperimentRun[WalkForwardSettings] | ExperimentRun[HoldoutSettings],
) -> DatasetSnapshot:
    date_column = dict(run.definition.target_builder.specification.parameters)["date_col"]
    assert isinstance(date_column, str)
    return DatasetSnapshot.from_frame(source, daily, date_column=date_column)
