import argparse
from datetime import date

import pytest

from rvol.application import HoldoutExperiment, WalkForwardExperiment
from rvol.cli._arguments import add_forecast_arguments, experiment_run_from_args
from rvol.domain import (
    EveryNSessions,
    ExperimentConfig,
    HoldoutSettings,
    RollingWindow,
    WalkForwardSettings,
)


def parse_config(arguments: list[str]) -> ExperimentConfig[WalkForwardSettings | HoldoutSettings]:
    parser = argparse.ArgumentParser()
    add_forecast_arguments(parser)
    return experiment_run_from_args(parser, parser.parse_args(arguments)).config


def test_default_forecast_arguments_preserve_the_domain_defaults():
    assert parse_config([]) == ExperimentConfig(execution=WalkForwardSettings())
    assert parse_config(["--training-window", "expanding"]) == ExperimentConfig(
        execution=WalkForwardSettings(),
    )


@pytest.mark.parametrize("update", [False, True])
def test_holdout_arguments_select_compatible_executor_and_settings(update):
    parser = argparse.ArgumentParser()
    add_forecast_arguments(parser)
    args = parser.parse_args([
        "--executor", "holdout", "--train-end", "2024-12-31", "--min-train-size", "20",
        *(["--update-observations"] if update else []),
    ])
    run = experiment_run_from_args(parser, args)

    assert run.definition.executor_factory is HoldoutExperiment
    assert run.config == ExperimentConfig(execution=HoldoutSettings(date(2024, 12, 31), 20, update))


def test_each_executor_uses_its_own_default_settings():
    assert parse_config(["--executor", "holdout", "--train-end", "2024-12-31"]) == ExperimentConfig(
        execution=HoldoutSettings(date(2024, 12, 31)),
    )
    parser = argparse.ArgumentParser()
    add_forecast_arguments(parser)
    run = experiment_run_from_args(parser, parser.parse_args(["--executor", "walk-forward"]))
    assert run.definition.executor_factory is WalkForwardExperiment
    assert run.config == ExperimentConfig(execution=WalkForwardSettings())


def test_expanding_history_can_refit_less_frequently():
    config = parse_config(["--min-train-size", "20", "--retrain-every", "5"])

    assert config == ExperimentConfig(
        execution=WalkForwardSettings(retrain=EveryNSessions(5), min_train_size=20),
    )


def test_rolling_arguments_build_the_requested_strategy_and_date_bounds():
    config = parse_config([
        "--min-train-size", "20", "--training-window", "rolling", "--window-size", "25",
        "--retrain-every", "5", "--forecast-start", "2022-03-14", "--forecast-end", "2022-03-21",
    ])

    assert config == ExperimentConfig(
        execution=WalkForwardSettings(RollingWindow(25), EveryNSessions(5), min_train_size=20),
        forecast_start=date(2022, 3, 14),
        forecast_end=date(2022, 3, 21),
    )


@pytest.mark.parametrize("option", ["--min-train-size", "--window-size", "--retrain-every"])
@pytest.mark.parametrize("value", ["0", "-1", "1.5", "abc"])
def test_observation_counts_and_refit_intervals_require_positive_integers(option, value, capsys):
    with pytest.raises(SystemExit) as error:
        parse_config([option, value])

    assert error.value.code == 2
    assert f"argument {option}: expected a positive integer" in capsys.readouterr().err


@pytest.mark.parametrize(
    "arguments, message",
    [
        (["--training-window", "unknown"], "invalid choice"),
        (["--executor", "unknown"], "invalid choice"),
        (["--executor", "holdout"], "--train-end is required"),
        (["--train-end", "2024-12-31"], "--train-end requires --executor holdout"),
        (["--update-observations"], "--update-observations requires --executor holdout"),
        (["--executor", "holdout", "--train-end", "2024-02-30"], "expected date"),
        (
            ["--executor", "holdout", "--train-end", "2024-12-31",
             "--training-window", "expanding"],
            "--training-window requires --executor walk-forward",
        ),
        (
            ["--executor", "holdout", "--train-end", "2024-12-31", "--window-size", "504"],
            "--window-size requires --executor walk-forward",
        ),
        (
            ["--executor", "holdout", "--train-end", "2024-12-31", "--retrain-every", "1"],
            "--retrain-every requires --executor walk-forward",
        ),
        (["--training-window", "rolling"], "--window-size is required"),
        (["--window-size", "504"], "--window-size requires --training-window rolling"),
        (
            ["--training-window", "expanding", "--window-size", "504"],
            "--window-size requires --training-window rolling",
        ),
        (
            ["--min-train-size", "20", "--training-window", "rolling", "--window-size", "19"],
            "training window size must be at least min_train_size",
        ),
        (["--forecast-start", "2022-02-30"], "expected date in YYYY-MM-DD format"),
        (
            ["--forecast-start", "2022-03-21", "--forecast-end", "2022-03-14"],
            "forecast_start must not be after forecast_end",
        ),
    ],
)
def test_invalid_forecast_options_produce_usage_errors(arguments, message, capsys):
    with pytest.raises(SystemExit) as error:
        parse_config(arguments)

    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert message in stderr
    assert "Traceback" not in stderr
