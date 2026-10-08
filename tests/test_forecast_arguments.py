import argparse
from datetime import date

import pytest

from rvol.cli._arguments import add_forecast_arguments, experiment_config_from_args
from rvol.domain import (
    EveryNSessions,
    ExperimentConfig,
    RollingWindow,
    WalkForwardStrategy,
)


def parse_config(arguments: list[str]) -> ExperimentConfig:
    parser = argparse.ArgumentParser()
    add_forecast_arguments(parser)
    return experiment_config_from_args(parser, parser.parse_args(arguments))


def test_default_forecast_arguments_preserve_the_domain_defaults():
    assert parse_config([]) == ExperimentConfig()
    assert parse_config(["--training-window", "expanding"]) == ExperimentConfig()


def test_expanding_history_can_refit_less_frequently():
    config = parse_config(["--min-train-size", "20", "--retrain-every", "5"])

    assert config == ExperimentConfig(
        min_train_size=20, strategy=WalkForwardStrategy(retrain=EveryNSessions(5)),
    )


def test_rolling_arguments_build_the_requested_strategy_and_date_bounds():
    config = parse_config([
        "--min-train-size", "20", "--training-window", "rolling", "--window-size", "25",
        "--retrain-every", "5", "--forecast-start", "2022-03-14", "--forecast-end", "2022-03-21",
    ])

    assert config == ExperimentConfig(
        min_train_size=20, forecast_start=date(2022, 3, 14), forecast_end=date(2022, 3, 21),
        strategy=WalkForwardStrategy(RollingWindow(25), EveryNSessions(5)),
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
