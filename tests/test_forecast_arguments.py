import argparse
from dataclasses import replace
from datetime import date

import pytest

from rvol.cli._arguments import add_forecast_arguments, experiment_from_args
from rvol.domain import (
    EveryNSessions,
    ExperimentConfig,
    HoldoutSettings,
    RollingWindow,
    RunIdentity,
    TargetHorizon,
    WalkForwardSettings,
)
from rvol.forecasting.composition import Experiment, standard_experiment
from rvol.forecasting.evaluation import (
    LargestErrorsCheck,
    LogMSEMetric,
    MonthlyLossCheck,
)
from rvol.forecasting.models import EWMAModel


def parse(arguments: list[str], experiment: Experiment | None = None) -> Experiment:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_forecast_arguments(parser)
    return experiment_from_args(parser, parser.parse_args(arguments), experiment)


def test_default_arguments_preserve_the_standard_experiment():
    actual = parse([])
    expected = standard_experiment()

    assert actual.config == ExperimentConfig(execution=WalkForwardSettings())
    assert actual.resolved_configuration == expected.resolved_configuration


@pytest.mark.parametrize("settings", [
    WalkForwardSettings(RollingWindow(25), EveryNSessions(5), min_train_size=20),
    HoldoutSettings(date(2024, 12, 31), 20, False),
    HoldoutSettings(date(2024, 12, 31), 20, True),
])
def test_arguments_preserve_the_python_execution_settings_and_components(settings):
    experiment = standard_experiment(
        ExperimentConfig(execution=settings), target_horizon=TargetHorizon(5, "cumulative"),
    )
    actual = parse([], experiment)

    assert actual.config.execution is settings
    assert actual.models == experiment.models
    assert all(a is b for a, b in zip(actual.models, experiment.models, strict=True))
    assert actual.feature_builder is experiment.feature_builder
    assert actual.target_builder is experiment.target_builder
    assert actual.resolved_configuration == experiment.resolved_configuration


def test_metadata_and_report_dates_override_only_the_supplied_run_configuration():
    original = standard_experiment(ExperimentConfig(
        execution=HoldoutSettings(date(2024, 6, 30), 20, True),
        identity=RunIdentity("saved", "old"), instrument="GBP/USD",
    ))
    actual = parse([
        "--instrument", "EUR/USD", "--run-id", "default", "--fold-id", "2024",
        "--forecast-start", "2024-07-01", "--forecast-end", "2024-12-31",
    ], original)

    assert actual.config == replace(
        original.config, instrument="EUR/USD", identity=RunIdentity("default", "2024"),
        forecast_start=date(2024, 7, 1), forecast_end=date(2024, 12, 31),
    )
    assert actual.config.execution is original.config.execution
    assert original.config.identity == RunIdentity("saved", "old")
    assert original.config.forecast_start is None
    assert original.config.instrument == "GBP/USD"


def test_model_selection_preserves_the_supplied_model_parameters_and_requested_order():
    original = standard_experiment()
    ewma = EWMAModel(decay=0.8)
    original = replace(original, models=tuple(
        ewma if model.name == "EWMA" else model for model in original.models
    ))
    actual = parse(["--models", "HAR", "EWMA"], original)

    assert [model.name for model in actual.models] == ["HAR", "EWMA"]
    assert actual.models[1] is ewma
    assert actual.baseline_models == ("EWMA",)
    assert actual.comparison_pairs == (("EWMA", "HAR"),)
    assert [model.name for model in original.models] == [
        "historical_mean", "naive", "EWMA", "AR1", "HAR",
    ]


def test_selection_cannot_construct_models_missing_from_the_python_setup():
    original = standard_experiment()
    selected = replace(
        original, models=tuple(model for model in original.models if model.name in {"EWMA", "HAR"}),
        baseline_models=("EWMA",), comparison_pairs=(("EWMA", "HAR"),),
    )

    with pytest.raises(SystemExit) as error:
        parse(["--models", "HAR", "naive"], selected)
    assert error.value.code == 2


def test_python_metric_and_plot_choices_survive_cli_model_selection():
    original = replace(
        standard_experiment(), metrics=(LogMSEMetric(),), plot_metric="log-RV MSE",
        diagnostic_checks=(MonthlyLossCheck(), LargestErrorsCheck(2)),
    )
    actual = parse(["--models", "EWMA", "HAR"], original)

    assert actual.metrics is original.metrics
    assert actual.plot_metric == "log-RV MSE"
    assert actual.diagnostic_checks == original.diagnostic_checks
    assert actual.comparison_test is original.comparison_test


@pytest.mark.parametrize("arguments, message", [
    (["--models", "missing"], "missing"),
    (["--models", "HAR", "HAR"], "unique"),
    (["--candidate", "missing"], "missing"),
    (["--models", "EWMA", "naive"], "candidate"),
    (["--models", "HAR", "naive", "--candidate", "EWMA"], "EWMA"),
    (["--baselines", "missing"], "missing"),
    (["--baselines", "EWMA", "EWMA"], "unique"),
    (["--baselines", "HAR"], "baseline"),
    (["--run-id", ""], "run_id"),
    (["--fold-id", ""], "fold_id"),
    (["--instrument", " "], "instrument"),
    (["--forecast-start", "2022-02-30"], "expected date in YYYY-MM-DD format"),
    (["--forecast-end", "2022-02-30"], "expected date in YYYY-MM-DD format"),
    (["--forecast-start", "2022-03-21", "--forecast-end", "2022-03-14"],
     "forecast_start must not be after forecast_end"),
])
def test_invalid_selections_and_metadata_produce_usage_errors(arguments, message, capsys):
    with pytest.raises(SystemExit) as error:
        parse(arguments)

    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert message in stderr
    assert "Traceback" not in stderr
