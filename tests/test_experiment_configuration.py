import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rvol.cli._arguments import add_forecast_arguments, experiment_run_from_args
from rvol.domain import ExperimentConfig, HoldoutSettings, TargetHorizon, WalkForwardSettings
from rvol.forecasting.composition import (
    ExperimentRun,
    experiment_run_from_mapping,
    standard_experiment_definition,
)
from rvol.infrastructure.configuration import load_experiment_configuration


def daily_sample() -> pd.DataFrame:
    values = -10 + np.sin(np.arange(75) / 4) / 5
    return pd.DataFrame({"date": pd.bdate_range("2024-01-02", periods=75), "log_rv": values})


def parse(arguments: list[str]):
    parser = argparse.ArgumentParser()
    add_forecast_arguments(parser)
    parser.add_argument("--compare", action="append", nargs=2)
    parser.add_argument("--metric")
    return experiment_run_from_args(parser, parser.parse_args(arguments))


def test_empty_configuration_preserves_default_components_and_numerical_results():
    expected = ExperimentRun(standard_experiment_definition(), ExperimentConfig(
        execution=WalkForwardSettings(min_train_size=20),
    ))
    actual = experiment_run_from_mapping({"execution": {"min_train_size": 20}})
    assert actual.resolved_configuration == expected.resolved_configuration
    assert actual.run(daily_sample()) == expected.run(daily_sample())


def test_configuration_constructs_selected_components_with_their_actual_parameters():
    run = experiment_run_from_mapping({
        "dataset": {"instrument": "GBP/USD"},
        "target": {"sessions": 3, "kind": "cumulative"},
        "features": {"weekly_window": 2, "monthly_window": 4, "check_gaps": False},
        "models": [{"name": "naive"}, {"name": "EWMA", "decay": 0.8}, {"name": "HAR"}],
        "evaluation": {
            "metrics": ["log-RV MSE"], "candidate": "EWMA", "baselines": ["naive", "HAR"],
            "comparison_pairs": [["HAR", "EWMA"]], "test": {"name": "DM", "hac_lags": 0},
        },
        "robustness": {"checks": ["monthly_losses", "largest_errors"], "top_errors": 2},
        "execution": {"min_train_size": 20, "training_window": "rolling", "window_size": 25,
                      "retrain_every": 3},
    })
    resolved = json.loads(json.dumps(run.resolved_configuration))
    assert resolved["dataset"]["instrument"] == "GBP/USD"
    assert resolved["features"]["parameters"]["max_gap_days"] is None
    assert resolved["target"]["name"] == "realized_variance"
    assert resolved["target"]["parameters"] == {
        "date_col": "date", "rv_col": "log_rv", "horizon_sessions": 3,
        "target_kind": "cumulative",
    }
    assert "horizon_sessions" not in resolved["features"]["parameters"]
    models = {model["name"]: model for model in resolved["models"]}
    assert models["EWMA"]["parameters"] == {"decay": 0.8}
    assert models["naive"]["parameters"]["variance_multiplier"] == 3.0
    assert run.plot_metric == "log-RV MSE"
    result = run.run(daily_sample())
    assert result.config is not None and result.config.instrument == "GBP/USD"
    evaluation = run.definition.evaluator().evaluate(result)
    assert {summary.metric for summary in evaluation.summaries} == {"log-RV MSE"}
    assert {comparison.candidate_model for comparison in evaluation.comparisons} == {"EWMA"}
    assert evaluation.instrument == "GBP/USD"


@pytest.mark.parametrize("document", [
    {"unknown": {}}, {"version": 2}, {"version": True}, {"version": 1.0},
    {"features": []}, {"features": {"unknown": 1}}, {"features": {"name": "unknown"}},
    {"features": {"weekly_window": 0}}, {"features": {"monthly_window": 2}},
    {"features": {"weekly_window": True}}, {"features": {"date_col": ""}},
    {"features": {"horizon": 2}}, {"features": {"check_gaps": 1}},
    {"models": []}, {"models": ["HAR"]}, {"models": [{}]},
    {"models": [{"name": "unknown"}]}, {"models": [{"name": "HAR", "decay": 0.8}]},
    {"models": [{"name": "EWMA", "decay": True}, {"name": "HAR"}]},
    {"models": [{"name": "EWMA", "decay": "oops"}, {"name": "HAR"}]},
    {"models": [{"name": "EWMA", "decay": float("nan")}, {"name": "HAR"}]},
    {"models": [{"name": "HAR"}, {"name": "HAR"}]},
    {"evaluation": {"metrics": ["unknown"]}}, {"evaluation": {"metrics": []}},
    {"evaluation": {"metrics": ["QLIKE", "QLIKE"]}},
    {"evaluation": {"candidate": "missing"}}, {"evaluation": {"baselines": ["missing"]}},
    {"evaluation": {"baselines": ["HAR"]}},
    {"evaluation": {"comparison_pairs": [["naive", "naive"]]}},
    {"evaluation": {"comparison_pairs": [["naive", "HAR"], ["naive", "HAR"]]}},
    {"evaluation": {"comparison_pairs": [["missing", "HAR"]]}},
    {"evaluation": {"comparison_pairs": ["naive", "HAR"]}},
    {"evaluation": {"test": {"name": "unknown"}}},
    {"evaluation": {"test": {"hac_lags": -1}}},
    {"evaluation": {"test": {"alternative": "unknown"}}},
    {"execution": {"name": "unknown"}}, {"execution": {"window_size": 25}},
    {"execution": {"min_train_size": True}}, {"run": {"forecast_start": "2024-02-30"}},
    {"dataset": {"instrument": " "}}, {"dataset": {"instrument": 123}},
    {"target": {"sessions": 1.5}}, {"target": {"kind": "unknown"}},
    {"target": {"name": "unknown"}}, {"target": {"unknown": 1}},
    {"target": {"date_col": ""}}, {"target": {"rv_col": 123}},
    {"target": {"horizon": {"sessions": 3}}},
    {"target": {"date_col": "same", "rv_col": "same"}},
    {"robustness": {"checks": ["unknown"]}}, {"robustness": {"hac_lags": [True]}},
    {"robustness": {"checks": ["monthly_losses"], "top_errors": 0}},
    {"plot": {"metric": "missing"}},
])
def test_invalid_configuration_is_rejected_during_composition(document):
    with pytest.raises(ValueError):
        experiment_run_from_mapping(document)


def test_explicit_flags_override_saved_values_even_when_equal_to_defaults(tmp_path):
    source = tmp_path / "experiment.toml"
    source.write_text('''
[dataset]
instrument = "GBP/USD"
[run]
run_id = "saved"
[target]
sessions = 5
kind = "cumulative"
[execution]
name = "holdout"
train_end = 2024-06-30
min_train_size = 20
update_observations = true
''')
    run = parse([
        "--config", str(source), "--instrument", "EUR/USD", "--run-id", "default",
        "--horizon", "1", "--target-kind", "session", "--no-update-observations",
    ])
    assert run.config.instrument == "EUR/USD"
    assert run.config.identity.run_id == "default"
    assert run.config.execution == HoldoutSettings(date(2024, 6, 30), 20, False)
    parameters = dict(run.definition.target_builder.specification.parameters)
    assert parameters["horizon_sessions"] == 1
    assert parameters["target_kind"] == "session"


def test_feature_and_target_source_columns_and_settings_are_independent():
    daily = daily_sample().rename(columns={"date": "session", "log_rv": "input_variance"})
    daily["outcome_variance"] = daily["input_variance"] + 1.5
    run = experiment_run_from_mapping({
        "features": {"date_col": "session", "rv_col": "input_variance", "weekly_window": 2,
                     "monthly_window": 4},
        "target": {"name": "realized_variance", "date_col": "session",
                   "rv_col": "outcome_variance", "sessions": 3, "kind": "cumulative"},
        "execution": {"min_train_size": 20},
    })
    prepared = run.definition.prepare(daily)
    assert run.definition.target_builder.horizon == TargetHorizon(3, "cumulative")
    for index, row in enumerate(prepared.itertuples(), start=3):
        assert row.rv_daily == daily.loc[index, "input_variance"]
        expected = np.exp(daily.loc[index + 1:index + 3, "outcome_variance"]).sum()
        assert row.target == pytest.approx(np.log(expected))
        assert row.target_date == daily.loc[index + 3, "session"]

    changed = experiment_run_from_mapping({
        "features": {"date_col": "session", "rv_col": "input_variance", "weekly_window": 3,
                     "monthly_window": 10},
        "target": {"name": "realized_variance", "date_col": "session",
                   "rv_col": "outcome_variance", "sessions": 3, "kind": "cumulative"},
        "execution": {"min_train_size": 20},
    })
    assert changed.definition.specification.target == run.definition.specification.target
    pd.testing.assert_frame_equal(
        changed.definition.target_builder(daily), run.definition.target_builder(daily),
    )


@pytest.mark.parametrize("source_execution, overrides, expected", [
    ('name="holdout"\ntrain_end=2024-06-30\nupdate_observations=true',
     ["--executor", "walk-forward"], WalkForwardSettings(min_train_size=20)),
    ('name="walk-forward"\ntraining_window="rolling"\nwindow_size=25\nretrain_every=3',
     ["--executor", "holdout", "--train-end", "2024-06-30"],
     HoldoutSettings(date(2024, 6, 30), 20)),
    ('name="walk-forward"\ntraining_window="rolling"\nwindow_size=25',
     ["--training-window", "expanding"], WalkForwardSettings(min_train_size=20)),
])
def test_switching_a_strategy_discards_its_previous_private_options(
    tmp_path, source_execution, overrides, expected,
):
    source = tmp_path / "experiment.toml"
    source.write_text("[execution]\nmin_train_size=20\n" + source_execution)
    assert parse(["--config", str(source), *overrides]).config.execution == expected


def test_model_selection_retains_saved_parameters_and_rebuilds_comparison_defaults(tmp_path):
    source = tmp_path / "experiment.toml"
    source.write_text('''
[[models]]
name = "naive"
[[models]]
name = "EWMA"
decay = 0.8
[[models]]
name = "HAR"
''')
    run = parse(["--config", str(source), "--models", "EWMA", "HAR"])
    assert [model.name for model in run.definition.models] == ["EWMA", "HAR"]
    assert dict(run.definition.models[0].specification.parameters) == {"decay": 0.8}
    assert run.definition.baseline_models == ("EWMA",)
    assert run.definition.comparison_pairs == (("EWMA", "HAR"),)


def test_toml_loader_supports_dates_and_reports_bad_or_missing_files(tmp_path):
    source = tmp_path / "experiment.toml"
    source.write_text("[run]\nforecast_start=2024-01-01")
    assert load_experiment_configuration(source)["run"]["forecast_start"] == date(2024, 1, 1)
    source.write_text("invalid = [")
    with pytest.raises(ValueError, match="cannot read experiment configuration"):
        load_experiment_configuration(source)
    with pytest.raises(ValueError, match="cannot read experiment configuration"):
        load_experiment_configuration(tmp_path / "missing.toml")


BASELINE_CONFIG = Path(__file__).resolve().parents[1] / "experiments" / "eurusd.toml"


@pytest.mark.parametrize("arguments, models, candidate, baselines, pairs", [
    (["--models", "HAR", "naive"], ["HAR", "naive"], "HAR", ("naive",), (("naive", "HAR"),)),
    (["--models", "HAR", "EWMA"], ["HAR", "EWMA"], "HAR", ("EWMA",), (("EWMA", "HAR"),)),
    (["--candidate", "naive"], ["historical_mean", "naive", "EWMA", "AR1", "HAR"], "naive",
     ("EWMA", "AR1", "historical_mean", "HAR"), (("EWMA", "naive"),)),
    (["--models", "EWMA", "naive", "--candidate", "naive"], ["EWMA", "naive"], "naive",
     ("EWMA",), (("EWMA", "naive"),)),
    (["--baselines", "EWMA"], ["historical_mean", "naive", "EWMA", "AR1", "HAR"], "HAR",
     ("EWMA",), (("EWMA", "HAR"),)),
    (["--models", "EWMA", "naive", "--candidate", "naive", "--compare", "naive", "EWMA"],
     ["EWMA", "naive"], "naive", ("EWMA",), (("naive", "EWMA"),)),
])
def test_overrides_reconcile_inherited_model_references(
    arguments, models, candidate, baselines, pairs,
):
    unchanged = BASELINE_CONFIG.read_bytes()
    run = parse(["--config", str(BASELINE_CONFIG), *arguments])
    assert [model.name for model in run.definition.models] == models
    assert run.definition.candidate_model == candidate
    assert run.definition.baseline_models == baselines
    assert run.definition.comparison_pairs == pairs
    assert BASELINE_CONFIG.read_bytes() == unchanged


def test_model_selection_preserves_surviving_comparisons_and_their_order(tmp_path):
    path = tmp_path / "pairs.toml"
    path.write_text(BASELINE_CONFIG.read_text().replace(
        'comparison_pairs = [["naive", "HAR"]]',
        'comparison_pairs = [["EWMA", "HAR"], ["AR1", "HAR"], ["naive", "HAR"]]',
    ))
    run = parse(["--config", str(path), "--models", "naive", "HAR", "EWMA"])
    assert run.definition.comparison_pairs == (("EWMA", "HAR"), ("naive", "HAR"))


@pytest.mark.parametrize("arguments", [
    ["--models", "HAR", "naive", "--baselines", "AR1"],
    ["--candidate", "naive", "--baselines", "naive"],
    ["--models", "HAR", "naive", "--compare", "EWMA", "HAR"],
    ["--models", "naive", "EWMA"],
])
def test_invalid_explicit_model_references_are_not_silently_discarded(arguments, capsys):
    with pytest.raises(SystemExit) as error:
        parse(["--config", str(BASELINE_CONFIG), *arguments])
    assert error.value.code == 2
    assert "Traceback" not in capsys.readouterr().err


def test_an_inherited_empty_comparison_list_remains_disabled(tmp_path):
    path = tmp_path / "disabled.toml"
    path.write_text(BASELINE_CONFIG.read_text().replace(
        'comparison_pairs = [["naive", "HAR"]]', 'comparison_pairs = []',
    ))
    run = parse(["--config", str(path), "--candidate", "naive"])
    assert run.definition.comparison_pairs == ()


def test_metric_selection_reconciles_only_the_inherited_plot_metric():
    run = parse(["--config", str(BASELINE_CONFIG), "--metrics", "log-RV MSE"])
    assert run.plot_metric == "log-RV MSE"
    retained = parse(["--config", str(BASELINE_CONFIG), "--metrics", "log-RV MSE", "QLIKE"])
    assert retained.plot_metric == "QLIKE"
    with pytest.raises(SystemExit):
        parse(["--config", str(BASELINE_CONFIG), "--metrics", "log-RV MSE", "--metric", "QLIKE"])


@pytest.mark.parametrize("execution, message", [
    ({"name": "holdout"}, "execution.train_end is required"),
    ({"train_end": "2024-12-31"}, "execution.train_end requires execution.name"),
    ({"update_observations": True}, "execution.update_observations requires execution.name"),
    ({"name": "holdout", "train_end": "2024-12-31", "window_size": 504},
     "execution.window_size requires execution.name"),
    ({"training_window": "rolling"}, "execution.window_size is required"),
    ({"window_size": 504}, "execution.window_size requires execution.training_window"),
])
def test_configuration_errors_name_fields_instead_of_cli_flags(execution, message):
    with pytest.raises(ValueError, match=message) as error:
        experiment_run_from_mapping({"execution": execution})
    assert "--" not in str(error.value)
