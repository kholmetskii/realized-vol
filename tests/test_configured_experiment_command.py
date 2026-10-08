import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rvol.cli import run_experiment
from rvol.composition import ExperimentRun, experiment_run_from_mapping
from rvol.domain import ExperimentResult
from rvol.evaluation import (
    EvaluationResult,
    ForecastEvaluator,
    ForecastRobustnessAnalyzer,
)
from rvol.infrastructure.configuration import load_experiment_configuration
from rvol.reporting import (
    DatasetSnapshot,
    ForecastArtifactWriter,
    RobustnessArtifactWriter,
)

CONFIGURATION = '''
version = 1
[dataset]
instrument = "GBP/USD"
[run]
run_id = "configured"
fold_id = "test"
forecast_start = 2022-03-14
forecast_end = 2022-03-21
[features]
name = "HAR"
date_col = "session"
rv_col = "value"
weekly_window = 2
monthly_window = 4
[target]
name = "realized_variance"
date_col = "session"
rv_col = "realized_value"
sessions = 3
kind = "cumulative"
[[models]]
name = "naive"
[[models]]
name = "EWMA"
decay = 0.8
[[models]]
name = "HAR"
[evaluation]
metrics = ["log-RV MSE"]
candidate = "EWMA"
baselines = ["naive"]
comparison_pairs = [["naive", "EWMA"]]
[evaluation.test]
name = "DM"
hac_lags = 0
[robustness]
checks = ["hac_sensitivity", "monthly_losses", "win_rates", "largest_errors"]
hac_lags = [0, 1]
top_errors = 2
'''


@pytest.mark.parametrize("executor", ["walk-forward", "holdout"])
@pytest.mark.parametrize("reports", [[], ["--plot"], ["--robustness"], ["--plot", "--robustness"]],
                         ids=["numbers", "plot", "robustness", "all"])
def test_command_runs_and_evaluates_once_and_all_outputs_match_the_python_api(
    tmp_path, monkeypatch, executor, reports,
):
    from rvol.plotting import forecasting

    daily = pd.DataFrame({
        "session": pd.bdate_range("2022-01-03", periods=75),
        "value": -10 + np.sin(np.arange(75) / 4) / 5,
        "realized_value": -9 + np.cos(np.arange(75) / 3) / 4,
    })
    source = tmp_path / "daily.parquet"
    daily.to_parquet(source, index=False)
    config_path = tmp_path / "experiment.toml"
    execution = (
        'training_window="rolling"\nwindow_size=25\nretrain_every=3'
        if executor == "walk-forward" else 'train_end=2022-02-18\nupdate_observations=true'
    )
    config_path.write_text(
        CONFIGURATION + f'\n[execution]\nname="{executor}"\nmin_train_size=20\n' + execution,
    )
    expected_run = experiment_run_from_mapping(load_experiment_configuration(config_path))
    expected = expected_run.run(daily)
    expected_evaluation = expected_run.definition.evaluator().evaluate(expected)
    expected_configuration = json.loads(json.dumps(expected_run.resolved_configuration))
    snapshot = DatasetSnapshot.from_frame(source, daily, date_column="session")
    expected_paths = ForecastArtifactWriter(tmp_path / "expected").write(
        expected, expected_evaluation, config=expected_run.config,
        specification=expected_run.definition.specification, dataset=snapshot,
        configuration=expected_run.resolved_configuration,
    ).all()
    if "--robustness" in reports:
        expected_paths += RobustnessArtifactWriter(tmp_path / "expected" / "robustness").write(
            expected_run.definition.robustness_analyzer(
                checks=expected_run.diagnostic_checks,
            ).analyze(expected, evaluation=expected_evaluation),
            configuration=expected_run.resolved_configuration,
        ).all()

    executions: list[ExperimentResult] = []
    evaluations: list[EvaluationResult] = []
    plot_records: list[ExperimentResult] = []
    diagnostic_records: list[ExperimentResult] = []
    original_run = ExperimentRun.run
    original_evaluate = ForecastEvaluator.evaluate
    original_analyze = ForecastRobustnessAnalyzer.analyze
    original_plot = forecasting.plot_forecast_evaluation

    def capture_run(self, data):
        result = original_run(self, data)
        executions.append(result)
        return result

    def capture_evaluate(self, experiment):
        assert experiment is executions[-1]
        result = original_evaluate(self, experiment)
        evaluations.append(result)
        return result

    def capture_analyze(self, experiment, *, evaluation=None):
        assert experiment is executions[-1]
        assert evaluation is evaluations[-1]
        diagnostic_records.append(experiment)
        return original_analyze(self, experiment, evaluation=evaluation)

    def capture_plot(experiment, evaluation, **kwargs):
        assert experiment is executions[-1]
        assert evaluation is evaluations[-1]
        plot_records.append(experiment)
        figure = original_plot(experiment, evaluation, **kwargs)
        title = figure.texts[-1].get_text()
        assert title.startswith("GBP/USD Out-of-sample")
        assert "EUR/USD" not in title
        labels = [line.get_label() for line in figure.axes[2].lines[1:]]
        assert labels == ["Naïve − EWMA"]
        return figure

    monkeypatch.setattr(ExperimentRun, "run", capture_run)
    monkeypatch.setattr(ForecastEvaluator, "evaluate", capture_evaluate)
    monkeypatch.setattr(ForecastRobustnessAnalyzer, "analyze", capture_analyze)
    monkeypatch.setattr(forecasting, "plot_forecast_evaluation", capture_plot)
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", [
        "run_experiment", str(source), "--config", str(config_path),
        "--output-dir", str(output), *reports,
    ])
    run_experiment.main()
    assert executions == [expected]
    assert evaluations == [expected_evaluation]
    assert plot_records == ([expected] if "--plot" in reports else [])
    assert diagnostic_records == ([expected] if "--robustness" in reports else [])

    payload = json.loads((output / "experiment.json").read_text())
    assert payload["dataset"]["instrument"] == "GBP/USD"
    assert payload["resolved_configuration"] == expected_configuration
    expected_bytes = {
        path.relative_to(tmp_path / "expected"): path.read_bytes() for path in expected_paths
    }
    assert expected_bytes == {path: (output / path).read_bytes() for path in expected_bytes}
    if "--plot" in reports:
        metadata = output / "forecasts.png.json"
        payload = json.loads(metadata.read_text())
        assert payload["dataset"]["instrument"] == "GBP/USD"
        assert payload["dataset"]["sha256"] == snapshot.sha256
        assert payload["resolved_configuration"] == expected_configuration
        assert (output / "forecasts.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        expected_bytes[Path("forecasts.png.json")] = metadata.read_bytes()
    else:
        assert not (output / "forecasts.png").exists()
        assert not (output / "forecasts.png.json").exists()
    if "--robustness" in reports:
        payload = json.loads((output / "robustness" / "robustness.json").read_text())
        assert payload["resolved_configuration"] == expected_configuration
        assert "Instrument: GBP/USD" in (output / "robustness" / "robustness.md").read_text()
    else:
        assert not (output / "robustness").exists()
    run_experiment.main()
    assert len(executions) == len(evaluations) == 2
    assert expected_bytes == {path: (output / path).read_bytes() for path in expected_bytes}


@pytest.mark.parametrize("contents", [
    "unknown=1", "version=2", "models=[]", '[features]\nmonthly_window=2',
    '[[models]]\nname="EWMA"\ndecay="oops"', '[evaluation]\nmetrics=["unknown"]',
    '[evaluation.test]\nhac_lags=-1', '[plot]\nmetric="unknown"',
    '[robustness]\nchecks=["unknown"]', '[robustness]\nhac_lags=[[0]]',
    '[execution]\nname="holdout"', 'invalid=[',
    '[target]\nname="unknown"', '[target]\nsessions=0', '[target]\ndate_col=""',
])
def test_bad_configurations_fail_before_loading_data(tmp_path, monkeypatch, capsys, contents):
    path = tmp_path / "bad.toml"
    path.write_text(contents)

    def unexpected_load(self):
        raise AssertionError("configuration must be validated before loading data")

    monkeypatch.setattr(run_experiment.ParquetDatasetRepository, "load", unexpected_load)
    monkeypatch.setattr(sys, "argv", ["run_experiment", "missing.parquet", "--config", str(path)])
    with pytest.raises(SystemExit) as error:
        run_experiment.main()
    assert error.value.code == 2
    assert "Traceback" not in capsys.readouterr().err


def test_plot_title_is_generic_when_instrument_is_unspecified(tmp_path, monkeypatch):
    from rvol.plotting import forecasting

    daily = pd.DataFrame({
        "date": pd.bdate_range("2022-01-03", periods=75),
        "log_rv": -10 + np.sin(np.arange(75) / 4) / 5,
    })
    source = tmp_path / "daily.parquet"
    daily.to_parquet(source, index=False)
    plot = forecasting.plot_forecast_evaluation
    titles = []

    def capture(*args, **kwargs):
        figure = plot(*args, **kwargs)
        titles.append(figure.texts[-1].get_text())
        return figure

    monkeypatch.setattr(forecasting, "plot_forecast_evaluation", capture)
    monkeypatch.setattr(sys, "argv", [
        "run_experiment", str(source), "--min-train-size", "20", "--plot",
        "--output-dir", str(tmp_path / "output"),
    ])
    run_experiment.main()
    assert titles[0].startswith("Out-of-sample")
    assert "EUR/USD" not in titles[0]


@pytest.mark.parametrize("arguments, message", [
    (["--metric", "QLIKE"], "--metric requires --plot"),
    (["--check", "monthly_losses"], "--check requires --robustness"),
    (["--max-hac-lag", "1"], "--max-hac-lag requires --robustness"),
    (["--top-errors", "2"], "--top-errors requires --robustness"),
    (["--out", "forecast.png"], "unrecognized arguments"),
    (["--metr", "QLIKE", "--plot"], "unrecognized arguments"),
])
def test_options_without_an_effect_fail_before_loading_data(
    monkeypatch, capsys, arguments, message,
):
    def unexpected_load(self):
        raise AssertionError("unsupported options must fail before data I/O")

    monkeypatch.setattr(run_experiment.ParquetDatasetRepository, "load", unexpected_load)
    monkeypatch.setattr(sys, "argv", ["run_experiment", "missing.parquet", *arguments])
    with pytest.raises(SystemExit) as error:
        run_experiment.main()
    assert error.value.code == 2
    assert message in capsys.readouterr().err
