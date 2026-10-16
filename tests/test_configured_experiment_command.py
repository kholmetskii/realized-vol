import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rvol.cli import run_experiment
from rvol.domain import (
    ComponentSpecification,
    EveryNSessions,
    ExperimentConfig,
    ExperimentResult,
    HoldoutSettings,
    RollingWindow,
    RunIdentity,
    TargetHorizon,
    WalkForwardSettings,
)
from rvol.forecasting.composition import (
    Experiment,
    standard_experiment,
)
from rvol.forecasting.evaluation import (
    DieboldMarianoTest,
    EvaluationResult,
    ForecastEvaluator,
    ForecastRobustnessAnalyzer,
    LagSensitivityCheck,
    LargestErrorsCheck,
    LogMSEMetric,
    MonthlyLossCheck,
    WinRateCheck,
)
from rvol.forecasting.features import HARFeatureBuilder
from rvol.forecasting.models import EWMAModel, HARModel
from rvol.forecasting.targets import RealizedVarianceTargetBuilder
from rvol.presentation.reporting import (
    DatasetSnapshot,
    ForecastArtifactWriter,
    RobustnessArtifactWriter,
)


def configured_experiment(executor: str) -> Experiment:
    settings = (
        WalkForwardSettings(RollingWindow(25), EveryNSessions(3), min_train_size=20)
        if executor == "walk-forward" else HoldoutSettings(date(2022, 2, 18), 20, True)
    )
    horizon = TargetHorizon(3, "cumulative")
    base = standard_experiment(ExperimentConfig(
        execution=settings, instrument="GBP/USD", identity=RunIdentity("configured", "test"),
        forecast_start=date(2022, 3, 14), forecast_end=date(2022, 3, 21),
    ), target_horizon=horizon)
    naive = next(model for model in base.models if model.name == "naive")
    return replace(
        base,
        feature_builder=HARFeatureBuilder("session", "value", weekly_window=2, monthly_window=4),
        target_builder=RealizedVarianceTargetBuilder(horizon, "session", "realized_value"),
        models=(naive, EWMAModel(decay=0.8), HARModel()),
        metrics=(LogMSEMetric(),), plot_metric="log-RV MSE", candidate_model="EWMA",
        baseline_models=("naive",), comparison_pairs=(("naive", "EWMA"),),
        comparison_test=DieboldMarianoTest(hac_lags=0, min_hac_lags=2),
        diagnostic_checks=(
            LagSensitivityCheck(hac_lags=(0, 1)),
            MonthlyLossCheck(),
            WinRateCheck(),
            LargestErrorsCheck(top_errors=2),
        ),
    )


@pytest.mark.parametrize("executor", ["walk-forward", "holdout"])
@pytest.mark.parametrize("reports", [[], ["--plot"], ["--robustness"], ["--plot", "--robustness"]],
                         ids=["numbers", "plot", "robustness", "all"])
def test_command_runs_and_evaluates_once_and_all_outputs_match_the_python_api(
    tmp_path, monkeypatch, executor, reports,
):
    from rvol.presentation.plotting import forecasting

    daily = pd.DataFrame({
        "session": pd.bdate_range("2022-01-03", periods=75),
        "value": -10 + np.sin(np.arange(75) / 4) / 5,
        "realized_value": -9 + np.cos(np.arange(75) / 3) / 4,
    })
    source = tmp_path / "daily.parquet"
    daily.to_parquet(source, index=False)
    expected_run = configured_experiment(executor)
    expected = expected_run.run(daily)
    expected_evaluation = expected_run.evaluator().evaluate(expected)
    expected_configuration = json.loads(json.dumps(expected_run.resolved_configuration))
    snapshot = DatasetSnapshot.from_frame(source, daily, date_column="session")
    expected_paths = ForecastArtifactWriter(tmp_path / "expected").write(
        expected, expected_evaluation, config=expected_run.config,
        specification=expected_run.specification, dataset=snapshot,
        configuration=expected_run.resolved_configuration,
    ).all()
    if "--robustness" in reports:
        expected_paths += RobustnessArtifactWriter(tmp_path / "expected" / "robustness").write(
            expected_run.robustness_analyzer().analyze(expected, evaluation=expected_evaluation),
            configuration=expected_run.resolved_configuration,
        ).all()

    executions: list[ExperimentResult] = []
    evaluations: list[EvaluationResult] = []
    plot_records: list[ExperimentResult] = []
    diagnostic_records: list[ExperimentResult] = []
    original_run = Experiment.run
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

    monkeypatch.setattr(Experiment, "run", capture_run)
    monkeypatch.setattr(ForecastEvaluator, "evaluate", capture_evaluate)
    monkeypatch.setattr(ForecastRobustnessAnalyzer, "analyze", capture_analyze)
    monkeypatch.setattr(forecasting, "plot_forecast_evaluation", capture_plot)
    output = tmp_path / "output"
    arguments = [str(source), "--output-dir", str(output), *reports]
    run_experiment.main(expected_run, argv=arguments)
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
    run_experiment.main(expected_run, argv=arguments)
    assert len(executions) == len(evaluations) == 2
    assert expected_bytes == {path: (output / path).read_bytes() for path in expected_bytes}


@pytest.mark.parametrize("arguments", [
    ["--models", "missing"], ["--models", "HAR", "HAR"],
    ["--models", "naive", "EWMA"], ["--candidate", "missing"],
    ["--baselines", "missing"], ["--baselines", "EWMA", "EWMA"],
    ["--run-id", ""], ["--forecast-start", "2022-02-30"],
    ["--forecast-start", "2022-03-22", "--forecast-end", "2022-03-14"],
])
def test_bad_selections_fail_before_loading_data(monkeypatch, capsys, arguments):
    def unexpected_load(self):
        raise AssertionError("selections must be validated before loading data")

    monkeypatch.setattr(run_experiment.ParquetDatasetRepository, "load", unexpected_load)
    with pytest.raises(SystemExit) as error:
        run_experiment.main(argv=["missing.parquet", *arguments])
    assert error.value.code == 2
    assert "Traceback" not in capsys.readouterr().err


def test_plot_title_is_generic_when_instrument_is_unspecified(tmp_path, monkeypatch):
    from rvol.presentation.plotting import forecasting

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
    experiment = standard_experiment(ExperimentConfig(
        execution=WalkForwardSettings(min_train_size=20),
    ))
    run_experiment.main(experiment, argv=[
        str(source), "--plot", "--output-dir", str(tmp_path / "output"),
    ])
    assert titles[0].startswith("Out-of-sample")
    assert "EUR/USD" not in titles[0]


@pytest.mark.parametrize("arguments", [
    ["--out", "forecast.png"], ["--metr", "QLIKE", "--plot"],
])
def test_unsupported_abbreviations_fail_before_loading_data(monkeypatch, capsys, arguments):
    def unexpected_load(self):
        raise AssertionError("unsupported options must fail before data I/O")

    monkeypatch.setattr(run_experiment.ParquetDatasetRepository, "load", unexpected_load)
    with pytest.raises(SystemExit) as error:
        run_experiment.main(argv=["missing.parquet", *arguments])
    assert error.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


class IndependentTargets:
    horizon = TargetHorizon()
    specification = ComponentSpecification("independent_targets", "test.IndependentTargets")

    def __call__(self, daily: pd.DataFrame) -> pd.DataFrame:
        return RealizedVarianceTargetBuilder(self.horizon)(daily)


@pytest.fixture
def canonical_dataset(tmp_path):
    daily = pd.DataFrame({
        "date": pd.bdate_range("2022-01-03", periods=75),
        "log_rv": -10 + np.sin(np.arange(75) / 4) / 5,
    })
    source = tmp_path / "daily.parquet"
    daily.to_parquet(source, index=False)
    return source, daily


def test_command_accepts_an_independent_target_builder_without_source_column_parameters(
    tmp_path, canonical_dataset,
):
    source, daily = canonical_dataset
    experiment = replace(
        standard_experiment(ExperimentConfig(execution=WalkForwardSettings(min_train_size=20))),
        target_builder=IndependentTargets(),
    )
    output = tmp_path / "output"
    run_experiment.main(experiment, argv=[str(source), "--output-dir", str(output)])
    result = experiment.run(daily)
    expected_paths = ForecastArtifactWriter(tmp_path / "expected").write(
        result, experiment.evaluator().evaluate(result), config=experiment.config,
        dataset=DatasetSnapshot.from_frame(source, daily), specification=experiment.specification,
        configuration=experiment.resolved_configuration,
    ).all()

    assert {p.name: p.read_bytes() for p in expected_paths} == {
        p.name: (output / p.name).read_bytes() for p in expected_paths
    }
    metadata = json.loads((output / "experiment.json").read_text())
    assert metadata["specification"]["target"]["parameters"] == {}


@pytest.mark.parametrize("date_column", ["", " ", 123, "absent"])
def test_invalid_target_source_date_column_fails_before_experiment_execution(
    monkeypatch, canonical_dataset, date_column,
):
    source, _ = canonical_dataset
    builder = IndependentTargets()
    builder.specification = replace(builder.specification, parameters=(("date_col", date_column),))
    experiment = replace(standard_experiment(), target_builder=builder)

    def unexpected_run(self, daily):
        raise AssertionError("source dates must be validated before fitting models")

    monkeypatch.setattr(Experiment, "run", unexpected_run)
    with pytest.raises(ValueError, match="date_col|date column"):
        run_experiment.main(experiment, argv=[str(source)])
