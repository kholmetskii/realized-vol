from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

from rvol.application import WalkForwardExperiment
from rvol.composition import (
    ExperimentDefinition,
    ExperimentRun,
    run_standard_forecast_experiment,
    standard_experiment_definition,
    standard_forecasters,
)
from rvol.domain import (
    ComponentSpecification,
    ExperimentConfig,
    ExperimentExecutor,
    ExperimentResult,
    Forecaster,
    ForecastRecord,
    WalkForwardSettings,
)
from rvol.evaluation import LogMSEMetric
from rvol.models import HistoricalMeanForecaster, NaiveForecaster


def daily_sample(n_sessions: int = 50) -> pd.DataFrame:
    step = np.arange(n_sessions, dtype="float64")
    log_rv = -10.0 + 0.2 * np.sin(step / 4) + 0.002 * step
    return pd.DataFrame({
        "date": pd.bdate_range("2022-01-03", periods=n_sessions),
        "rv_5min": np.exp(log_rv),
        "log_rv": log_rv,
        "observation_count": np.full(n_sessions, 288),
    })


def test_standard_forecasters_define_the_project_model_set():
    assert [model.name for model in standard_forecasters()] == [
        "historical_mean",
        "naive",
        "EWMA",
        "AR1",
        "HAR",
    ]


def test_standard_experiment_builds_features_and_runs_every_model():
    result = run_standard_forecast_experiment(
        daily_sample(),
        ExperimentConfig(execution=WalkForwardSettings(min_train_size=10)),
    )

    assert result.models == ("AR1", "EWMA", "HAR", "historical_mean", "naive")
    assert result.n_forecasts == 90
    assert {record.n_train for record in result.records} == set(range(10, 28))


def test_standard_definition_preserves_each_workflows_comparison_choices():
    definition = standard_experiment_definition()
    assert definition.executor_factory is WalkForwardExperiment
    assert isinstance(
        definition.executor_factory(definition.models, ExperimentConfig(
            execution=WalkForwardSettings(),
        )), ExperimentExecutor,
    )
    experiment = definition.run(daily_sample(), ExperimentConfig(
        execution=WalkForwardSettings(min_train_size=10),
    ))

    evaluation = definition.evaluator(hac_lags=0).evaluate(experiment)
    robustness = definition.robustness_analyzer(hac_lags=(0,), top_errors=1).analyze(experiment)

    assert [metric.name for metric in definition.metrics] == ["QLIKE", "log-RV MSE"]
    assert definition.candidate_model == "HAR"
    assert definition.all_comparison_pairs == (
        ("AR1", "HAR"), ("EWMA", "HAR"), ("historical_mean", "HAR"), ("naive", "HAR"),
    )
    assert [
        (item.baseline_model, item.candidate_model) for item in evaluation.comparisons
    ] == [("naive", "HAR"), ("naive", "HAR")]
    assert [item.baseline_model for item in robustness.win_rates[:4]] == [
        "naive", "EWMA", "AR1", "historical_mean",
    ]


def test_definition_preserves_comparison_overrides_including_empty_pairs():
    definition = standard_experiment_definition()
    experiment = definition.run(daily_sample(), ExperimentConfig(
        execution=WalkForwardSettings(min_train_size=10),
    ))

    overridden = definition.evaluator(
        comparison_pairs=(("EWMA", "AR1"),), hac_lags=0,
    ).evaluate(experiment)
    summaries_only = definition.evaluator(comparison_pairs=()).evaluate(experiment)

    assert {
        (item.baseline_model, item.candidate_model) for item in overridden.comparisons
    } == {("EWMA", "AR1")}
    assert summaries_only.comparisons == ()
    assert summaries_only.summaries == overridden.summaries


def test_definition_supports_other_feature_builders_models_metrics_and_candidates():
    class DailyFeatures:
        specification = ComponentSpecification(
            name="daily", implementation="test.DailyFeatures", feature_names=("rv_daily",),
        )

        def __call__(self, daily: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame({
                "origin_date": daily["date"].iloc[:-1].to_numpy(),
                "target_date": daily["date"].iloc[1:].to_numpy(),
                "rv_daily": daily["log_rv"].iloc[:-1].to_numpy(),
                "target": daily["log_rv"].iloc[1:].to_numpy(),
            })

    definition = ExperimentDefinition(
        models=(HistoricalMeanForecaster(), NaiveForecaster()),
        feature_builder=DailyFeatures(),
        executor_factory=WalkForwardExperiment,
        metrics=(LogMSEMetric(),),
        comparison_pairs=(("historical_mean", "naive"),),
        candidate_model="naive",
        baseline_models=("historical_mean",),
    )
    experiment = definition.run(daily_sample(8), ExperimentConfig(
        execution=WalkForwardSettings(min_train_size=3),
    ))
    evaluation = definition.evaluator(hac_lags=0).evaluate(experiment)
    robustness = definition.robustness_analyzer(hac_lags=(0,), top_errors=1).analyze(experiment)

    assert experiment.models == ("historical_mean", "naive")
    assert experiment.n_forecasts == 8
    assert {item.metric for item in evaluation.summaries} == {"log-RV MSE"}
    assert len(evaluation.comparisons) == len(robustness.hac_comparisons) == 1
    assert evaluation.comparisons[0].candidate_model == "naive"
    assert robustness.win_rates[0].candidate_model == "naive"
    assert definition.all_comparison_pairs == (("historical_mean", "naive"),)
    assert definition.specification.features.name == "daily"


def test_standard_definitions_do_not_share_model_or_metric_instances():
    first = standard_experiment_definition()
    second = standard_experiment_definition()

    assert all(a is not b for a, b in zip(first.models, second.models, strict=True))
    assert all(a is not b for a, b in zip(first.metrics, second.metrics, strict=True))


def test_experiment_run_binds_definition_and_config_for_repeatable_execution():
    definition = standard_experiment_definition()
    config = ExperimentConfig(execution=WalkForwardSettings(min_train_size=10))
    run = ExperimentRun(definition, config)
    daily = daily_sample()

    assert run.run(daily) == definition.run(daily, config)
    assert run.run(daily) == run.run(daily)
    assert run.definition is definition
    assert run.config is config


def test_definition_rejects_references_to_models_that_are_not_configured():
    with pytest.raises(ValueError, match="unknown models: missing"):
        replace(standard_experiment_definition(), candidate_model="missing")


def test_definition_delegates_to_an_independent_executor_with_prepared_features():
    prepared = pd.DataFrame({"custom_input": [1.0, 2.0]})

    @dataclass(frozen=True)
    class CustomSettings:
        sample_count: int

    class CustomFeatures:
        specification = ComponentSpecification("custom", "test.CustomFeatures")

        def __call__(self, daily: pd.DataFrame) -> pd.DataFrame:
            return prepared.copy()

    class CustomExecutor:
        def __init__(self, models: Sequence[Forecaster], config: ExperimentConfig[CustomSettings]):
            self.models = models
            self.config = config
            self.received: list[pd.DataFrame] = []
            self.result = ExperimentResult(records=(ForecastRecord(
                model=models[0].name, origin_date=date(2022, 1, 3),
                target_date=date(2022, 1, 4), n_train=config.execution.sample_count,
                actual_log_rv=-10.0, predicted_log_rv=-9.0,
            ),), config=config)

        def run(self, features: pd.DataFrame) -> ExperimentResult:
            self.received.append(features.copy())
            return self.result

    created: list[CustomExecutor] = []

    def create_executor(
        models: Sequence[Forecaster], config: ExperimentConfig[CustomSettings],
    ) -> CustomExecutor:
        executor = CustomExecutor(models, config)
        created.append(executor)
        return executor

    definition = replace(
        standard_experiment_definition(executor_factory=create_executor),
        models=(HistoricalMeanForecaster(),), feature_builder=CustomFeatures(),
        metrics=(LogMSEMetric(),), comparison_pairs=(),
        candidate_model="historical_mean", baseline_models=(),
    )
    daily = daily_sample()
    unchanged = daily.copy(deep=True)
    configs = (ExperimentConfig(execution=CustomSettings(10)),
               ExperimentConfig(execution=CustomSettings(15)))

    results = [definition.run(daily, config) for config in configs]

    assert len(created) == 2
    assert created[0] is not created[1]
    for executor, config, result in zip(created, configs, results, strict=True):
        assert isinstance(executor, ExperimentExecutor)
        assert executor.models is definition.models
        assert executor.config is config
        assert result is executor.result
        assert len(executor.received) == 1
        pd.testing.assert_frame_equal(executor.received[0], prepared)
        evaluation = definition.evaluator().evaluate(result)
        assert len(evaluation.summaries) == 1
        assert evaluation.summaries[0].mean_loss == 1.0
    pd.testing.assert_frame_equal(daily, unchanged)
