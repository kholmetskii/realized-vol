from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

from rvol.domain import (
    ComponentSpecification,
    ExperimentConfig,
    HoldoutSettings,
    WalkForwardSettings,
)
from rvol.forecasting.composition import (
    Experiment,
    standard_experiment,
    standard_forecasters,
)
from rvol.forecasting.evaluation import LogMSEMetric
from rvol.forecasting.execution import HoldoutExperiment, WalkForwardExperiment
from rvol.forecasting.models import HistoricalMeanForecaster, NaiveForecaster
from rvol.forecasting.targets import RealizedVarianceTargetBuilder


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
    result = standard_experiment(
        ExperimentConfig(execution=WalkForwardSettings(min_train_size=10)),
    ).run(daily_sample())

    assert result.models == ("AR1", "EWMA", "HAR", "historical_mean", "naive")
    assert result.n_forecasts == 90
    assert {record.n_train for record in result.records} == set(range(10, 28))


def test_standard_experiment_preserves_each_workflows_comparison_choices():
    definition = standard_experiment(ExperimentConfig(
        execution=WalkForwardSettings(min_train_size=10),
    ))
    experiment = definition.run(daily_sample())

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
    assert [item["baseline_model"] for item in robustness.check("win_rates").records()[:4]] == [
        "AR1", "EWMA", "historical_mean", "naive",
    ]


def test_definition_preserves_comparison_overrides_including_empty_pairs():
    definition = standard_experiment(ExperimentConfig(
        execution=WalkForwardSettings(min_train_size=10),
    ))
    experiment = definition.run(daily_sample())

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
                "origin_date": daily["date"],
                "rv_daily": daily["log_rv"],
            })

    definition = Experiment(
        config=ExperimentConfig(execution=WalkForwardSettings(min_train_size=3)),
        models=(HistoricalMeanForecaster(), NaiveForecaster()),
        feature_builder=DailyFeatures(),
        target_builder=RealizedVarianceTargetBuilder(),
        metrics=(LogMSEMetric(),),
        comparison_pairs=(("historical_mean", "naive"),),
        candidate_model="naive",
        baseline_models=("historical_mean",),
    )
    experiment = definition.run(daily_sample(8))
    evaluation = definition.evaluator(hac_lags=0).evaluate(experiment)
    robustness = definition.robustness_analyzer(hac_lags=(0,), top_errors=1).analyze(experiment)

    assert experiment.models == ("historical_mean", "naive")
    assert experiment.n_forecasts == 8
    assert {item.metric for item in evaluation.summaries} == {"log-RV MSE"}
    assert len(evaluation.comparisons) == len(robustness.check("hac_sensitivity").rows) == 1
    assert evaluation.comparisons[0].candidate_model == "naive"
    assert robustness.check("win_rates").records()[0]["candidate_model"] == "naive"
    assert definition.all_comparison_pairs == (("historical_mean", "naive"),)
    assert definition.specification.features.name == "daily"


def test_standard_definitions_do_not_share_model_or_metric_instances():
    first = standard_experiment()
    second = standard_experiment()

    assert all(a is not b for a, b in zip(first.models, second.models, strict=True))
    assert all(a is not b for a, b in zip(first.metrics, second.metrics, strict=True))


def test_experiment_binds_config_for_repeatable_execution_without_mutating_input():
    config = ExperimentConfig(execution=WalkForwardSettings(min_train_size=10))
    run = standard_experiment(config)
    daily = daily_sample()
    unchanged = daily.copy(deep=True)

    assert run.run(daily) == run.run(daily)
    assert run.config is config
    pd.testing.assert_frame_equal(daily, unchanged)


def test_definition_rejects_references_to_models_that_are_not_configured():
    with pytest.raises(ValueError, match="unknown models: missing"):
        replace(standard_experiment(), candidate_model="missing")


@pytest.mark.parametrize("settings", [
    WalkForwardSettings(min_train_size=10),
    HoldoutSettings(date(2022, 2, 18), min_train_size=10),
    HoldoutSettings(date(2022, 2, 18), min_train_size=10, update_observations=True),
])
def test_experiment_dispatches_settings_to_the_matching_chronological_executor(settings):
    definition = standard_experiment(ExperimentConfig(execution=settings))
    daily = daily_sample()
    prepared = definition.prepare(daily)
    unchanged = daily.copy(deep=True)
    executor = HoldoutExperiment if isinstance(settings, HoldoutSettings) else WalkForwardExperiment

    expected = executor(definition.models, definition.config).run(prepared)
    result = definition.run(daily)

    assert result == expected
    assert definition.run(daily) == result
    assert result.execution is not None
    assert result.execution.executor.endswith(executor.__name__)
    assert result.n_forecasts > 0
    assert len(definition.evaluator().evaluate(result).summaries) == 10
    pd.testing.assert_frame_equal(daily, unchanged)
