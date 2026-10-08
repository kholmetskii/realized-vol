from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from rvol.application import HoldoutExperiment
from rvol.composition import standard_experiment_definition
from rvol.domain import ExperimentConfig, HoldoutSettings
from rvol.domain.contracts import FloatArray
from rvol.models import EWMAForecaster, HistoricalMeanForecaster, NaiveForecaster
from rvol.models.predictors import ConstantPredictor
from rvol.reporting import DatasetSnapshot, ForecastArtifactWriter


def features_sample(n: int = 30) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=n + 1)
    values = -10 + 0.2 * np.sin(np.arange(n + 1) / 3)
    return pd.DataFrame({
        "origin_date": dates[:-1], "target_date": dates[1:],
        "rv_daily": values[:-1], "target": values[1:],
    })


def cutoff_for(features: pd.DataFrame, position: int = 9) -> date:
    return pd.Timestamp(features["target_date"].iloc[position]).date()


class RecordingMean(HistoricalMeanForecaster):
    def __init__(self) -> None:
        self.training_targets: list[FloatArray] = []

    def fit(self, features: FloatArray, target: FloatArray) -> ConstantPredictor:
        self.training_targets.append(target.copy())
        return super().fit(features, target)


def test_holdout_fits_once_on_available_labels_and_retains_fit_boundaries():
    features = features_sample()
    unchanged = features.copy(deep=True)
    model = RecordingMean()
    cutoff = cutoff_for(features)
    config = ExperimentConfig(execution=HoldoutSettings(cutoff, min_train_size=5))
    experiment = HoldoutExperiment((model, NaiveForecaster()), config)

    result = experiment.run(features)

    assert len(model.training_targets) == 1
    np.testing.assert_array_equal(model.training_targets[0], features["target"].iloc[:10])
    assert result.n_forecasts == 40
    for record in result.records:
        assert record.origin_date >= cutoff
        assert record.target_date > cutoff
        assert record.n_train == 10
        assert record.fit_date == cutoff
        assert record.train_start_date == cutoff_for(features, 0)
        assert record.train_end_date == cutoff
    assert result.execution is not None
    assert result.execution.executor.endswith("HoldoutExperiment")
    assert result.execution.settings.policies == ()
    assert dict(result.execution.facts) == {
        "fit_cutoff": cutoff.isoformat(), "observation_update_policy": "disabled",
    }
    assert experiment.run(features) == result
    pd.testing.assert_frame_equal(features, unchanged)


@pytest.mark.parametrize("update", [False, True])
def test_holdout_reporting_dates_preserve_full_run_predictions_and_state(update):
    features = features_sample()
    settings = HoldoutSettings(cutoff_for(features), min_train_size=5, update_observations=update)
    config = ExperimentConfig(execution=settings)
    models = (EWMAForecaster(decay=0.5), NaiveForecaster())
    full = HoldoutExperiment(models, config).run(features)
    start, end = cutoff_for(features, 17), cutoff_for(features, 24)
    bounded = HoldoutExperiment(models, replace(
        config, forecast_start=start, forecast_end=end,
    )).run(features)

    assert bounded.records == tuple(r for r in full.records if start <= r.target_date <= end)
    assert bounded.execution == full.execution


@pytest.mark.parametrize("update", [False, True])
def test_future_outcomes_cannot_enter_their_own_forecast(update):
    original = features_sample()
    changed = original.copy()
    changed.loc[15:, "target"] += 1
    config = ExperimentConfig(execution=HoldoutSettings(
        cutoff_for(original), min_train_size=5, update_observations=update,
    ))
    runner = HoldoutExperiment((EWMAForecaster(), HistoricalMeanForecaster()), config)
    before = runner.run(original)
    after = runner.run(changed)
    day = cutoff_for(original, 15)

    assert [r.predicted_log_rv for r in before.records if r.target_date <= day] == [
        r.predicted_log_rv for r in after.records if r.target_date <= day
    ]


def test_holdout_uses_fresh_features_without_refitting():
    features = features_sample()
    result = HoldoutExperiment((NaiveForecaster(),), ExperimentConfig(
        execution=HoldoutSettings(cutoff_for(features), min_train_size=5),
    )).run(features)

    np.testing.assert_array_equal(
        [r.predicted_log_rv for r in result.records], features["rv_daily"].iloc[10:],
    )


@pytest.mark.parametrize("update", [False, True])
def test_holdout_ewma_observation_updates_follow_the_explicit_setting(update):
    features = features_sample()
    result = HoldoutExperiment((EWMAForecaster(decay=0.5),), ExperimentConfig(
        execution=HoldoutSettings(cutoff_for(features), 5, update),
    )).run(features)
    variances = np.exp(features["target"].to_numpy())
    state = variances[0]
    for value in variances[1:10]:
        state = 0.5 * state + 0.5 * value
    expected = []
    for position in range(10, len(features)):
        if update and position > 10:
            state = 0.5 * state + 0.5 * variances[position - 1]
        expected.append(state)

    np.testing.assert_allclose([r.predicted_rv for r in result.records], expected)


def test_holdout_updates_handle_delayed_labels_and_multirow_batches():
    features = pd.DataFrame({
        "origin_date": pd.to_datetime(["2024-01-01", "2024-01-04", "2024-01-05", "2024-01-10"]),
        "target_date": pd.to_datetime(["2024-01-03", "2024-01-07", "2024-01-09", "2024-01-11"]),
        "target": np.log([1.0, 4.0, 9.0, 16.0]),
    })
    result = HoldoutExperiment((EWMAForecaster(decay=0.5),), ExperimentConfig(
        execution=HoldoutSettings(date(2024, 1, 3), 1, True),
    )).run(features)

    np.testing.assert_allclose([r.predicted_rv for r in result.records], [1.0, 1.0, 5.75])
    assert {r.n_train for r in result.records} == {1}


def test_nontrading_cutoff_uses_only_labels_already_available():
    features = features_sample()
    friday = cutoff_for(features, 8)
    assert friday.weekday() == 4
    saturday = friday + timedelta(days=1)
    result = HoldoutExperiment((HistoricalMeanForecaster(),), ExperimentConfig(
        execution=HoldoutSettings(saturday, min_train_size=5),
    )).run(features)

    assert {r.n_train for r in result.records} == {9}
    assert {r.fit_date for r in result.records} == {saturday}
    assert {r.train_end_date for r in result.records} == {friday}
    assert result.records[0].origin_date > saturday


def test_holdout_requires_enough_training_data_at_the_fixed_cutoff():
    features = features_sample()
    with pytest.raises(ValueError, match="contains 10 observations.*requires at least 11"):
        HoldoutExperiment((HistoricalMeanForecaster(),), ExperimentConfig(
            execution=HoldoutSettings(cutoff_for(features), min_train_size=11),
        )).run(features)


def test_holdout_empty_reporting_range_preserves_metadata_without_fitting():
    features = features_sample()
    model = RecordingMean()
    result = HoldoutExperiment((model,), ExperimentConfig(
        execution=HoldoutSettings(cutoff_for(features), min_train_size=5),
        forecast_end=cutoff_for(features),
    )).run(features)

    assert result.records == ()
    assert result.execution is not None
    assert model.training_targets == []


@pytest.mark.parametrize("kwargs", [
    {"min_train_size": 0}, {"min_train_size": True}, {"update_observations": "yes"},
    {"train_end": "2024-01-01"},
])
def test_holdout_settings_validate_their_own_options(kwargs):
    with pytest.raises(ValueError):
        HoldoutSettings(**({"train_end": date(2024, 1, 1)} | kwargs))


def test_holdout_composes_with_the_common_evaluator_and_artifact_writer(tmp_path):
    dates = pd.bdate_range("2024-01-01", periods=60)
    daily = pd.DataFrame({"date": dates, "log_rv": -10 + np.sin(np.arange(60) / 4) / 5})
    source = tmp_path / "daily.parquet"
    daily.to_parquet(source, index=False)
    definition = standard_experiment_definition(executor_factory=HoldoutExperiment)
    config = ExperimentConfig(execution=HoldoutSettings(dates[40].date(), min_train_size=10))
    result = definition.run(daily, config)
    evaluation = definition.evaluator(hac_lags=0).evaluate(result)
    snapshot = DatasetSnapshot.from_frame(source, daily)
    writer = ForecastArtifactWriter(tmp_path / "output")
    paths = writer.write(
        result, evaluation, config=config, dataset=snapshot, specification=definition.specification,
    )
    first = {p.name: p.read_bytes() for p in paths.all()}
    repeated = definition.run(daily, config)
    writer.write(
        repeated, definition.evaluator(hac_lags=0).evaluate(repeated),
        config=config, dataset=snapshot, specification=definition.specification,
    )

    assert result.n_forecasts == 95
    assert len(evaluation.summaries) == 10
    assert {r.n_train for r in result.records} == {19}
    assert {p.name: p.read_bytes() for p in paths.all()} == first
