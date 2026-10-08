from datetime import date
from typing import cast

import numpy as np
import pandas as pd
import pytest

from rvol.application import WalkForwardExperiment
from rvol.domain import EveryNSessions, ExperimentConfig, RollingWindow, WalkForwardStrategy
from rvol.domain.contracts import FloatArray
from rvol.models import (
    AR1Forecaster,
    EWMAForecaster,
    HARForecaster,
    HistoricalMeanForecaster,
    NaiveForecaster,
)
from rvol.models.predictors import ConstantPredictor


def feature_sample(n: int = 45, seed: int = 19) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n + 1)
    matrix = rng.normal(loc=-10.0, scale=0.5, size=(n, 3))
    target = -0.4 + matrix @ np.array([0.5, 0.3, 0.15])
    target += rng.normal(scale=0.1, size=n)
    return pd.DataFrame({
        "origin_date": dates[:-1],
        "target_date": dates[1:],
        "rv_daily": matrix[:, 0],
        "rv_weekly": matrix[:, 1],
        "rv_monthly": matrix[:, 2],
        "target": target,
    })


def all_models():
    return (
        HistoricalMeanForecaster(),
        NaiveForecaster(),
        EWMAForecaster(),
        AR1Forecaster(),
        HARForecaster(),
    )


def prediction_map(result, target_date):
    return {
        record.model: record.predicted_log_rv
        for record in result.records
        if record.target_date == target_date
    }


def date_at(features: pd.DataFrame, position: int, column: str = "target_date") -> date:
    return cast(pd.Timestamp, features.loc[position, column]).date()


def test_experiment_runs_every_model_on_identical_expanding_windows():
    experiment = WalkForwardExperiment(
        models=all_models(),
        config=ExperimentConfig(min_train_size=20),
    )

    result = experiment.run(feature_sample())

    assert result.models == ("AR1", "EWMA", "HAR", "historical_mean", "naive")
    assert result.n_forecasts == 125
    for model in result.models:
        records = [record for record in result.records if record.model == model]
        assert len(records) == 25
        assert [record.n_train for record in records] == list(range(20, 45))
        assert [record.target_date for record in records] == sorted(
            record.target_date for record in records
        )


def test_date_bounds_are_inclusive_and_preserve_prior_training_data():
    features = feature_sample()
    start = cast(pd.Timestamp, features.loc[30, "target_date"]).date()
    end = cast(pd.Timestamp, features.loc[35, "target_date"]).date()
    experiment = WalkForwardExperiment(
        models=all_models(),
        config=ExperimentConfig(
            min_train_size=20,
            forecast_start=start,
            forecast_end=end,
        ),
    )

    result = experiment.run(features)

    target_dates = sorted({record.target_date for record in result.records})
    assert target_dates == list(pd.bdate_range(start, end).date)
    assert {record.n_train for record in result.records if record.target_date == start} == {30}


def test_current_target_cannot_change_its_own_prediction_for_any_model():
    original = feature_sample()
    changed = original.copy()
    changed.loc[20, "target"] = cast(float, changed.loc[20, "target"]) + 100.0
    experiment = WalkForwardExperiment(
        models=all_models(),
        config=ExperimentConfig(min_train_size=20),
    )
    target_day = cast(pd.Timestamp, original.loc[20, "target_date"]).date()

    before = experiment.run(original)
    after = experiment.run(changed)

    assert prediction_map(before, target_day) == prediction_map(after, target_day)
    before_actual = {
        record.actual_log_rv for record in before.records if record.target_date == target_day
    }
    after_actual = {
        record.actual_log_rv for record in after.records if record.target_date == target_day
    }
    assert before_actual != after_actual


def test_experiment_rejects_duplicate_model_names_and_missing_features():
    with pytest.raises(ValueError, match="names must be unique"):
        WalkForwardExperiment(
            models=(NaiveForecaster(), NaiveForecaster()),
            config=ExperimentConfig(),
        )

    experiment = WalkForwardExperiment(
        models=(HARForecaster(),),
        config=ExperimentConfig(min_train_size=20),
    )
    with pytest.raises(ValueError, match="rv_monthly"):
        experiment.run(feature_sample().drop(columns="rv_monthly"))


def test_rolling_refits_use_exact_windows_and_record_the_last_actual_fit():
    class RecordingMean(HistoricalMeanForecaster):
        def __init__(self) -> None:
            self.training_targets: list[FloatArray] = []

        def fit(self, features: FloatArray, target: FloatArray) -> ConstantPredictor:
            self.training_targets.append(target.copy())
            return super().fit(features, target)

    features = feature_sample(18)
    model = RecordingMean()
    result = WalkForwardExperiment(
        models=(model,),
        config=ExperimentConfig(
            min_train_size=5,
            strategy=WalkForwardStrategy(RollingWindow(8), EveryNSessions(3)),
        ),
    ).run(features)

    fit_positions = (5, 8, 11, 14, 17)
    assert len(model.training_targets) == len(fit_positions)
    for position, observed in zip(fit_positions, model.training_targets, strict=True):
        np.testing.assert_array_equal(
            observed, features["target"].iloc[max(0, position - 8):position],
        )
    assert result.strategy_anchor == date_at(features, 5, "origin_date")
    for position, record in enumerate(result.records, start=5):
        fitted_at = 5 + ((position - 5) // 3) * 3
        first_training = max(0, fitted_at - 8)
        assert record.n_train == fitted_at - first_training
        assert record.fit_date == date_at(features, fitted_at, "origin_date")
        assert record.train_start_date == date_at(features, first_training)
        assert record.train_end_date == date_at(features, fitted_at - 1)
        assert record.predicted_log_rv == pytest.approx(
            features["target"].iloc[first_training:fitted_at].mean(),
        )


@pytest.mark.parametrize("model", [HARForecaster(), AR1Forecaster(), NaiveForecaster()])
def test_cached_parameters_still_receive_fresh_features_between_refits(model):
    features = feature_sample(12)
    result = WalkForwardExperiment(
        models=(model,),
        config=ExperimentConfig(
            min_train_size=5, strategy=WalkForwardStrategy(retrain=EveryNSessions(4)),
        ),
    ).run(features)
    columns = list(model.feature_names)
    fitted = model.fit(
        features.loc[:4, columns].to_numpy(), features.loc[:4, "target"].to_numpy(),
    )

    np.testing.assert_array_equal(
        [record.predicted_log_rv for record in result.records[:4]],
        [
            fitted.predict(features.loc[[position], columns].to_numpy())[0]
            for position in range(5, 9)
        ],
    )
    assert {record.n_train for record in result.records[:4]} == {5}
    assert {record.fit_date for record in result.records[:4]} == {
        date_at(features, 5, "origin_date"),
    }


def test_ewma_updates_between_expanding_refits_match_daily_fitting():
    features = feature_sample()
    daily = WalkForwardExperiment(
        (EWMAForecaster(),), ExperimentConfig(min_train_size=5),
    ).run(features)
    periodic = WalkForwardExperiment(
        (EWMAForecaster(),),
        ExperimentConfig(min_train_size=5, strategy=WalkForwardStrategy(retrain=EveryNSessions(5))),
    ).run(features)

    np.testing.assert_array_equal(
        [record.predicted_log_rv for record in periodic.records],
        [record.predicted_log_rv for record in daily.records],
    )
    assert periodic.records[4].n_train == 5
    assert periodic.records[5].n_train == 10


def test_rolling_ewma_resets_on_refit_and_updates_between_refits():
    features = feature_sample(18)
    result = WalkForwardExperiment(
        (EWMAForecaster(decay=0.5),),
        ExperimentConfig(
            min_train_size=5, strategy=WalkForwardStrategy(RollingWindow(8), EveryNSessions(3)),
        ),
    ).run(features)
    variance = 0.0
    expected = []
    for position in range(5, 18):
        if (position - 5) % 3 == 0:
            training = np.exp(features["target"].iloc[max(0, position - 8):position].to_numpy())
            variance = float(training[0])
            for observed in training[1:]:
                variance = 0.5 * variance + 0.5 * observed
        else:
            observed = np.exp(cast(float, features.loc[position - 1, "target"]))
            variance = 0.5 * variance + 0.5 * observed
        expected.append(variance)

    np.testing.assert_allclose([record.predicted_rv for record in result.records], expected)


def test_updates_consume_only_newly_available_labels_including_empty_and_multirow_batches():
    features = pd.DataFrame({
        "origin_date": pd.to_datetime(["2024-01-01", "2024-01-04", "2024-01-05", "2024-01-10"]),
        "target_date": pd.to_datetime(["2024-01-03", "2024-01-07", "2024-01-09", "2024-01-11"]),
        "target": np.log([1.0, 4.0, 9.0, 16.0]),
    })
    result = WalkForwardExperiment(
        (EWMAForecaster(decay=0.5),),
        ExperimentConfig(min_train_size=1, strategy=WalkForwardStrategy(retrain=EveryNSessions(5))),
    ).run(features)

    np.testing.assert_allclose([record.predicted_rv for record in result.records], [1.0, 1.0, 5.75])
    assert {record.n_train for record in result.records} == {1}


def test_reporting_bounds_preserve_the_global_schedule_and_model_state():
    features = feature_sample()
    strategy = WalkForwardStrategy(RollingWindow(12), EveryNSessions(5))
    full = WalkForwardExperiment(
        all_models(), ExperimentConfig(min_train_size=5, strategy=strategy),
    )
    complete = full.run(features)
    start = date_at(features, 13)
    end = date_at(features, 23)
    bounded = WalkForwardExperiment(
        all_models(), ExperimentConfig(5, start, end, strategy),
    ).run(features)

    assert bounded.records == tuple(
        record for record in complete.records if start <= record.target_date <= end
    )
    assert bounded.strategy_anchor == complete.strategy_anchor
    assert bounded.records[0].fit_date is not None
    assert bounded.records[0].fit_date < bounded.records[0].origin_date
    assert full.run(features) == complete


@pytest.mark.parametrize("position", [6, 8])
def test_current_outcome_cannot_enter_either_an_update_or_a_refit(position):
    original = feature_sample(18)
    changed = original.copy()
    changed.loc[position, "target"] += 1.0
    experiment = WalkForwardExperiment(
        all_models(),
        ExperimentConfig(min_train_size=5, strategy=WalkForwardStrategy(retrain=EveryNSessions(3))),
    )
    target_day = date_at(original, position)

    assert prediction_map(experiment.run(original), target_day) == prediction_map(
        experiment.run(changed), target_day,
    )


def test_first_eligible_origin_always_initializes_the_models():
    class NoScheduledRefits:
        def should_refit(self, eligible_origin_index: int) -> bool:
            return False

    result = WalkForwardExperiment(
        (HistoricalMeanForecaster(),),
        ExperimentConfig(
            min_train_size=5, strategy=WalkForwardStrategy(retrain=NoScheduledRefits()),
        ),
    ).run(feature_sample(12))

    assert len(result.records) == 7
    assert {record.n_train for record in result.records} == {5}


def test_origins_must_advance_chronologically_for_cached_state():
    features = feature_sample()
    features.loc[6, "origin_date"] = features.loc[4, "origin_date"]

    with pytest.raises(ValueError, match="origin dates must be strictly increasing"):
        WalkForwardExperiment(all_models(), ExperimentConfig(min_train_size=5)).run(features)


@pytest.mark.parametrize("selected", [slice(-1, 3), slice(0, 1000), slice(0, 0, -1)])
def test_custom_windows_cannot_select_future_or_reversed_history(selected):
    class InvalidWindow:
        max_size: int | None = None

        def select(self, n_available: int) -> slice:
            return selected

    config = ExperimentConfig(min_train_size=5, strategy=WalkForwardStrategy(InvalidWindow()))
    with pytest.raises(ValueError, match="chronological, available observations"):
        WalkForwardExperiment(all_models(), config).run(feature_sample())
