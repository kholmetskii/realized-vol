import numpy as np
import pandas as pd
import pytest

from rvol.application import WalkForwardExperiment
from rvol.domain import ExperimentConfig
from rvol.models import (
    AR1Forecaster,
    HARForecaster,
    HistoricalMeanForecaster,
    NaiveForecaster,
)


def feature_sample(n: int = 45, seed: int = 19) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n + 1)
    matrix = rng.normal(loc=-10.0, scale=0.5, size=(n, 3))
    target = -0.4 + matrix @ np.array([0.5, 0.3, 0.15])
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
        AR1Forecaster(),
        HARForecaster(),
    )


def prediction_map(result, target_date):
    return {
        record.model: record.predicted_log_rv
        for record in result.records
        if record.target_date == target_date
    }


def test_experiment_runs_every_model_on_identical_expanding_windows():
    experiment = WalkForwardExperiment(
        models=all_models(),
        config=ExperimentConfig(min_train_size=20),
    )

    result = experiment.run(feature_sample())

    assert result.models == ("AR1", "HAR", "historical_mean", "naive")
    assert result.n_forecasts == 100
    for model in result.models:
        records = [record for record in result.records if record.model == model]
        assert len(records) == 25
        assert [record.n_train for record in records] == list(range(20, 45))
        assert [record.target_date for record in records] == sorted(
            record.target_date for record in records
        )


def test_date_bounds_are_inclusive_and_preserve_prior_training_data():
    features = feature_sample()
    start = features.loc[30, "target_date"].date()
    end = features.loc[35, "target_date"].date()
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
    changed.loc[20, "target"] += 100.0
    experiment = WalkForwardExperiment(
        models=all_models(),
        config=ExperimentConfig(min_train_size=20),
    )
    target_day = original.loc[20, "target_date"].date()

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
