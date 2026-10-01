from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from rvol.domain import Forecaster
from rvol.models import (
    AR1Forecaster,
    HARForecaster,
    HistoricalMeanForecaster,
    NaiveForecaster,
)
from rvol.models.base import LinearPredictor


def assign_attribute(instance: object, name: str, value: object) -> None:
    setattr(instance, name, value)


def test_all_forecasters_implement_the_common_contract():
    models = [
        HistoricalMeanForecaster(),
        NaiveForecaster(),
        AR1Forecaster(),
        HARForecaster(),
    ]

    assert all(isinstance(model, Forecaster) for model in models)
    assert [model.name for model in models] == ["historical_mean", "naive", "AR1", "HAR"]


def test_historical_mean_predicts_the_training_target_mean():
    target = np.array([-11.0, -10.0, -9.0])
    fitted = HistoricalMeanForecaster().fit(np.empty((3, 0)), target)

    predicted = fitted.predict(np.empty((2, 0)))

    np.testing.assert_allclose(predicted, [-10.0, -10.0])


def test_naive_forecaster_returns_the_current_daily_feature():
    training = np.array([[-11.0], [-10.0], [-9.0]])
    fitted = NaiveForecaster().fit(training, np.array([-10.0, -9.0, -8.0]))
    future = np.array([[-8.5], [-8.0]])

    predicted = fitted.predict(future)

    np.testing.assert_array_equal(predicted, future[:, 0])
    assert not np.shares_memory(predicted, future)


def test_ar1_recovers_known_coefficients():
    daily = np.linspace(-12.0, -8.0, 30).reshape(-1, 1)
    target = 0.7 + 0.6 * daily[:, 0]

    fitted = AR1Forecaster().fit(daily, target)

    assert isinstance(fitted, LinearPredictor)
    assert np.isclose(fitted.intercept, 0.7)
    np.testing.assert_allclose(fitted.coefficients, [0.6])


def test_har_forecaster_recovers_known_coefficients_without_mutating_inputs():
    rng = np.random.default_rng(23)
    features = rng.normal(size=(100, 3))
    target = 0.4 + features @ np.array([0.5, 0.3, 0.1])
    original_features = features.copy()
    original_target = target.copy()

    fitted = HARForecaster().fit(features, target)
    predicted = fitted.predict(features[:4])

    assert np.isclose(fitted.intercept, 0.4)
    np.testing.assert_allclose(fitted.coefficients, [0.5, 0.3, 0.1])
    assert predicted.shape == (4,)
    assert np.isfinite(predicted).all()
    np.testing.assert_array_equal(features, original_features)
    np.testing.assert_array_equal(target, original_target)


def test_fitted_linear_models_are_immutable():
    daily = np.arange(5, dtype="float64").reshape(-1, 1)
    fitted = AR1Forecaster().fit(daily, daily[:, 0])

    with pytest.raises(FrozenInstanceError):
        assign_attribute(fitted, "intercept", 2.0)


def test_forecasters_reject_misaligned_training_arrays():
    with pytest.raises(ValueError, match="same number of rows"):
        HARForecaster().fit(np.ones((5, 3)), np.ones(4))
