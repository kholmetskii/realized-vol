from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from rvol.domain import FittedForecaster, Forecaster
from rvol.models import (
    AR1Forecaster,
    EWMAForecaster,
    HARForecaster,
    HistoricalMeanForecaster,
    NaiveForecaster,
)
from rvol.models.linear import LinearPredictor


def assign_attribute(instance: object, name: str, value: object) -> None:
    setattr(instance, name, value)


@pytest.mark.parametrize(
    "model, expected_variance",
    [
        pytest.param(HistoricalMeanForecaster(), 3.0, id="historical_mean"),
        pytest.param(NaiveForecaster(), 9.0, id="naive"),
        pytest.param(EWMAForecaster(decay=0.5), 6.0, id="EWMA"),
        pytest.param(AR1Forecaster(), 3.0, id="AR1"),
        pytest.param(HARForecaster(), 5.0, id="HAR"),
    ],
)
def test_forecasters_share_the_log_variance_fit_predict_contract(
    model: Forecaster, expected_variance: float,
):
    training = np.zeros((4, len(model.feature_names)))
    target = np.log([1.0, 9.0, 1.0, 9.0])
    current = np.full((2, len(model.feature_names)), np.log(9.0))
    original_training = training.copy()
    original_target = target.copy()
    original_current = current.copy()

    fitted = model.fit(training, target)
    predicted = fitted.predict(current)

    assert isinstance(model, Forecaster)
    assert isinstance(fitted, FittedForecaster)
    assert predicted.shape == (2,)
    np.testing.assert_allclose(predicted, np.log([expected_variance] * 2))
    np.testing.assert_array_equal(training, original_training)
    np.testing.assert_array_equal(target, original_target)
    np.testing.assert_array_equal(current, original_current)


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


def test_ewma_forecaster_applies_fixed_decay_on_the_variance_scale():
    target = np.log(np.array([1.0, 4.0, 9.0]))
    fitted = EWMAForecaster(decay=0.5).fit(np.empty((3, 0)), target)

    predicted = fitted.predict(np.empty((2, 0)))

    expected_variance = 0.5 * (0.5 * 1.0 + 0.5 * 4.0) + 0.5 * 9.0
    np.testing.assert_allclose(predicted, np.log([expected_variance, expected_variance]))


@pytest.mark.parametrize("decay", [0.0, 1.0, -0.1, float("nan")])
def test_ewma_forecaster_rejects_invalid_decay(decay):
    with pytest.raises(ValueError, match="decay"):
        EWMAForecaster(decay=decay)


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
    np.testing.assert_allclose(predicted, target[:4])
    assert predicted.shape == (4,)
    assert np.isfinite(predicted).all()
    np.testing.assert_array_equal(features, original_features)
    np.testing.assert_array_equal(target, original_target)


def test_har_recovers_arithmetic_variance_mean_from_log_predictions():
    features = np.zeros((4, 3))
    target = np.log(np.array([1.0, 9.0, 1.0, 9.0]))
    original_target = target.copy()

    fitted = HARForecaster().fit(features, target)

    np.testing.assert_allclose(np.exp(fitted.predict(features[:1])), [5.0])
    np.testing.assert_array_equal(target, original_target)


def test_har_preserves_regression_effects_with_multiplicative_errors():
    rng = np.random.default_rng(31)
    features = np.repeat(rng.normal(size=(30, 3)), 2, axis=0)
    coefficients = np.array([0.5, 0.3, 0.1])
    target = 0.4 + features @ coefficients + np.tile([-np.log(3), np.log(3)], 30)
    future = rng.normal(size=(5, 3))

    fitted = HARForecaster().fit(features, target)

    # At each feature vector the two equally likely variances are one third
    # and three times exp(0.4 + x @ coefficients), with arithmetic mean 5/3.
    expected = (5 / 3) * np.exp(0.4 + future @ coefficients)
    np.testing.assert_allclose(np.exp(fitted.predict(future)), expected)
    np.testing.assert_allclose(fitted.coefficients, coefficients)


def test_har_stays_finite_when_exponentiating_residuals_would_overflow():
    features = np.zeros((12, 3))
    target = np.array([-700.0] * 11 + [700.0])

    with np.errstate(over="raise", invalid="raise"):
        fitted = HARForecaster().fit(features, target)
        predicted = fitted.predict(features[:1])
        variance = np.exp(predicted)

    np.testing.assert_allclose(predicted, [700 - np.log(12)])
    assert np.isfinite(variance).all()


def test_fitted_linear_models_are_immutable():
    daily = np.arange(5, dtype="float64").reshape(-1, 1)
    fitted = AR1Forecaster().fit(daily, daily[:, 0])

    with pytest.raises(FrozenInstanceError):
        assign_attribute(fitted, "intercept", 2.0)


def test_forecasters_reject_misaligned_training_arrays():
    with pytest.raises(ValueError, match="same number of rows"):
        HARForecaster().fit(np.ones((5, 3)), np.ones(4))
