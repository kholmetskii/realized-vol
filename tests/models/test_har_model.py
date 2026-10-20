import numpy as np
import pytest

from rvol.models import (
    HARModel,
)


def test_har_model_recovers_known_coefficients_without_mutating_inputs():
    rng = np.random.default_rng(23)
    features = rng.normal(size=(100, 3))
    target = 0.4 + features @ np.array([0.5, 0.3, 0.1])
    original_features = features.copy()
    original_target = target.copy()

    fitted = HARModel().fit(features, target)
    predicted = fitted.predict(features[:4])

    assert np.isclose(fitted.intercept, 0.4)
    np.testing.assert_allclose(fitted.coefficients, [0.5, 0.3, 0.1])
    np.testing.assert_allclose(predicted, target[:4])
    assert predicted.shape == (4,)
    assert np.isfinite(predicted).all()
    np.testing.assert_array_equal(features, original_features)
    np.testing.assert_array_equal(target, original_target)


def test_har_predicts_mean_log_target_without_correction():
    features = np.zeros((4, 3))
    target = np.log(np.array([1.0, 9.0, 1.0, 9.0]))
    original_target = target.copy()

    fitted = HARModel().fit(features, target)

    np.testing.assert_allclose(fitted.predict(features[:1]), [np.log(3.0)])
    np.testing.assert_array_equal(target, original_target)


def test_har_preserves_regression_effects_with_centered_errors():
    rng = np.random.default_rng(31)
    features = np.repeat(rng.normal(size=(30, 3)), 2, axis=0)
    coefficients = np.array([0.5, 0.3, 0.1])
    target = 0.4 + features @ coefficients + np.tile([-np.log(3), np.log(3)], 30)
    future = rng.normal(size=(5, 3))

    fitted = HARModel().fit(features, target)

    expected = 0.4 + future @ coefficients
    np.testing.assert_allclose(fitted.predict(future), expected)
    assert fitted.intercept == pytest.approx(0.4)
    np.testing.assert_allclose(fitted.coefficients, coefficients)


def test_har_preserves_wide_target_range_without_transformations():
    features = np.zeros((12, 3))
    target = np.array([-700.0] * 11 + [700.0])

    with np.errstate(over="raise", invalid="raise"):
        fitted = HARModel().fit(features, target)
        predicted = fitted.predict(features[:1])

    np.testing.assert_allclose(predicted, [target.mean()])
    assert np.isfinite(predicted).all()
