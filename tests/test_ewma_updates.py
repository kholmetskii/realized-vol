from dataclasses import FrozenInstanceError
from typing import Any, cast

import numpy as np
import pytest

from rvol.domain import FittedForecaster, UpdatablePredictor
from rvol.models import (
    AR1Forecaster,
    EWMAForecaster,
    HARForecaster,
    HistoricalMeanForecaster,
    NaiveForecaster,
)


def test_ewma_update_consumes_new_variance_without_changing_previous_state_or_inputs():
    fitted: UpdatablePredictor = EWMAForecaster(decay=0.5).fit(
        np.empty((2, 0)), np.log([1.0, 4.0]),
    )
    features = np.empty((1, 0))
    target = np.log([9.0])
    original_target = target.copy()
    original_features = features.copy()

    updated = fitted.update(features, target)

    assert isinstance(updated, FittedForecaster)
    assert isinstance(updated, UpdatablePredictor)
    np.testing.assert_allclose(fitted.predict(np.empty((2, 0))), np.log([2.5, 2.5]))
    np.testing.assert_allclose(updated.predict(np.empty((2, 0))), np.log([5.75, 5.75]))
    np.testing.assert_array_equal(target, original_target)
    np.testing.assert_array_equal(features, original_features)
    with pytest.raises(FrozenInstanceError):
        cast(Any, updated).value = 0.0


@pytest.mark.parametrize("decay", [0.5, 0.94, 0.99])
def test_single_and_batch_updates_match_fitting_the_complete_observation_history(decay):
    model = EWMAForecaster(decay=decay)
    target = np.random.default_rng(43).normal(-10.0, 0.5, size=30)
    prefix = model.fit(np.empty((3, 0)), target[:3])
    single = prefix
    for observed in target[3:]:
        single = single.update(np.empty((1, 0)), np.array([observed]))
    batch = prefix
    for start, end in ((3, 10), (10, 14), (14, 30)):
        batch = batch.update(np.empty((end - start, 0)), target[start:end])
    refitted = model.fit(np.empty((len(target), 0)), target)

    np.testing.assert_array_equal(single.predict(np.empty((1, 0))), [refitted.value])
    np.testing.assert_array_equal(batch.predict(np.empty((1, 0))), [refitted.value])
    assert single.decay == batch.decay == decay


def test_an_empty_update_preserves_the_fitted_state():
    fitted = EWMAForecaster().fit(np.empty((1, 0)), np.array([-10.0]))

    assert fitted.update(np.empty((0, 0)), np.empty(0)) is fitted


@pytest.mark.parametrize(
    "features, target, message",
    [
        (np.empty((2, 0)), np.array([-10.0]), "same number of rows"),
        (np.empty((1, 0)), np.array([[-10.0]]), "one-dimensional"),
        (np.empty((1, 0)), np.array([float("nan")]), "finite"),
        (np.empty((1, 0)), np.array([float("inf")]), "finite"),
        (np.zeros((1, 1)), np.array([-10.0]), "exactly 0 columns"),
        (np.empty(0), np.empty(0), "two-dimensional"),
    ],
)
def test_updates_validate_alignment_shapes_and_finite_targets(features, target, message):
    fitted = EWMAForecaster().fit(np.empty((1, 0)), np.array([-10.0]))

    with pytest.raises(ValueError, match=message):
        fitted.update(features, target)


def test_updates_remain_stable_for_widely_separated_log_variances():
    model = EWMAForecaster(decay=0.5)
    target = np.array([-700.0, 700.0, -700.0, 700.0])

    with np.errstate(over="raise", invalid="raise"):
        initial = model.fit(np.empty((1, 0)), target[:1])
        updated = initial.update(np.empty((3, 0)), target[1:])
        refitted = model.fit(np.empty((4, 0)), target)
        variance = np.exp(updated.predict(np.empty((1, 0))))

    assert np.isfinite(variance).all()
    np.testing.assert_array_equal(updated.predict(np.empty((1, 0))), [refitted.value])


@pytest.mark.parametrize(
    "model", [HistoricalMeanForecaster(), NaiveForecaster(), AR1Forecaster(), HARForecaster()],
)
def test_observation_updates_are_optional_for_other_fitted_models(model):
    fitted = model.fit(np.zeros((4, len(model.feature_names))), np.log([1.0, 2.0, 3.0, 4.0]))

    assert isinstance(fitted, FittedForecaster)
    assert not isinstance(fitted, UpdatablePredictor)
