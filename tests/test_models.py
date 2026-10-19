import numpy as np
import pytest

from rvol.models import (
    AR1Model,
    ConstantPredictor,
    EWMAModel,
    EWMAPredictor,
    FeatureColumnPredictor,
    HARModel,
    HistoricalMeanModel,
    LinearPredictor,
    LinearRegressionModel,
    NaiveModel,
)
from rvol.models.contracts import ForecastModel, Predictor


@pytest.mark.parametrize(
    "model, n_features, expected_target",
    [
        pytest.param(HistoricalMeanModel(), 0, np.log(3.0), id="historical_mean"),
        pytest.param(NaiveModel(), 1, np.log(9.0), id="naive"),
        pytest.param(EWMAModel(decay=0.5), 0, 0.625 * np.log(9.0), id="EWMA"),
        pytest.param(AR1Model(), 1, np.log(3.0), id="AR1"),
        pytest.param(
            HARModel(),
            3,
            np.log(3.0),
            id="HAR",
        ),
    ],
)
def test_models_accept_log_targets_without_implicit_transformations(
    model: ForecastModel,
    n_features: int,
    expected_target: float,
):
    training = np.zeros((4, n_features))
    target = np.log([1.0, 9.0, 1.0, 9.0])
    current = np.full((2, n_features), np.log(9.0))
    original_training = training.copy()
    original_target = target.copy()
    original_current = current.copy()

    fitted: Predictor = model.fit(training, target)
    predicted = fitted.predict(current)

    assert predicted.shape == (2,)
    np.testing.assert_allclose(predicted, [expected_target] * 2)
    np.testing.assert_array_equal(training, original_training)
    np.testing.assert_array_equal(target, original_target)
    np.testing.assert_array_equal(current, original_current)


def test_historical_mean_predicts_the_training_target_mean():
    target = np.array([-11.0, -10.0, -9.0])
    fitted = HistoricalMeanModel().fit(np.empty((3, 0)), target)

    predicted = fitted.predict(np.empty((2, 0)))

    np.testing.assert_allclose(predicted, [-10.0, -10.0])


def test_naive_model_returns_the_current_daily_feature():
    training = np.array([[-11.0], [-10.0], [-9.0]])
    fitted = NaiveModel().fit(training, np.array([-10.0, -9.0, -8.0]))
    future = np.array([[-8.5], [-8.0]])

    predicted = fitted.predict(future)

    np.testing.assert_array_equal(predicted, future[:, 0])
    assert not np.shares_memory(predicted, future)


def test_ewma_model_applies_fixed_decay_to_supplied_targets():
    target = np.array([1.0, 4.0, 9.0])
    fitted = EWMAModel(decay=0.5).fit(np.empty((3, 0)), target)

    predicted = fitted.predict(np.empty((2, 0)))

    np.testing.assert_allclose(predicted, [5.75, 5.75])


@pytest.mark.parametrize("decay", [0.0, 1.0, -0.1, float("nan")])
def test_ewma_model_rejects_invalid_decay(decay):
    with pytest.raises(ValueError, match="decay"):
        EWMAModel(decay=decay)


def test_ar1_recovers_known_coefficients():
    daily = np.linspace(-12.0, -8.0, 30).reshape(-1, 1)
    target = 0.7 + 0.6 * daily[:, 0]

    fitted = AR1Model().fit(daily, target)

    assert isinstance(fitted, LinearPredictor)
    assert np.isclose(fitted.intercept, 0.7)
    np.testing.assert_allclose(fitted.coefficients, [0.6])


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


def test_fitting_creates_independent_predictor_state():
    daily = np.arange(5, dtype="float64").reshape(-1, 1)
    model = AR1Model()
    fitted = model.fit(daily, daily[:, 0])
    other = model.fit(daily, daily[:, 0] + 2.0)

    assert fitted is not other
    np.testing.assert_allclose(fitted.predict(daily), daily[:, 0], atol=1e-12)
    np.testing.assert_allclose(other.predict(daily), daily[:, 0] + 2.0, atol=1e-12)


@pytest.mark.parametrize(
    "predictor, features, expected_state, predicted",
    [
        pytest.param(
            LinearPredictor(1.0, (2.0, 3.0)),
            np.array([[4.0, 5.0]]),
            {"intercept": 1.0, "coefficients": (2.0, 3.0)},
            [24.0],
            id="linear",
        ),
        pytest.param(
            ConstantPredictor(7.0, n_features=2),
            np.array([[4.0, 5.0]]),
            {"value": 7.0, "n_features": 2},
            [7.0],
            id="constant",
        ),
        pytest.param(
            EWMAPredictor(8.0, decay=0.5),
            np.empty((2, 0)),
            {"value": 8.0, "decay": 0.5},
            [8.0, 8.0],
            id="EWMA",
        ),
        pytest.param(
            FeatureColumnPredictor(n_features=2, column=1, offset=0.5),
            np.array([[4.0, 5.0]]),
            {"n_features": 2, "column": 1, "offset": 0.5},
            [5.5],
            id="feature_column",
        ),
    ],
)
def test_predictor_state_can_be_read_but_not_replaced(
    predictor, features, expected_state, predicted
):
    for attribute, expected in expected_state.items():
        assert getattr(predictor, attribute) == expected
        with pytest.raises(AttributeError):
            setattr(predictor, attribute, float("nan"))
        assert getattr(predictor, attribute) == expected
    np.testing.assert_array_equal(predictor.predict(features), predicted)


def test_linear_predictor_copies_coefficients_from_mutable_input():
    coefficients = [2.0, 3.0]
    predictor = LinearPredictor(intercept=1.0, coefficients=coefficients)

    coefficients[0] = 100.0
    coefficients.append(4.0)

    assert predictor.coefficients == (2.0, 3.0)
    np.testing.assert_array_equal(predictor.predict(np.array([[4.0, 5.0]])), [24.0])


@pytest.mark.parametrize(
    "model, attribute, expected, features, target, future, prediction",
    [
        pytest.param(
            EWMAModel(decay=0.5),
            "decay",
            0.5,
            np.empty((2, 0)),
            np.array([2.0, 6.0]),
            np.empty((1, 0)),
            [4.0],
            id="EWMA",
        ),
        pytest.param(
            NaiveModel(offset=2.0),
            "offset",
            2.0,
            np.array([[1.0], [2.0]]),
            np.array([2.0, 3.0]),
            np.array([[10.0]]),
            [12.0],
            id="naive",
        ),
    ],
)
def test_validated_model_parameters_cannot_be_replaced(
    model, attribute, expected, features, target, future, prediction
):
    assert getattr(model, attribute) == expected
    with pytest.raises(AttributeError):
        setattr(model, attribute, float("nan"))

    np.testing.assert_array_equal(model.fit(features, target).predict(future), prediction)


def test_models_reject_misaligned_training_arrays():
    with pytest.raises(ValueError, match="same number of rows"):
        HARModel().fit(np.ones((5, 3)), np.ones(4))


@pytest.mark.parametrize("n_features", [0, 2, 4, 7])
def test_linear_regression_accepts_arbitrary_prepared_feature_counts(n_features):
    features = np.random.default_rng(24).normal(size=(50, n_features))
    coefficients = np.arange(n_features) + 0.5
    target = 2.5 + features @ coefficients

    fitted = LinearRegressionModel().fit(features, target)

    assert fitted.intercept == pytest.approx(2.5)
    np.testing.assert_allclose(fitted.coefficients, coefficients, atol=1e-12)
    np.testing.assert_allclose(fitted.predict(features), target, atol=1e-12)


@pytest.mark.parametrize(
    "model, width, expected",
    [
        (AR1Model(), 0, 1),
        (AR1Model(), 2, 1),
        (AR1Model(), 3, 1),
        (HARModel(), 0, 3),
        (HARModel(), 1, 3),
        (HARModel(), 2, 3),
        (HARModel(), 4, 3),
    ],
)
def test_named_autoregressions_reject_incompatible_feature_counts(model, width, expected):
    with pytest.raises(ValueError, match=f"exactly {expected} columns"):
        model.fit(np.zeros((8, width)), np.arange(8, dtype=float))


@pytest.mark.parametrize("model, width", [(AR1Model(), 1), (HARModel(), 3)])
def test_named_autoregressions_require_enough_observations_for_the_coefficients(model, width):
    with pytest.raises(ValueError, match=f"at least {width + 1} observations"):
        model.fit(np.zeros((width, width)), np.arange(width, dtype=float))
