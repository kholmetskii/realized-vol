import numpy as np
import pytest

from rvol.models import (
    AR1Model,
    ConstantPredictor,
    EWMAModel,
    EWMAPredictor,
    FeatureColumnPredictor,
    ForecastModel,
    HARModel,
    HistoricalMeanModel,
    LinearPredictor,
    LinearRegressionModel,
    NaiveModel,
    Predictor,
    UpdatablePredictor,
)


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


@pytest.mark.parametrize(
    "implementation, interface",
    [
        (AR1Model, ForecastModel),
        (EWMAModel, ForecastModel),
        (HARModel, ForecastModel),
        (HistoricalMeanModel, ForecastModel),
        (LinearRegressionModel, ForecastModel),
        (NaiveModel, ForecastModel),
        (ConstantPredictor, Predictor),
        (EWMAPredictor, UpdatablePredictor),
        (FeatureColumnPredictor, Predictor),
        (LinearPredictor, Predictor),
    ],
)
def test_implementations_declare_the_interface_they_implement(implementation, interface):
    assert interface in implementation.__mro__


def test_linear_regression_name_is_set_once_and_read_only():
    model = LinearRegressionModel(name="custom")

    assert model.name == "custom"
    with pytest.raises(AttributeError):
        model.name = "other"  # type: ignore[misc]
