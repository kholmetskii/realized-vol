import numpy as np
import pytest

from rvol.models import ExponentialSmoothing, ForecastModel, LinearRegression


def regression_data(n=40, seed=4):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n, 3))
    target = 2.0 + features @ np.array([0.5, -1.0, 3.0]) + rng.normal(0, 0.1, n)
    return features, target


def test_least_squares_recovers_an_exact_linear_relation():
    rng = np.random.default_rng(4)
    features = rng.normal(size=(30, 3))
    target = 2.0 + features @ np.array([0.5, -1.0, 3.0])

    model = LinearRegression().fit(features, target)

    assert model.intercept == pytest.approx(2.0)
    np.testing.assert_allclose(model.coefficients, [0.5, -1.0, 3.0])
    np.testing.assert_allclose(model.predict(features[:2]), target[:2])


def test_without_regressors_the_regression_forecasts_the_training_mean():
    model = LinearRegression().fit(np.empty((3, 0)), np.array([1.0, 2.0, 6.0]))

    np.testing.assert_allclose(model.predict(np.empty((2, 0))), [3.0, 3.0])


def test_ridge_matches_its_closed_form_with_an_unpenalized_intercept():
    features, target = regression_data()
    ridge = 5.0

    model = LinearRegression(ridge=ridge).fit(features, target)

    x = features - features.mean(axis=0)
    y = target - target.mean()
    expected = np.linalg.solve(x.T @ x + ridge * np.eye(3), x.T @ y)
    np.testing.assert_allclose(model.coefficients, expected)
    assert model.intercept == pytest.approx(target.mean() - features.mean(axis=0) @ expected)


def test_a_vanishing_penalty_approaches_least_squares():
    features, target = regression_data()

    ols = LinearRegression().fit(features, target)
    ridge = LinearRegression(ridge=1e-10).fit(features, target)

    np.testing.assert_allclose(ridge.coefficients, ols.coefficients, rtol=1e-8)
    assert ridge.intercept == pytest.approx(ols.intercept)


def test_a_large_penalty_shrinks_the_forecast_to_the_training_mean():
    features, target = regression_data()

    model = LinearRegression(ridge=1e12).fit(features, target)

    np.testing.assert_allclose(model.coefficients, 0, atol=1e-9)
    np.testing.assert_allclose(model.predict(features[:3]), target.mean(), atol=1e-8)


def test_ridge_without_regressors_is_still_the_mean():
    model = LinearRegression(ridge=1.0).fit(np.empty((3, 0)), np.array([1.0, 2.0, 6.0]))

    np.testing.assert_allclose(model.predict(np.empty((1, 0))), [3.0])


@pytest.mark.parametrize("ridge", [-1.0, float("nan"), float("inf")])
def test_invalid_penalties_are_rejected(ridge):
    with pytest.raises(ValueError, match="ridge"):
        LinearRegression(ridge=ridge)


def test_regression_needs_more_observations_than_regressors():
    with pytest.raises(ValueError, match="more observations"):
        LinearRegression().fit(np.ones((3, 3)), np.ones(3))


def test_smoothing_applies_the_recursion_from_the_first_target():
    # s1 = -3, s2 = 0.5 * -3 + 0.5 * 0 = -1.5, s3 = 0.5 * -1.5 + 0.5 * 9 = 3.75
    model = ExponentialSmoothing(decay=0.5).fit(np.empty((3, 0)), np.array([-3.0, 0.0, 9.0]))

    assert model.value == pytest.approx(3.75)
    np.testing.assert_allclose(model.predict(np.empty((2, 0))), [3.75, 3.75])


def test_smoothing_puts_weight_one_minus_decay_on_the_latest_target():
    model = ExponentialSmoothing(decay=0.94).fit(np.empty((2, 0)), np.array([1.0, 9.0]))

    assert model.value == pytest.approx(0.94 * 1.0 + 0.06 * 9.0)


def test_zero_decay_is_the_naive_forecast_of_the_last_target():
    model = ExponentialSmoothing(decay=0.0).fit(np.empty((3, 0)), np.array([4.0, -2.0, 7.5]))

    np.testing.assert_array_equal(model.predict(np.empty((2, 0))), [7.5, 7.5])


@pytest.mark.parametrize("decay", [1.0, -0.1, float("nan")])
def test_smoothing_rejects_a_decay_outside_zero_to_one(decay):
    with pytest.raises(ValueError, match="decay"):
        ExponentialSmoothing(decay=decay)


def test_refitting_replaces_the_previous_state():
    model = LinearRegression()
    model.fit(np.empty((2, 0)), np.array([1.0, 1.0]))
    model.fit(np.empty((2, 0)), np.array([5.0, 5.0]))

    np.testing.assert_allclose(model.predict(np.empty((1, 0))), [5.0])


def test_a_model_names_a_rule_applied_to_chosen_regressors():
    model = ForecastModel("HAR", LinearRegression(), ("daily", "weekly", "monthly"))

    assert model.name == "HAR"
    assert isinstance(model.rule, LinearRegression)
    assert model.regressors == ("daily", "weekly", "monthly")
    assert ForecastModel("mean", LinearRegression()).regressors == ()


@pytest.mark.parametrize(
    "name, regressors",
    [("", ()), (" ", ()), ("m", ("a", "a")), ("m", ("",)), ("m", ["a"]), ("m", "ab")],
)
def test_invalid_model_definitions_are_rejected(name, regressors):
    with pytest.raises(ValueError):
        ForecastModel(name, LinearRegression(), regressors)
