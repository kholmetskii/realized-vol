import numpy as np

from rvol.evaluation import squared_error


def test_perfect_forecasts_have_zero_squared_error():
    log_variance = np.log(np.array([0.01, 0.02, 0.04]))

    assert np.allclose(squared_error(log_variance, log_variance), 0.0)


def test_squared_error_on_logged_values_treats_reciprocal_multipliers_equally():
    actual = np.log(np.array([1.0, 1.0]))
    forecasts = np.log(np.array([2.0, 0.5]))

    losses = squared_error(actual, forecasts)

    assert np.isclose(losses[0], losses[1])
