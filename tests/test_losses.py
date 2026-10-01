import numpy as np
import pytest

from rvol.evaluation.losses import log_squared_error, qlike


def test_perfect_forecasts_have_zero_loss():
    variance = np.array([0.01, 0.02, 0.04])
    log_variance = np.log(variance)

    assert np.allclose(qlike(variance, variance), 0.0)
    assert np.allclose(log_squared_error(log_variance, log_variance), 0.0)


def test_log_squared_error_treats_reciprocal_multipliers_equally():
    actual = np.log(np.array([1.0, 1.0]))
    forecasts = np.log(np.array([2.0, 0.5]))

    losses = log_squared_error(actual, forecasts)

    assert np.isclose(losses[0], losses[1])


def test_qlike_penalizes_underprediction_more_than_reciprocal_overprediction():
    actual = np.array([1.0])

    under = qlike(actual, np.array([0.5]))[0]
    over = qlike(actual, np.array([2.0]))[0]

    assert under > over > 0


def test_qlike_rejects_nonpositive_variances():
    with pytest.raises(ValueError, match="strictly positive"):
        qlike(np.array([0.0]), np.array([1.0]))
