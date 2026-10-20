import numpy as np
import pytest

from rvol.evaluation.volatility import QLikeMetric, qlike


def test_perfect_forecasts_have_zero_qlike():
    variance = np.array([0.01, 0.02, 0.04])

    assert np.allclose(qlike(variance, variance), 0.0)


def test_qlike_penalizes_underprediction_more_than_reciprocal_overprediction():
    actual = np.array([1.0])

    under = qlike(actual, np.array([0.5]))[0]
    over = qlike(actual, np.array([2.0]))[0]

    assert under > over > 0


def test_qlike_rejects_nonpositive_variances():
    with pytest.raises(ValueError, match="strictly positive"):
        qlike(np.array([0.0]), np.array([1.0]))


def test_qlike_metric_does_not_transform_its_inputs():
    actual = np.array([0.5, 1.0, 2.0])
    predicted = np.array([1.0, 1.0, 1.0])

    assert np.allclose(QLikeMetric().losses(actual, predicted), qlike(actual, predicted))
