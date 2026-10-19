import numpy as np
import pytest

from rvol.estimators.variance import realized_variance


@pytest.mark.parametrize(
    "estimator, options, message",
    [
        (realized_variance, {"step": True}, "step"),
        (realized_variance, {"step": 1.5}, "step"),
        (realized_variance, {"step": 0}, "step"),
        (realized_variance, {"step": np.bool_(True)}, "step"),
        (realized_variance, {"offset": True}, "offset"),
        (realized_variance, {"offset": 0.5}, "offset"),
        (realized_variance, {"offset": -1}, "offset"),
        (realized_variance, {"step": 2, "offset": 2}, "offset"),
    ],
)
def test_realized_variance_rejects_invalid_sampling_parameters(estimator, options, message):
    prices = np.exp(np.arange(6, dtype="float64"))

    with pytest.raises(ValueError, match=message):
        estimator(prices, **options)


@pytest.mark.parametrize("integer", [int, np.int32, np.int64])
def test_realized_variance_accepts_python_and_numpy_integer_grids(integer):
    prices = np.exp(np.arange(6, dtype="float64"))

    assert realized_variance(prices, step=integer(2), offset=integer(0)) == pytest.approx(8.0)
