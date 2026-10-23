import numpy as np
import pytest

from rvol.estimators import realized_variance


def test_realized_variance_sums_squared_log_returns():
    # log prices 0, 1, ..., 5: five returns of 1
    assert realized_variance(np.exp(np.arange(6, dtype="float64"))) == pytest.approx(5.0)


def test_realized_variance_is_invariant_to_the_price_unit():
    prices = np.array([1.10, 1.12, 1.09, 1.11])

    assert realized_variance(100 * prices) == pytest.approx(realized_variance(prices))


@pytest.mark.parametrize(
    "prices",
    [np.array([1.0]), np.ones((2, 2)), np.array([1.0, 0.0]), np.array([1.0, np.nan])],
)
def test_invalid_price_paths_are_rejected(prices):
    with pytest.raises(ValueError, match="prices"):
        realized_variance(prices)
