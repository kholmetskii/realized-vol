import numpy as np
import pytest

from rvol.simulation.noise import add_bid_ask_bounce, add_iid_log_noise


def test_zero_noise_preserves_prices_without_aliasing():
    prices = np.array([1.0, 1.01, 1.02])
    observed = add_iid_log_noise(prices, 0.0)

    assert np.array_equal(observed, prices)
    assert observed is not prices


def test_bid_ask_bounce_places_every_trade_on_one_quote_side():
    efficient = np.linspace(1.0, 1.01, 200)
    half_spread = 2e-4
    observed = add_bid_ask_bounce(
        efficient, half_spread, rng=np.random.default_rng(10)
    )

    displacement = np.log(observed) - np.log(efficient)
    assert np.allclose(np.abs(displacement), half_spread)
    assert set(np.sign(displacement)) == {-1.0, 1.0}


@pytest.mark.parametrize("function, parameter", [
    (add_iid_log_noise, "noise_sd"),
    (add_bid_ask_bounce, "half_spread"),
])
def test_noise_scales_must_be_nonnegative(function, parameter):
    with pytest.raises(ValueError, match="nonnegative"):
        function(np.array([1.0, 1.01]), **{parameter: -1e-4})
