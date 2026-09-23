"""Observation models applied to an already simulated efficient price path."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _prices(values: ArrayLike) -> NDArray[np.float64]:
    prices = np.asarray(values, dtype=np.float64)
    if prices.ndim != 1 or len(prices) < 2:
        raise ValueError("prices must be a one-dimensional array with at least two values")
    if not np.all(np.isfinite(prices)) or np.any(prices <= 0):
        raise ValueError("prices must be finite and strictly positive")
    return prices


def add_iid_log_noise(
    efficient_prices: ArrayLike,
    noise_sd: float,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Observe an efficient path with independent Gaussian log-price noise.

    ``noise_sd`` is the standard deviation of the observation error in log-price
    units. This is the conventional ``observed log price = efficient log price +
    noise`` model; it is not a bid-ask-bounce model.
    """
    if noise_sd < 0:
        raise ValueError("noise_sd must be nonnegative")
    prices = _prices(efficient_prices)
    if noise_sd == 0:
        return prices.copy()
    rng = rng or np.random.default_rng()
    return prices * np.exp(rng.normal(0.0, noise_sd, size=prices.shape))


def add_bid_ask_bounce(
    efficient_prices: ArrayLike,
    half_spread: float,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Observe trades randomly at the bid or ask around an efficient price.

    ``half_spread`` is the log-price distance from the efficient price to either
    quote. Independent trade directions create the negative first-order return
    autocorrelation associated with bid-ask bounce while keeping the efficient
    path unchanged.
    """
    if half_spread < 0:
        raise ValueError("half_spread must be nonnegative")
    prices = _prices(efficient_prices)
    if half_spread == 0:
        return prices.copy()
    rng = rng or np.random.default_rng()
    side = rng.choice(np.array([-1.0, 1.0]), size=prices.shape)
    return prices * np.exp(side * half_spread)
