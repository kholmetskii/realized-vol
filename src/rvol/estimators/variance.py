"""Regular-grid realized-variance formulas on price arrays, observed or simulated."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _log_prices(values: ArrayLike) -> NDArray[np.float64]:
    prices = np.asarray(values, dtype=np.float64)
    if prices.ndim != 1 or len(prices) < 2:
        raise ValueError("prices must be a one-dimensional array with at least two values")
    if not np.all(np.isfinite(prices)) or np.any(prices <= 0):
        raise ValueError("prices must be finite and strictly positive")
    return np.log(prices)


def realized_variance(prices: ArrayLike, step: int = 1, offset: int = 0) -> float:
    """Sum squared log returns on one regular sampling grid.

    ``step`` is measured in observations of the input path and ``offset`` moves
    the grid origin. The result is on the variance scale, without annualisation.
    """
    _positive_integer(step, "step")
    if (
        isinstance(offset, (bool, np.bool_))
        or not isinstance(offset, (int, np.integer))
        or not 0 <= offset < step
    ):
        raise ValueError("offset must be an integer satisfying 0 <= offset < step")
    sampled = _log_prices(prices)[offset::step]
    if len(sampled) < 2:
        raise ValueError("sampling grid must contain at least two prices")
    returns = np.diff(sampled)
    return float(np.sum(returns**2))
