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


def subsampled_variance(
    prices: ArrayLike,
    step: int,
    n_grids: int | None = None,
) -> float:
    """Average regular-grid RV over equally spaced grid origins.

    This reduces sensitivity to an arbitrary grid origin but does not remove
    observation-noise bias. With ``n_grids=None``, every possible origin is used.
    """
    _positive_integer(step, "step")
    if n_grids is None:
        n_grids = step
    _positive_integer(n_grids, "n_grids")
    if n_grids > step:
        raise ValueError("n_grids must satisfy 1 <= n_grids <= step")

    offsets = np.linspace(0, step - 1, n_grids, dtype=int)
    values = [realized_variance(prices, step=step, offset=int(offset)) for offset in offsets]
    return float(np.mean(values))


def cumulative_realized_variance(prices: ArrayLike) -> NDArray[np.float64]:
    """Running RV along a path: zero at the first price, then summed squared log returns.

    The last value equals ``realized_variance(prices)``.
    """
    returns = np.diff(_log_prices(prices))
    return np.r_[0.0, np.cumsum(returns**2)]


def grid_realized_variances(prices: ArrayLike, step: int) -> NDArray[np.float64]:
    """RV on every grid origin ``0 .. step - 1`` for one sampling step.

    Their mean is the subsampled RV over all origins.
    """
    _positive_integer(step, "step")
    return np.array(
        [realized_variance(prices, step=step, offset=offset) for offset in range(step)]
    )
