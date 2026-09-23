"""Realized-variance estimators for uniformly sampled simulated paths."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


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
    if step < 1:
        raise ValueError("step must be at least one")
    if not 0 <= offset < step:
        raise ValueError("offset must satisfy 0 <= offset < step")
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
    if step < 1:
        raise ValueError("step must be at least one")
    if n_grids is None:
        n_grids = step
    if not 1 <= n_grids <= step:
        raise ValueError("n_grids must satisfy 1 <= n_grids <= step")

    offsets = np.linspace(0, step - 1, n_grids, dtype=int)
    values = [realized_variance(prices, step=step, offset=int(offset)) for offset in offsets]
    return float(np.mean(values))
