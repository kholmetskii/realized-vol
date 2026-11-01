"""Realized variance of a price path, observed or simulated, and its split by sign."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _log_returns(prices: ArrayLike) -> np.ndarray:
    values = np.asarray(prices, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("prices must be a one-dimensional array with at least two values")
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("prices must be finite and strictly positive")
    return np.diff(np.log(values))


def realized_variance(prices: ArrayLike) -> float:
    """Sum of the squared log returns between consecutive prices.

    n prices give n - 1 returns. The result is on the variance scale of the
    path's own period, without annualization.
    """
    return float(np.sum(_log_returns(prices) ** 2))


def realized_semivariances(prices: ArrayLike) -> tuple[float, float]:
    """Realized variance split by the sign of each return: (positive, negative).

    RS+ sums the squared positive log returns and RS- the squared negative
    ones (Barndorff-Nielsen, Kinnebrock & Shephard, 2010). A zero return adds
    to neither, so RS+ + RS- equals the realized variance up to rounding.
    """
    returns = _log_returns(prices)
    squared = returns**2
    return float(np.sum(squared[returns > 0])), float(np.sum(squared[returns < 0]))
