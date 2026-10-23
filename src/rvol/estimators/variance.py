"""Realized variance of a price path, observed or simulated."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def realized_variance(prices: ArrayLike) -> float:
    """Sum of the squared log returns between consecutive prices.

    n prices give n - 1 returns. The result is on the variance scale of the
    path's own period, without annualization.
    """
    values = np.asarray(prices, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("prices must be a one-dimensional array with at least two values")
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("prices must be finite and strictly positive")
    return float(np.sum(np.diff(np.log(values)) ** 2))
