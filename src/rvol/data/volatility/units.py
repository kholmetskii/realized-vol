"""Daily variance in the unit traders quote volatility in."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def annualized_volatility_pct(variance: ArrayLike, periods_per_year: float) -> np.ndarray:
    """Per-period variance -> annualized volatility in percent: 100 * sqrt(N * variance).

    "Vol is 7%" means a typical yearly move of about 7%.
    """
    values = np.asarray(variance, dtype="float64")
    if not np.isfinite(periods_per_year) or periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    if np.any(values < 0):
        raise ValueError("variance must be non-negative")
    return 100 * np.sqrt(periods_per_year * values)
