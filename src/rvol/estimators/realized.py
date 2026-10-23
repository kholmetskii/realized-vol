"""Realized variance of timestamped prices sampled on a regular time grid."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rvol.estimators.sampling import last_tick_sample
from rvol.estimators.variance import realized_variance


def sampled_realized_variance(prices: pd.Series, frequency: str) -> float:
    """Realized variance of one period's prices after sampling them on a grid.

    The caller decides which prices form the period. Only returns between
    prices inside the period count: the return from the previous period's last
    price to this period's first is not included. Fewer than two sampled
    prices give NaN.
    """
    sampled = last_tick_sample(prices, frequency)
    if len(sampled) < 2:
        return np.nan
    return realized_variance(sampled.to_numpy())
