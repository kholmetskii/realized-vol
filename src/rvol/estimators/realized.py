"""Realized variance of timestamped prices sampled on a regular time grid."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rvol.estimators.sampling import SamplingOrigin, last_tick_sample
from rvol.estimators.variance import realized_variance as array_realized_variance


def sampled_realized_variance(
    prices: pd.Series,
    frequency: str,
    origin: SamplingOrigin = "start_day",
) -> float:
    """Sample timestamped prices and estimate variance on the resulting array.

    The caller decides which prices form one period, for example one trading
    session; fewer than three sampled prices give NaN.
    """
    sampled = last_tick_sample(prices, frequency, origin)
    if len(sampled) < 3:
        return np.nan
    return array_realized_variance(sampled.to_numpy())
