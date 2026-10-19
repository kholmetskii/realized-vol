"""Adapters from timestamped market prices to pure variance estimators."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rvol.data.market.sampling import SamplingOrigin, last_tick_sample
from rvol.data.market.sessions import session_labels
from rvol.estimators.variance import realized_variance as array_realized_variance


def sampled_realized_variance(
    prices: pd.Series,
    frequency: str,
    origin: SamplingOrigin = "start_day",
) -> float:
    """Sample timestamped prices and estimate variance on the resulting array."""
    sampled = last_tick_sample(prices, frequency, origin)
    if len(sampled) < 3:
        return np.nan
    return array_realized_variance(sampled.to_numpy())


def daily_realized_variance(prices: pd.Series, frequency: str) -> pd.Series:
    """Return one realized-variance estimate per FX session."""
    return prices.groupby(session_labels(prices)).apply(
        lambda session: sampled_realized_variance(session, frequency)
    ).dropna()


def daily_observation_count(prices: pd.Series, frequency: str) -> pd.Series:
    """Count sampled prices in every FX session."""
    return prices.groupby(session_labels(prices)).apply(
        lambda session: float(len(last_tick_sample(session, frequency)))
    )
