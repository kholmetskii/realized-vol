"""Adapters from timestamped market prices to pure variance estimators."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rvol.estimators.realized import realized_variance as array_realized_variance
from rvol.market.sampling import SamplingOrigin, grid_origins, last_tick_sample
from rvol.market.sessions import session_labels


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


def subsampled_realized_variance(
    prices: pd.Series,
    frequency: str,
    n_grids: int = 12,
) -> float:
    """Average timestamped-price RV over shifted regular sampling grids."""
    estimates = [
        sampled_realized_variance(prices, frequency, origin=origin)
        for origin in grid_origins(prices, frequency, n_grids)
    ]
    valid = [estimate for estimate in estimates if not np.isnan(estimate)]
    return float(np.mean(valid)) if valid else np.nan


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
