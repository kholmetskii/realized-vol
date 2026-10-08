"""Sampling grids for irregular timestamped market observations."""

from __future__ import annotations

from typing import TypeAlias

import pandas as pd

SamplingOrigin: TypeAlias = pd.Timestamp | str


def last_tick_sample(
    prices: pd.Series,
    frequency: str,
    origin: SamplingOrigin = "start_day",
) -> pd.Series:
    """Put irregular prices on a regular grid using the last tick in each bin."""
    return prices.resample(frequency, origin=origin).last().dropna()


def grid_origins(
    prices: pd.Series,
    frequency: str,
    n_grids: int,
) -> list[pd.Timestamp]:
    """Return equally spaced grid origins spanning one sampling interval."""
    if n_grids < 1:
        raise ValueError("n_grids must be at least one")
    if prices.empty:
        raise ValueError("prices must not be empty")

    step = pd.Timedelta(frequency)
    if step <= pd.Timedelta(0):
        raise ValueError("frequency must be positive")

    base = prices.index[0].normalize()
    return [base + step * grid / n_grids for grid in range(n_grids)]
