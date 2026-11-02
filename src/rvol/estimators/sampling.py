"""Regular time grids for irregularly timestamped prices."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from rvol.estimators.variance import realized_variance


def last_tick_sample(prices: pd.Series, frequency: str) -> pd.Series:
    """Put irregular prices on a regular grid, keeping the last price in each bin.

    Bins are anchored at midnight UTC; empty bins are dropped.
    """
    return prices.resample(frequency, origin="start_day").last().dropna()


def realized_variance_by_frequency(prices: pd.Series, frequencies: Sequence[str]) -> pd.Series:
    """Realized variance of one period's prices at each sampling frequency.

    Averaged over many periods, these values are a signature plot (Andersen,
    Bollerslev, Diebold & Labys, 2000): flat across frequencies where noise is
    negligible, rising as the interval shrinks once noise dominates.
    """
    if not frequencies:
        raise ValueError("frequencies must not be empty")
    values = {f: realized_variance(last_tick_sample(prices, f).to_numpy()) for f in frequencies}
    return pd.Series(values, name="realized_variance", dtype="float64").rename_axis("frequency")
