"""Regular time grids for irregularly timestamped prices."""

from __future__ import annotations

import pandas as pd


def last_tick_sample(prices: pd.Series, frequency: str) -> pd.Series:
    """Put irregular prices on a regular grid, keeping the last price in each bin.

    Bins are anchored at midnight UTC; empty bins are dropped.
    """
    return prices.resample(frequency, origin="start_day").last().dropna()
