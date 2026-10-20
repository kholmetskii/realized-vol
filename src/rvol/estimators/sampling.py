"""Regular time grids for irregularly timestamped prices."""

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
