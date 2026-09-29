"""Volatility signatures across a range of sampling frequencies."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rvol.features.realized import daily_realized_variance
from rvol.market.sessions import MIN_SESSION_HOURS, full_sessions

FREQS: list[str] = [
    "1s",
    "2s",
    "5s",
    "10s",
    "15s",
    "30s",
    "1min",
    "2min",
    "5min",
    "10min",
    "15min",
    "30min",
    "60min",
]

TRADING_DAYS_PER_YEAR = 252


def freq_seconds(frequency: str) -> float:
    """Convert a pandas frequency string to seconds."""
    return pd.Timedelta(frequency).total_seconds()


def signature(
    ticks: pd.DataFrame,
    price_col: str = "mid",
    freqs: list[str] | None = None,
    min_hours: float = MIN_SESSION_HOURS,
) -> pd.DataFrame:
    """Summarise average daily RV across sampling frequencies.

    The result contains variance-scale estimates and annualised volatility.
    Market-specific preparation lives below this diagnostic in ``market`` and
    ``features``; numerical RV itself lives in ``estimators``.
    """
    frequencies = freqs or FREQS
    prices = ticks.set_index("ts")[price_col].sort_index()
    prices = full_sessions(prices, min_hours)

    rows = []
    for frequency in frequencies:
        daily = daily_realized_variance(prices, frequency)
        mean_rv = daily.mean()
        rows.append(
            {
                "freq": frequency,
                "seconds": freq_seconds(frequency),
                "mean_RV": mean_rv,
                "ann_vol_pct": np.sqrt(mean_rv * TRADING_DAYS_PER_YEAR) * 100,
                "n_days": len(daily),
                "se": daily.std(ddof=1) / np.sqrt(len(daily)),
            }
        )

    result = pd.DataFrame(rows)
    result["ann_vol_se"] = (
        result["se"]
        * np.sqrt(TRADING_DAYS_PER_YEAR)
        / (2 * np.sqrt(result["mean_RV"]))
        * 100
    )
    return result
