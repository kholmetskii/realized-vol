"""Volatility signature: realized variance as a function of sampling frequency.

As sampling moves toward the tick, RV drifts upward — you are summing squared
microstructure noise rather than information. The plateau at moderate
frequencies is the reason the industry standard is five minutes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Frequency grid from 5 seconds to an hour
FREQS: list[str] = ["5s", "10s", "15s", "30s", "1min", "2min", "5min",
                    "10min", "15min", "30min", "60min"]


def freq_seconds(freq: str) -> float:
    return pd.Timedelta(freq).total_seconds()


def realized_variance(prices: pd.Series, freq: str) -> float:
    """RV for one day using last-tick sampling at step `freq`."""
    p = prices.resample(freq).last().dropna()
    if len(p) < 3:
        return np.nan
    r = np.diff(np.log(p.to_numpy()))
    return float(np.sum(r ** 2))


def signature(ticks: pd.DataFrame, price_col: str = "mid",
              freqs: list[str] | None = None) -> pd.DataFrame:
    """Average daily RV across a grid of sampling frequencies.

    Averages per day first, so that long days do not dominate.
    """
    freqs = freqs or FREQS
    s = ticks.set_index("ts")[price_col].sort_index()

    rows = []
    for f in freqs:
        daily = s.groupby(s.index.normalize()).apply(
            lambda x, f=f: realized_variance(x, f)
        ).dropna()
        rv = daily.mean()
        rows.append({
            "freq": f,
            "seconds": freq_seconds(f),
            "mean_RV": rv,
            "ann_vol_pct": np.sqrt(rv * 252) * 100,   # annualised, 252 days
            "n_days": len(daily),
            "se": daily.std(ddof=1) / np.sqrt(len(daily)),
        })
    return pd.DataFrame(rows)
