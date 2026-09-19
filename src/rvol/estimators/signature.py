"""Volatility signature: realized variance as a function of sampling frequency.

As sampling moves toward the tick, RV drifts upward — you are summing squared
microstructure noise rather than information. The plateau at moderate
frequencies is the reason the industry standard is five minutes.

Days are FX trading days, not calendar days: the week opens on Sunday evening
and each session runs 21:00 UTC to 21:00 UTC (17:00 New York, the market
convention). Splitting at midnight UTC instead cuts the New York afternoon in
half and turns each Sunday evening into a two-hour "day" whose tiny RV drags
the average down.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Frequency grid from 1 second to an hour
FREQS: list[str] = ["1s", "2s", "5s", "10s", "15s", "30s", "1min", "2min",
                    "5min", "10min", "15min", "30min", "60min"]

#: The session boundary, in hours UTC: 21:00 is 17:00 New York.
SESSION_CLOSE_HOUR = 21

#: A session shorter than this is a holiday or a half-open Sunday, not a day.
MIN_SESSION_HOURS = 12.0


def freq_seconds(freq: str) -> float:
    return pd.Timedelta(freq).total_seconds()


def trading_day(ts: pd.DatetimeIndex, close_hour: int = SESSION_CLOSE_HOUR
                ) -> pd.DatetimeIndex:
    """Label each timestamp with the trading day it belongs to.

    A session runs from `close_hour` UTC to `close_hour` UTC and is named after
    the calendar date it ends on, so Sunday 22:00 and Monday 14:00 share the
    label Monday.
    """
    shifted = ts + pd.Timedelta(hours=24 - close_hour)
    return shifted.normalize().tz_localize(None)


def realized_variance(prices: pd.Series, freq: str) -> float:
    """RV for one day using last-tick sampling at step `freq`."""
    p = prices.resample(freq).last().dropna()
    if len(p) < 3:
        return np.nan
    r = np.diff(np.log(p.to_numpy()))
    return float(np.sum(r**2))


def full_sessions(s: pd.Series, min_hours: float = MIN_SESSION_HOURS
                  ) -> pd.Series:
    """Drop sessions that cover too little of the day to be comparable.

    A Sunday holds only the two hours after the weekly open; keeping it as a
    whole day would bias every average downward.
    """
    day = trading_day(pd.DatetimeIndex(s.index))
    span = s.groupby(day).apply(
        lambda x: (x.index[-1] - x.index[0]).total_seconds() / 3600.0
    )
    keep = set(span[span >= min_hours].index)
    return s[pd.Series(day, index=s.index).isin(keep)]


def signature(ticks: pd.DataFrame, price_col: str = "mid",
              freqs: list[str] | None = None,
              min_hours: float = MIN_SESSION_HOURS) -> pd.DataFrame:
    """Average daily RV across a grid of sampling frequencies.

    Averages per trading day first, so that long days do not dominate, and
    keeps only sessions covering at least `min_hours` hours.
    """
    freqs = freqs or FREQS
    s = ticks.set_index("ts")[price_col].sort_index()
    s = full_sessions(s, min_hours)
    day = pd.Series(trading_day(pd.DatetimeIndex(s.index)), index=s.index)

    rows = []
    for f in freqs:
        daily = s.groupby(day).apply(
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
    out = pd.DataFrame(rows)

    # Standard error of the annualised figure, by the delta method:
    # d/dRV of sqrt(252 RV) = sqrt(252) / (2 sqrt(RV)).
    out["ann_vol_se"] = (
        out["se"] * np.sqrt(252) / (2 * np.sqrt(out["mean_RV"])) * 100
    )
    return out
