"""Foreign-exchange session conventions.

This module owns market-specific time rules.  It deliberately contains no
variance estimators: assigning quotes to sessions is a finance convention,
not a numerical estimation formula.
"""

from __future__ import annotations

import pandas as pd

#: The conventional foreign-exchange session close in New York local time.
SESSION_CLOSE_HOUR = 17

#: The close follows US daylight saving, so it must not be fixed in UTC.
SESSION_TZ = "America/New_York"

#: Shorter sessions are holidays or partial weekly opens, not full observations.
MIN_SESSION_HOURS = 12.0


def trading_day(
    timestamps: pd.DatetimeIndex,
    close_hour: int = SESSION_CLOSE_HOUR,
    timezone: str = SESSION_TZ,
) -> pd.DatetimeIndex:
    """Return the FX session date for each timestamp.

    A session is named after the local date on which it closes.  Naive input
    timestamps are interpreted as UTC, which is also the Dukascopy convention.
    """
    index = timestamps.tz_localize("UTC") if timestamps.tz is None else timestamps
    shifted = index.tz_convert(timezone) + pd.Timedelta(hours=24 - close_hour)
    return pd.DatetimeIndex(shifted.normalize().tz_localize(None))


def session_labels(prices: pd.Series) -> pd.Series:
    """Return one FX-session label per timestamp, aligned with ``prices``."""
    labels = trading_day(pd.DatetimeIndex(prices.index))
    return pd.Series(labels, index=prices.index)


def full_sessions(
    prices: pd.Series,
    min_hours: float = MIN_SESSION_HOURS,
) -> pd.Series:
    """Keep sessions whose first-to-last observation span is long enough."""
    labels = session_labels(prices)
    span_hours = prices.groupby(labels).apply(
        lambda session: (
            session.index[-1] - session.index[0]
        ).total_seconds()
        / 3600.0
    )
    retained = set(span_hours[span_hours >= min_hours].index)
    return prices[labels.isin(retained)]
