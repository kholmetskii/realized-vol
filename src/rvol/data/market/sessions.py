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

#: A session runs from one close to the next: 24 hours, except across a DST change.
SESSION_LENGTH = pd.Timedelta(hours=24)


def prices_per_session(frequency: str) -> int:
    """How many sampled prices a full session holds at ``frequency``.

    One price closes each interval, so a 24-hour session sampled every five
    minutes holds 288 prices and 287 returns between them.
    """
    step = pd.Timedelta(frequency)
    if step <= pd.Timedelta(0) or SESSION_LENGTH % step != pd.Timedelta(0):
        raise ValueError("frequency must be positive and divide the session length")
    return SESSION_LENGTH // step


def trading_day(timestamps: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return the FX session date for each timestamp.

    A session is named after the local date on which it closes.  Naive input
    timestamps are interpreted as UTC, which is also the Dukascopy convention.
    """
    index = timestamps.tz_localize("UTC") if timestamps.tz is None else timestamps
    shifted = index.tz_convert(SESSION_TZ) + pd.Timedelta(hours=24 - SESSION_CLOSE_HOUR)
    return pd.DatetimeIndex(shifted.normalize().tz_localize(None))


def session_window(day: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """UTC open and close of the session named ``day``.

    It opens at 17:00 New York on the previous calendar day and closes at 17:00
    New York on ``day``, both in local wall-clock time across DST changes.
    """
    close_day = pd.Timestamp(day).normalize()
    open_day = close_day - pd.Timedelta(days=1)
    open_, close = (
        pd.Timestamp(d.year, d.month, d.day, SESSION_CLOSE_HOUR).tz_localize(SESSION_TZ)
        for d in (open_day, close_day)
    )
    return open_.tz_convert("UTC"), close.tz_convert("UTC")


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
        lambda session: (session.index[-1] - session.index[0]).total_seconds() / 3600.0
    )
    retained = set(span_hours[span_hours >= min_hours].index)
    return prices[labels.isin(retained)]
