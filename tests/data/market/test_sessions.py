"""FX sessions end at 17:00 New York; short sessions are not observations."""

from typing import cast

import pandas as pd
import pytest

from rvol.data.market import full_sessions, prices_per_session, session_window, trading_day


def test_trading_day_labels_sunday_evening_as_monday():
    """A session ends at 17:00 New York and is named after the day it ends."""
    ts = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-01-14 23:30", tz="UTC"),  # Sunday, after the open
            pd.Timestamp("2024-01-15 14:00", tz="UTC"),  # Monday, London afternoon
            pd.Timestamp("2024-01-15 21:59", tz="UTC"),  # Monday, just before close
            pd.Timestamp("2024-01-15 22:30", tz="UTC"),  # already Tuesday's session
        ]
    )
    labels = trading_day(ts)
    assert list(labels[:3]) == [pd.Timestamp("2024-01-15")] * 3
    assert labels[3] == pd.Timestamp("2024-01-16")


def test_session_boundary_follows_new_york_daylight_saving():
    """17:00 New York is 22:00 UTC in winter and 21:00 UTC in summer.

    A boundary pinned to a fixed UTC hour would be an hour out for half the
    year, cutting one session short and stretching its neighbour.
    """
    winter = pd.Timestamp("2024-01-10 21:30", tz="UTC")  # 16:30 New York, EST
    summer = pd.Timestamp("2024-03-13 21:30", tz="UTC")  # 17:30 New York, EDT

    assert trading_day(pd.DatetimeIndex([winter]))[0] == pd.Timestamp("2024-01-10")
    assert trading_day(pd.DatetimeIndex([summer]))[0] == pd.Timestamp("2024-03-14")


def test_naive_timestamps_are_read_as_utc():
    naive = pd.DatetimeIndex([pd.Timestamp("2024-01-15 14:00")])
    aware = pd.DatetimeIndex([pd.Timestamp("2024-01-15 14:00", tz="UTC")])
    assert trading_day(naive)[0] == trading_day(aware)[0]


def test_short_sessions_are_dropped():
    """A two-hour fragment, such as a holiday close, is not a day.

    Keeping it would bias every average.
    """
    # 22:00 UTC is the winter open: 17:00 New York.
    full = pd.date_range("2024-01-15 22:00", periods=24, freq="1h", tz="UTC")
    short = pd.date_range("2024-01-19 22:00", periods=2, freq="1h", tz="UTC")
    s = pd.Series(1.0, index=full.append(short))

    kept = full_sessions(s)
    assert len(kept) == len(full)
    assert set(trading_day(cast(pd.DatetimeIndex, kept.index))) == {pd.Timestamp("2024-01-16")}


def test_session_windows_follow_new_york_wall_clock_across_daylight_saving():
    """US clocks moved forward on Sunday 10 March 2024."""
    friday = session_window(pd.Timestamp("2024-03-08"))
    monday = session_window(pd.Timestamp("2024-03-11"))

    assert friday == (
        pd.Timestamp("2024-03-07 22:00", tz="UTC"),
        pd.Timestamp("2024-03-08 22:00", tz="UTC"),
    )
    assert monday == (
        pd.Timestamp("2024-03-10 21:00", tz="UTC"),
        pd.Timestamp("2024-03-11 21:00", tz="UTC"),
    )


@pytest.mark.parametrize(("frequency", "count"), [("5min", 288), ("1h", 24), ("30s", 2880)])
def test_prices_per_session_divides_a_day_by_the_sampling_step(frequency, count):
    assert prices_per_session(frequency) == count


@pytest.mark.parametrize("frequency", ["0min", "-5min", "7min"])
def test_prices_per_session_needs_a_step_that_divides_the_session(frequency):
    with pytest.raises(ValueError, match="divide"):
        prices_per_session(frequency)
