"""Decoder tests: build a synthetic .bi5 from known values and check we get
exactly those values back."""

import datetime as dt
import lzma
import struct

import pandas as pd

from rvol.data.dukascopy import decode_hour, hour_url, point_divider

HOUR = dt.datetime(2024, 1, 15, 14)


def make_bi5(records: list[tuple[int, int, int, float, float]]) -> bytes:
    body = b"".join(struct.pack(">IIIff", *r) for r in records)
    return lzma.compress(body, format=lzma.FORMAT_ALONE)


def test_decode_roundtrip():
    raw = make_bi5([
        (0, 109123, 109118, 1.5, 2.0),
        (1500, 109125, 109120, 0.7, 0.9),
        (3_599_999, 109130, 109124, 3.0, 1.1),
    ])
    df = decode_hour(raw, HOUR, point_divider("EURUSD"))

    assert len(df) == 3
    assert df["ts"].iloc[0] == pd.Timestamp("2024-01-15 14:00:00", tz="UTC")
    assert df["ts"].iloc[-1] == pd.Timestamp("2024-01-15 14:59:59.999", tz="UTC")
    assert abs(df["ask"].iloc[0] - 1.09123) < 1e-12
    assert abs(df["bid"].iloc[0] - 1.09118) < 1e-12
    assert abs(df["ask_vol"].iloc[0] - 1.5) < 1e-6
    assert (df["ask"] >= df["bid"]).all()


def test_empty_hour_is_not_an_error():
    """Weekends and holidays come back as a 404 or a zero-length body."""
    assert len(decode_hour(b"", HOUR, 1e5)) == 0
    assert len(decode_hour(make_bi5([]), HOUR, 1e5)) == 0


def test_jpy_uses_a_different_divider():
    assert point_divider("USDJPY") == 1e3
    assert point_divider("EURUSD") == 1e5


def test_url_month_is_zero_indexed():
    """The classic Dukascopy trap: month counts from zero, day and hour do not."""
    assert hour_url("EURUSD", dt.datetime(2015, 1, 15, 14)).endswith(
        "/EURUSD/2015/00/15/14h_ticks.bi5"
    )
    assert hour_url("EURUSD", dt.datetime(2015, 12, 3, 9)).endswith(
        "/EURUSD/2015/11/03/09h_ticks.bi5"
    )
