"""Decoder tests: build a synthetic .bi5 from known values and check we get
exactly those values back."""

import pandas as pd
import pytest

from rvol.data.sources.dukascopy import decode_hour, point_divider
from tests.data.sources.dukascopy.fakes import HOUR, make_bi5


def test_decode_roundtrip():
    raw = make_bi5(
        [
            (0, 109123, 109118, 1.5, 2.0),
            (1500, 109125, 109120, 0.7, 0.9),
            (3_599_999, 109130, 109124, 3.0, 1.1),
        ]
    )
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


@pytest.mark.parametrize(
    ("symbol", "raw_price", "price"),
    [
        # Integer prices observed in the week of 2023-01-09, and the price each
        # instrument traded at then.
        ("EURUSD", 107524, 1.07524),
        ("USDJPY", 131816, 131.816),
        ("XAUUSD", 1879095, 1879.095),
        ("USA500IDXUSD", 3940397, 3940.397),
    ],
)
def test_point_divider_gives_the_traded_price(symbol, raw_price, price):
    df = decode_hour(make_bi5([(0, raw_price, raw_price, 1.0, 1.0)]), HOUR, point_divider(symbol))
    assert df["ask"].iloc[0] == pytest.approx(price, rel=1e-12)


def test_point_divider_ignores_case():
    assert point_divider("xauusd") == point_divider("XAUUSD")


def test_unknown_symbol_has_no_point_divider():
    with pytest.raises(ValueError, match="no point divider known for 'GBPUSD'"):
        point_divider("GBPUSD")
