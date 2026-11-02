"""Decode Dukascopy's compressed hourly tick records."""

from __future__ import annotations

import datetime as dt
import lzma

import numpy as np
import pandas as pd

EMPTY_COLUMNS = ["ts", "bid", "ask", "bid_vol", "ask_vol"]

RECORD = np.dtype(
    [
        ("ms", ">u4"),
        ("ask", ">u4"),
        ("bid", ">u4"),
        ("ask_vol", ">f4"),
        ("bid_vol", ">f4"),
    ]
)


#: Integer price divider per instrument. Each entry was checked by decoding the
#: week of 2023-01-09 and comparing the price with the instrument's known level
#: then: EURUSD 107524 -> 1.075, USDJPY 131816 -> 131.8, XAUUSD (gold, in USD
#: per ounce) 1879095 -> 1879, USA500IDXUSD (S&P 500 CFD) 3940397 -> 3940.
POINT_DIVIDERS: dict[str, float] = {
    "EURUSD": 1e5,
    "USDJPY": 1e3,
    "XAUUSD": 1e3,
    "USA500IDXUSD": 1e3,
}


def point_divider(symbol: str) -> float:
    """The integer price divider of a known instrument.

    A wrong divider would store every price scaled by a constant. Log returns
    and RV would not change, but the prices would be silently wrong, so an
    unknown symbol raises instead of guessing.
    """
    try:
        return POINT_DIVIDERS[symbol.upper()]
    except KeyError:
        raise ValueError(
            f"no point divider known for {symbol!r}: decode one hour with divisor 1, "
            "compare the integer price with the instrument's known price level, "
            "and add the power of ten to POINT_DIVIDERS"
        ) from None


def decode_hour(raw: bytes | None, hour_start: dt.datetime, divisor: float) -> pd.DataFrame:
    """Decompress and decode one hourly body into a DataFrame."""
    if not raw:
        return pd.DataFrame(columns=EMPTY_COLUMNS)

    body = lzma.decompress(raw)
    if len(body) == 0:
        return pd.DataFrame(columns=EMPTY_COLUMNS)
    if len(body) % RECORD.itemsize:
        raise ValueError(
            f"{hour_start}: body length {len(body)} is not a multiple of {RECORD.itemsize}"
        )

    arr = np.frombuffer(body, dtype=RECORD)
    ts = pd.Timestamp(hour_start, tz="UTC") + pd.to_timedelta(arr["ms"].astype("int64"), unit="ms")
    return pd.DataFrame(
        {
            "ts": ts,
            "bid": arr["bid"].astype("float64") / divisor,
            "ask": arr["ask"].astype("float64") / divisor,
            "bid_vol": arr["bid_vol"].astype("float64"),
            "ask_vol": arr["ask_vol"].astype("float64"),
        }
    )
