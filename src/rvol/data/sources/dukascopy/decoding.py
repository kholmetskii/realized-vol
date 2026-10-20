"""Decode Dukascopy's compressed hourly tick records."""

from __future__ import annotations

import datetime as dt
import lzma

import numpy as np
import pandas as pd

EMPTY_COLUMNS = ["ts", "bid", "ask", "bid_vol", "ask_vol"]

RECORD = np.dtype([
    ("ms", ">u4"),
    ("ask", ">u4"),
    ("bid", ">u4"),
    ("ask_vol", ">f4"),
    ("bid_vol", ">f4"),
])


def point_divider(symbol: str) -> float:
    """Integer price divider. JPY pairs and most CFDs quote to 3 decimals,
    the rest of FX to 5."""
    return 1e3 if "JPY" in symbol.upper() else 1e5


def decode_hour(
    raw: bytes | None, hour_start: dt.datetime, divisor: float
) -> pd.DataFrame:
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
    ts = pd.Timestamp(hour_start, tz="UTC") + pd.to_timedelta(
        arr["ms"].astype("int64"), unit="ms"
    )
    return pd.DataFrame({
        "ts": ts,
        "bid": arr["bid"].astype("float64") / divisor,
        "ask": arr["ask"].astype("float64") / divisor,
        "bid_vol": arr["bid_vol"].astype("float64"),
        "ask_vol": arr["ask_vol"].astype("float64"),
    })
