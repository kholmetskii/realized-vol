"""Build compact daily realized-volatility datasets from tick-data chunks."""

from __future__ import annotations

import pathlib
from collections.abc import Iterable

import numpy as np
import pandas as pd

from rvol.features.realized import daily_observation_count, daily_realized_variance
from rvol.market.sessions import MIN_SESSION_HOURS, full_sessions, session_labels


def _read_mid_prices(path: pathlib.Path) -> pd.Series:
    """Read and validate the timestamp and mid-price columns from one chunk."""
    try:
        ticks = pd.read_parquet(path, columns=["ts", "mid"])
    except (KeyError, ValueError) as error:
        raise ValueError(f"{path} must contain 'ts' and 'mid' columns") from error

    timestamps = pd.to_datetime(ticks["ts"], utc=True, errors="coerce")
    mid = pd.to_numeric(ticks["mid"], errors="coerce")
    valid = timestamps.notna() & np.isfinite(mid) & (mid > 0)
    if not valid.all():
        invalid = int((~valid).sum())
        raise ValueError(f"{path} contains {invalid} invalid timestamps or mid prices")

    frame = pd.DataFrame({"ts": timestamps, "mid": mid.astype("float64")})
    frame = frame.drop_duplicates("ts", keep="last").sort_values("ts")
    return frame.set_index("ts")["mid"]


def _summarize_sessions(
    prices: pd.Series,
    frequency: str,
    min_session_hours: float,
) -> pd.DataFrame:
    """Turn complete timestamped sessions into one row per trading day."""
    if prices.empty:
        return pd.DataFrame(columns=["date", f"rv_{frequency}", "log_rv", "observation_count"])

    complete = full_sessions(prices, min_hours=min_session_hours)
    if complete.empty:
        return pd.DataFrame(columns=["date", f"rv_{frequency}", "log_rv", "observation_count"])

    rv = daily_realized_variance(complete, frequency)
    counts = daily_observation_count(complete, frequency).reindex(rv.index)
    rv_name = f"rv_{frequency}"
    result = pd.DataFrame({
        "date": pd.to_datetime(rv.index),
        rv_name: rv.to_numpy(dtype="float64"),
        "observation_count": counts.to_numpy(dtype="int64"),
    })
    result = result[np.isfinite(result[rv_name]) & (result[rv_name] > 0)].copy()
    result["log_rv"] = np.log(result[rv_name])
    return result[["date", rv_name, "log_rv", "observation_count"]]


def build_daily_dataset(
    paths: Iterable[str | pathlib.Path],
    *,
    frequency: str = "5min",
    min_session_hours: float = MIN_SESSION_HOURS,
) -> pd.DataFrame:
    """Build daily RV while reading tick chunks one at a time.

    The final FX session in each file is carried into the next file because a
    17:00 New York session normally crosses a UTC calendar boundary. Overlapping
    timestamps and trading days are deduplicated, making repeated chunk ranges
    safe to combine.
    """
    sources = sorted({pathlib.Path(path) for path in paths})
    if not sources:
        raise ValueError("at least one tick-data file is required")
    if min_session_hours <= 0:
        raise ValueError("min_session_hours must be positive")
    try:
        if pd.Timedelta(frequency) <= pd.Timedelta(0):
            raise ValueError
    except (TypeError, ValueError) as error:
        raise ValueError("frequency must be a positive pandas frequency") from error

    carry = pd.Series(dtype="float64", name="mid")
    summaries: list[pd.DataFrame] = []

    for path in sources:
        if not path.is_file():
            raise FileNotFoundError(path)
        prices = _read_mid_prices(path)
        combined = pd.concat([carry, prices])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        if combined.empty:
            continue

        labels = session_labels(combined)
        final_label = labels.iloc[-1]
        ready = combined[labels != final_label]
        carry = combined[labels == final_label]
        summary = _summarize_sessions(ready, frequency, min_session_hours)
        if not summary.empty:
            summaries.append(summary)

    final_summary = _summarize_sessions(carry, frequency, min_session_hours)
    if not final_summary.empty:
        summaries.append(final_summary)

    rv_name = f"rv_{frequency}"
    columns = ["date", rv_name, "log_rv", "observation_count"]
    if not summaries:
        return pd.DataFrame(columns=columns)

    result = pd.concat(summaries, ignore_index=True)
    # Overlapping source chunks can produce the same session twice. Prefer the
    # copy with more sampled observations because a chunk-edge copy may contain
    # only the latter portion of an otherwise complete session.
    result = result.sort_values(["date", "observation_count"])
    result = result.drop_duplicates("date", keep="last").sort_values("date")
    return result.reset_index(drop=True)[columns]
