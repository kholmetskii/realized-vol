"""Build compact daily realized-volatility datasets from tick-data chunks."""

from __future__ import annotations

import pathlib
from collections.abc import Iterable

import numpy as np
import pandas as pd

from rvol.data.layout import DataLayout
from rvol.data.market import MIN_SESSION_HOURS, full_sessions, session_labels, session_window
from rvol.estimators import last_tick_sample, realized_semivariances, realized_variance

#: Five-minute prices: fine enough to measure the day's movement, coarse enough
#: that bid-ask bounce does not dominate it (the usual compromise in the literature).
SAMPLING_FREQUENCY = "5min"

#: The output columns. ``rv`` is the session's realized variance at the chosen
#: sampling frequency, ``rs_pos`` and ``rs_neg`` its parts from positive and
#: negative returns, and ``observation_count`` the number of sampled prices.
VARIANCE_COLUMNS = ["rv", "rs_pos", "rs_neg"]
COLUMNS = [
    "date",
    *VARIANCE_COLUMNS,
    *(f"log_{name}" for name in VARIANCE_COLUMNS),
    "observation_count",
]

#: How far the data may start after the first session opens, or end before the
#: last session closes, for those sessions to count as complete.
EDGE_TOLERANCE = pd.Timedelta(hours=1)


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
        return pd.DataFrame(columns=COLUMNS)

    complete = full_sessions(prices, min_hours=min_session_hours)
    if complete.empty:
        return pd.DataFrame(columns=COLUMNS)

    rows = []
    for day, session in complete.groupby(session_labels(complete)):
        sampled = last_tick_sample(session, frequency).to_numpy()
        if len(sampled) < 2:
            continue
        rs_pos, rs_neg = realized_semivariances(sampled)
        variances = (realized_variance(sampled), rs_pos, rs_neg)
        rows.append((day, *variances, len(sampled)))
    result = pd.DataFrame(rows, columns=["date", *VARIANCE_COLUMNS, "observation_count"])
    result["date"] = pd.to_datetime(result["date"])
    result = result.astype({name: "float64" for name in VARIANCE_COLUMNS})
    result = result.astype({"observation_count": "int64"})
    # Every log must exist: a session whose returns all share one sign has a
    # zero semivariance and is dropped, like a session without price changes.
    positive = (result[VARIANCE_COLUMNS] > 0).all(axis=1)
    result = result[positive & np.isfinite(result[VARIANCE_COLUMNS]).all(axis=1)].copy()
    for name in VARIANCE_COLUMNS:
        result[f"log_{name}"] = np.log(result[name])
    return result[COLUMNS]


def build_daily_dataset(
    paths: Iterable[str | pathlib.Path],
    *,
    frequency: str = SAMPLING_FREQUENCY,
    min_session_hours: float = MIN_SESSION_HOURS,
) -> pd.DataFrame:
    """Build daily RV while reading tick chunks one at a time, in time order.

    Each session's prices are sampled on a ``frequency`` grid, and RV and its
    semivariances sum the squared log returns between those prices. Only
    returns inside a session count: the return from one session's last price to
    the next session's first is left out.

    The final FX session in each file is carried into the next file because a
    17:00 New York session normally crosses a UTC calendar boundary, so files
    must sort by name in time order (ISO dates in the names do); a file that
    starts before the previous one is rejected. Overlapping timestamps and
    trading days are deduplicated, making repeated chunk ranges safe to combine.

    The first and last sessions are dropped when the data starts more than
    ``EDGE_TOLERANCE`` after the first one opens or ends that long before the
    last one closes: a file cut mid-session would otherwise pass the
    minimum-length filter with too few prices and too small an RV.
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
    previous_start: pd.Timestamp | None = None
    first_tick: pd.Timestamp | None = None
    last_tick: pd.Timestamp | None = None

    for path in sources:
        if not path.is_file():
            raise FileNotFoundError(path)
        prices = _read_mid_prices(path)
        if not prices.empty:
            start, end = prices.index[0], prices.index[-1]
            if previous_start is not None and start < previous_start:
                raise ValueError(
                    f"{path} starts before the previous file; name files in time order"
                )
            previous_start = start
            first_tick = start if first_tick is None else min(first_tick, start)
            last_tick = end if last_tick is None else max(last_tick, end)
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

    if not summaries:
        return pd.DataFrame(columns=COLUMNS)

    result = pd.concat(summaries, ignore_index=True)
    # Overlapping source chunks can produce the same session twice. Prefer the
    # copy with more sampled observations because a chunk-edge copy may contain
    # only the latter portion of an otherwise complete session.
    result = result.sort_values(["date", "observation_count"])
    result = result.drop_duplicates("date", keep="last").sort_values("date")
    result = result.reset_index(drop=True)[COLUMNS]
    if first_tick is not None and last_tick is not None and not result.empty:
        first_open, _ = session_window(result["date"].iloc[0])
        _, last_close = session_window(result["date"].iloc[-1])
        keep = pd.Series(True, index=result.index)
        keep.iloc[0] = first_tick <= first_open + EDGE_TOLERANCE
        keep.iloc[-1] = keep.iloc[-1] and last_tick >= last_close - EDGE_TOLERANCE
        result = result[keep].reset_index(drop=True)
    return result


def load_daily_dataset(
    paths: Iterable[str | pathlib.Path],
    cache: str | pathlib.Path,
    *,
    frequency: str = SAMPLING_FREQUENCY,
) -> pd.DataFrame:
    """The daily dataset of these tick files, read from ``cache`` when it is current.

    The cache is rebuilt, and rewritten, when it is missing, older than any tick
    file, or holds other columns than ``COLUMNS``. A change to how RV is
    computed that keeps the columns is not detected: delete the cache then.
    """
    sources = sorted({pathlib.Path(path) for path in paths})
    if not sources:
        raise ValueError("at least one tick-data file is required")
    target = pathlib.Path(cache)
    newest_tick = max(path.stat().st_mtime for path in sources)
    if target.is_file() and target.stat().st_mtime > newest_tick:
        cached = pd.read_parquet(target)
        if list(cached.columns) == COLUMNS:
            return cached
    daily = build_daily_dataset(sources, frequency=frequency)
    target.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(target, index=False)
    return daily


def daily_dataset(
    asset: str, layout: DataLayout | None = None, *, frequency: str = SAMPLING_FREQUENCY
) -> pd.DataFrame:
    """An asset's daily dataset from its tick files, through its cache in ``layout``."""
    paths = layout or DataLayout()
    ticks = sorted(paths.ticks(asset).glob("*.parquet"))
    if not ticks:
        raise FileNotFoundError(f"no tick files in {paths.ticks(asset)}")
    return load_daily_dataset(ticks, paths.daily_rv(asset, frequency), frequency=frequency)
