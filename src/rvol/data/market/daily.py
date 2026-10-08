"""Validation of an observed daily log-variance series."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ordered_log_variance(
    daily: pd.DataFrame, *, date_col: str, rv_col: str, max_gap_days: int | None = None,
) -> pd.DataFrame:
    """Return a copy with canonical columns, ordered by unique observed session."""
    missing = {date_col, rv_col}.difference(daily.columns)
    if missing:
        raise ValueError(f"daily data is missing required columns: {', '.join(sorted(missing))}")
    dates = pd.to_datetime(daily[date_col], errors="coerce")
    values = pd.to_numeric(daily[rv_col], errors="coerce")
    valid = dates.notna() & np.isfinite(values)
    if not valid.all():
        invalid = int((~valid).sum())
        raise ValueError(f"daily data contains {invalid} invalid dates or {rv_col} values")
    ordered = pd.DataFrame({"date": dates, "log_rv": values.astype("float64")})
    ordered = ordered.sort_values("date").reset_index(drop=True)
    if ordered["date"].duplicated().any():
        raise ValueError("daily data must contain one row per session date")
    if max_gap_days is not None:
        gap_days = ordered["date"].diff().dt.total_seconds() / (24 * 60 * 60)
        large = gap_days > max_gap_days
        if large.any():
            position = int(np.flatnonzero(large.to_numpy())[0])
            previous = ordered["date"].iloc[position - 1].date()
            following = ordered["date"].iloc[position].date()
            days = int(gap_days.iloc[position])
            raise ValueError(
                f"daily data contains a {days}-day gap from {previous} to {following}"
            )
    return ordered


def validate_daily_columns(date_col: str, rv_col: str) -> None:
    if any(not isinstance(name, str) or not name.strip() for name in (date_col, rv_col)):
        raise ValueError("daily column names must be non-empty strings")
    if date_col == rv_col:
        raise ValueError("date and log-variance columns must be distinct")
