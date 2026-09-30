"""Leakage-safe features for realized-volatility forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_har_features(
    daily: pd.DataFrame,
    *,
    date_col: str = "date",
    rv_col: str = "log_rv",
    weekly_window: int = 5,
    monthly_window: int = 22,
) -> pd.DataFrame:
    """Build one-step-ahead daily, weekly, and monthly HAR-RV features.

    Each row is a forecast made after ``origin_date`` for ``target_date``, the
    following observed session. Rolling windows count trading sessions rather
    than calendar days, so weekends and market holidays do not create gaps.
    """
    missing = {date_col, rv_col}.difference(daily.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"daily data is missing required columns: {names}")
    if weekly_window < 1:
        raise ValueError("weekly_window must be positive")
    if monthly_window < weekly_window:
        raise ValueError("monthly_window must be at least weekly_window")

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

    log_rv = ordered["log_rv"]
    features = pd.DataFrame({
        "origin_date": ordered["date"],
        "target_date": ordered["date"].shift(-1),
        "rv_daily": log_rv,
        "rv_weekly": log_rv.rolling(weekly_window).mean(),
        "rv_monthly": log_rv.rolling(monthly_window).mean(),
        "target": log_rv.shift(-1),
    })
    return features.dropna().reset_index(drop=True)
