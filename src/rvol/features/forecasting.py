"""Leakage-safe features for realized-volatility forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd

from rvol.domain.specifications import ComponentSpecification


@dataclass(frozen=True)
class HARFeatureBuilder:
    """Bind HAR feature settings to both preparation and experiment metadata."""

    date_col: str = "date"
    rv_col: str = "log_rv"
    weekly_window: int = 5
    monthly_window: int = 22
    max_gap_days: int | None = 7

    feature_names: ClassVar[tuple[str, ...]] = ("rv_daily", "rv_weekly", "rv_monthly")

    @property
    def specification(self) -> ComponentSpecification:
        return ComponentSpecification(
            name="HAR_features",
            implementation=f"{type(self).__module__}.{type(self).__qualname__}",
            feature_names=self.feature_names,
            parameters=(
                ("date_col", self.date_col),
                ("rv_col", self.rv_col),
                ("weekly_window", self.weekly_window),
                ("monthly_window", self.monthly_window),
                ("max_gap_days", self.max_gap_days),
            ),
        )

    def __call__(self, daily: pd.DataFrame) -> pd.DataFrame:
        return build_har_features(
            daily,
            date_col=self.date_col,
            rv_col=self.rv_col,
            weekly_window=self.weekly_window,
            monthly_window=self.monthly_window,
            max_gap_days=self.max_gap_days,
        )


def build_har_features(
    daily: pd.DataFrame,
    *,
    date_col: str = HARFeatureBuilder.date_col,
    rv_col: str = HARFeatureBuilder.rv_col,
    weekly_window: int = HARFeatureBuilder.weekly_window,
    monthly_window: int = HARFeatureBuilder.monthly_window,
    max_gap_days: int | None = HARFeatureBuilder.max_gap_days,
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
    if max_gap_days is not None and max_gap_days < 1:
        raise ValueError("max_gap_days must be positive or None")

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
        ordered_dates = pd.DatetimeIndex(ordered["date"])
        gap_days = ordered["date"].diff().dt.total_seconds() / (24 * 60 * 60)
        large = gap_days > max_gap_days
        if large.any():
            position = int(np.flatnonzero(large.to_numpy())[0])
            previous = ordered_dates[position - 1].date()
            following = ordered_dates[position].date()
            days = int(gap_days.iloc[position])
            raise ValueError(
                f"daily data contains a {days}-day gap from {previous} to {following}"
            )

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
