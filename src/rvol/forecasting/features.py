"""Past-only features for realized-volatility forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import pandas as pd

from rvol.data.market.daily import ordered_log_variance, validate_daily_columns
from rvol.domain.specifications import ComponentSpecification


@dataclass(frozen=True)
class HARFeatureBuilder:
    """Build HAR inputs at each origin, independently of future outcomes."""

    date_col: str = "date"
    rv_col: str = "log_rv"
    weekly_window: int = 5
    monthly_window: int = 22
    max_gap_days: int | None = 7

    feature_names: ClassVar[tuple[str, ...]] = ("rv_daily", "rv_weekly", "rv_monthly")

    def __post_init__(self) -> None:
        validate_daily_columns(self.date_col, self.rv_col)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in (
            self.weekly_window, self.monthly_window,
        )):
            raise ValueError("HAR window sizes must be positive integers")
        if self.monthly_window < self.weekly_window:
            raise ValueError("monthly_window must be at least weekly_window")
        if self.max_gap_days is not None and (
            isinstance(self.max_gap_days, bool) or not isinstance(self.max_gap_days, int)
            or self.max_gap_days < 1
        ):
            raise ValueError("max_gap_days must be a positive integer or None")

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
        ordered = ordered_log_variance(
            daily, date_col=self.date_col, rv_col=self.rv_col, max_gap_days=self.max_gap_days,
        )
        log_rv = ordered["log_rv"]
        return pd.DataFrame({
            "origin_date": ordered["date"],
            "rv_daily": log_rv,
            "rv_weekly": log_rv.rolling(self.weekly_window).mean(),
            "rv_monthly": log_rv.rolling(self.monthly_window).mean(),
        }).dropna().reset_index(drop=True)


def build_har_features(
    daily: pd.DataFrame, *,
    date_col: str = HARFeatureBuilder.date_col,
    rv_col: str = HARFeatureBuilder.rv_col,
    weekly_window: int = HARFeatureBuilder.weekly_window,
    monthly_window: int = HARFeatureBuilder.monthly_window,
    max_gap_days: int | None = HARFeatureBuilder.max_gap_days,
) -> pd.DataFrame:
    """Return past-only HAR inputs, including the latest observed session.

    Rolling windows count observed sessions. Warm-up rows without enough
    history are omitted; future labels are built and aligned separately.
    """
    return HARFeatureBuilder(
        date_col=date_col, rv_col=rv_col, weekly_window=weekly_window,
        monthly_window=monthly_window, max_gap_days=max_gap_days,
    )(daily)
