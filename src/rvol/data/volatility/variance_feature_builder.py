"""Past-only features for realized-volatility forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd


class VarianceFeatureBuilder:
    """Build selected log-variance inputs using only history at each origin.

    log_rv_daily is the current log variance. log_rv_weekly and log_rv_monthly
    are arithmetic means of daily log variances over the configured windows,
    rather than logarithms of mean variances.

    Empty feature selections retain the full origin timeline for models that
    learn from target history alone. Warm-up depends only on selected inputs.
    """

    def __init__(
        self,
        date_col: str = "date",
        rv_col: str = "log_rv",
        weekly_window: int = 5,
        monthly_window: int = 22,
        max_gap_days: int | None = 7,
        feature_names: tuple[str, ...] = ("log_rv_daily", "log_rv_weekly", "log_rv_monthly"),
    ) -> None:
        self.date_col = date_col
        self.rv_col = rv_col
        self.weekly_window = weekly_window
        self.monthly_window = monthly_window
        self.max_gap_days = max_gap_days
        self.feature_names = feature_names
        self._validate_settings()

    def _validate_settings(self) -> None:
        validate_daily_columns(self.date_col, self.rv_col)
        if len(self.feature_names) != len(set(self.feature_names)):
            raise ValueError("feature names must be unique")
        unknown = set(self.feature_names).difference(
            ("log_rv_daily", "log_rv_weekly", "log_rv_monthly")
        )
        if unknown:
            raise ValueError(f"unknown variance features: {', '.join(sorted(unknown))}")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in (
                self.weekly_window,
                self.monthly_window,
            )
        ):
            raise ValueError("HAR window sizes must be positive integers")
        if self.monthly_window < self.weekly_window:
            raise ValueError("monthly_window must be at least weekly_window")
        if self.max_gap_days is not None and (
            isinstance(self.max_gap_days, bool)
            or not isinstance(self.max_gap_days, int)
            or self.max_gap_days < 1
        ):
            raise ValueError("max_gap_days must be a positive integer or None")

    def __call__(self, daily: pd.DataFrame) -> pd.DataFrame:
        self._validate_settings()
        ordered = ordered_log_variance(
            daily,
            date_col=self.date_col,
            rv_col=self.rv_col,
            max_gap_days=self.max_gap_days,
        )
        log_rv = ordered["log_rv"]
        features = pd.DataFrame({"origin_date": ordered["date"]})
        for name in self.feature_names:
            if name == "log_rv_daily":
                features[name] = log_rv
            else:
                window = self.weekly_window if name == "log_rv_weekly" else self.monthly_window
                features[name] = log_rv.rolling(window).mean()
        return features.dropna().reset_index(drop=True)


def ordered_log_variance(
    daily: pd.DataFrame,
    *,
    date_col: str,
    rv_col: str,
    max_gap_days: int | None = None,
) -> pd.DataFrame:
    """Return a copy with canonical columns, ordered by unique observed session."""
    validate_daily_columns(date_col, rv_col)
    if not daily.columns.is_unique:
        raise ValueError("daily data columns must be unique")
    missing = {date_col, rv_col}.difference(daily.columns)
    if missing:
        raise ValueError(f"daily data is missing required columns: {', '.join(sorted(missing))}")
    dates = pd.to_datetime(daily[date_col], errors="coerce")
    values = pd.to_numeric(daily[rv_col], errors="coerce")
    valid = dates.notna() & np.isfinite(values.to_numpy(dtype="float64", na_value=np.nan))
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
            raise ValueError(f"daily data contains a {days}-day gap from {previous} to {following}")
    return ordered


def validate_daily_columns(date_col: str, rv_col: str) -> None:
    if any(not isinstance(name, str) or not name.strip() for name in (date_col, rv_col)):
        raise ValueError("daily column names must be non-empty strings")
    if date_col == rv_col:
        raise ValueError("date and log-variance columns must be distinct")
