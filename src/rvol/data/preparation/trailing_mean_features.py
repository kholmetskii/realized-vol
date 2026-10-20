"""Past-only features: trailing means of one series over chosen windows."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


class TrailingMeanFeatures:
    """Build one feature per named window: the mean of the latest observations.

    A window of 1 is the current value. Windows count observations, not calendar
    days, and each feature uses only values up to its origin. Rows before the
    longest window is filled are dropped; with no windows the full origin
    timeline is kept for models that learn from target history alone.
    """

    def __init__(
        self,
        windows: Mapping[str, int],
        *,
        date_col: str = "date",
        value_col: str = "value",
        max_gap_days: int | None = None,
    ) -> None:
        self.windows = dict(windows)
        self.date_col = date_col
        self.value_col = value_col
        self.max_gap_days = max_gap_days
        self._validate_settings()

    def _validate_settings(self) -> None:
        columns = (self.date_col, self.value_col)
        if (
            any(not isinstance(name, str) or not name.strip() for name in columns)
            or self.date_col == self.value_col
        ):
            raise ValueError("date and value columns must have distinct non-empty names")
        for name, window in self.windows.items():
            if not isinstance(name, str) or not name.strip() or name == "origin_date":
                raise ValueError("feature names must be non-empty and differ from 'origin_date'")
            if isinstance(window, bool) or not isinstance(window, int) or window < 1:
                raise ValueError(f"window for {name!r} must be a positive integer")
        if self.max_gap_days is not None and (
            isinstance(self.max_gap_days, bool)
            or not isinstance(self.max_gap_days, int)
            or self.max_gap_days < 1
        ):
            raise ValueError("max_gap_days must be a positive integer or None")

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        self._validate_settings()
        ordered = self._ordered(data)
        values = ordered["value"]
        features = pd.DataFrame({"origin_date": ordered["date"]})
        for name, window in self.windows.items():
            features[name] = values if window == 1 else values.rolling(window).mean()
        return features.dropna().reset_index(drop=True)

    def _ordered(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return canonical date and value columns, one row per date, in date order."""
        if not data.columns.is_unique:
            raise ValueError("input columns must be unique")
        missing = {self.date_col, self.value_col}.difference(data.columns)
        if missing:
            raise ValueError(f"missing input columns: {', '.join(sorted(missing))}")
        dates = pd.to_datetime(data[self.date_col], errors="coerce")
        values = pd.to_numeric(data[self.value_col], errors="coerce")
        valid = dates.notna() & np.isfinite(values.to_numpy(dtype="float64", na_value=np.nan))
        if not valid.all():
            invalid = int((~valid).sum())
            raise ValueError(f"{invalid} rows have invalid dates or {self.value_col!r} values")
        ordered = pd.DataFrame({"date": dates, "value": values.astype("float64")})
        ordered = ordered.sort_values("date").reset_index(drop=True)
        if ordered["date"].duplicated().any():
            raise ValueError("input must contain one row per date")
        if self.max_gap_days is not None:
            gap_days = ordered["date"].diff().dt.total_seconds() / (24 * 60 * 60)
            large = gap_days > self.max_gap_days
            if large.any():
                position = int(np.flatnonzero(large.to_numpy())[0])
                previous = ordered["date"].iloc[position - 1].date()
                following = ordered["date"].iloc[position].date()
                days = int(gap_days.iloc[position])
                raise ValueError(f"input contains a {days}-day gap from {previous} to {following}")
        return ordered
