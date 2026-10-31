"""Past-only features: trailing means of one series over chosen windows."""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from itertools import pairwise

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TrailingMeanFeatures:
    """Trailing means of one series, one feature per window.

    The feature ``<value_col>_mean_<w>`` at an origin is the mean of the latest
    ``w`` observations up to and including it, so a window of 1 is the current
    value. Windows count observations, not calendar days, and list the features
    in increasing order. Rows before the longest window is filled are dropped;
    with no windows the full origin timeline is kept for models that learn from
    target history alone.
    """

    value_col: str
    windows: tuple[int, ...]
    _: KW_ONLY
    date_col: str = "date"
    max_gap_days: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "windows", tuple(self.windows))
        columns = (self.date_col, self.value_col)
        if (
            any(not isinstance(name, str) or not name.strip() for name in columns)
            or self.date_col == self.value_col
        ):
            raise ValueError("date and value columns must have distinct non-empty names")
        if any(isinstance(w, bool) or not isinstance(w, int) or w < 1 for w in self.windows):
            raise ValueError("windows must be positive integers")
        if any(a >= b for a, b in pairwise(self.windows)):
            raise ValueError("windows must be strictly increasing")
        if self.max_gap_days is not None and (
            isinstance(self.max_gap_days, bool)
            or not isinstance(self.max_gap_days, int)
            or self.max_gap_days < 1
        ):
            raise ValueError("max_gap_days must be a positive integer or None")

    def column(self, window: int) -> str:
        """The name of the feature averaging the latest ``window`` observations."""
        if window not in self.windows:
            raise ValueError(f"no window of {window}; windows are {self.windows}")
        return f"{self.value_col}_mean_{window}"

    @property
    def columns(self) -> tuple[str, ...]:
        """Every feature name, in window order."""
        return tuple(self.column(window) for window in self.windows)

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        ordered = self._ordered(data)
        values = ordered["value"]
        features = pd.DataFrame({"origin_date": ordered["date"]})
        for window, name in zip(self.windows, self.columns, strict=True):
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
