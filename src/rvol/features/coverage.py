"""Coverage and quality checks for derived daily volatility datasets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SessionGap:
    """A calendar gap between consecutive observed trading sessions."""

    previous_date: pd.Timestamp
    next_date: pd.Timestamp
    calendar_days: int


@dataclass(frozen=True)
class YearCoverage:
    """Number of unique sessions observed in one calendar year."""

    year: int
    sessions: int


@dataclass(frozen=True)
class CoverageAudit:
    """Summary of daily dataset coverage and detected quality problems."""

    passed: bool
    n_rows: int
    n_sessions: int
    start_date: pd.Timestamp | None
    end_date: pd.Timestamp | None
    sessions_by_year: tuple[YearCoverage, ...]
    median_observation_count: float
    duplicate_dates: int
    dates_out_of_order: bool
    invalid_dates: int
    invalid_rv: int
    invalid_observation_counts: int
    unusual_observation_sessions: int
    large_gaps: tuple[SessionGap, ...]


def audit_daily_dataset(
    daily: pd.DataFrame,
    *,
    date_col: str = "date",
    rv_col: str = "rv_5min",
    observation_count_col: str = "observation_count",
    expected_observations: int = 288,
    observation_tolerance: float = 0.10,
    max_gap_days: int = 7,
) -> CoverageAudit:
    """Audit ordering, uniqueness, completeness, and numerical validity.

    Observation counts within ``observation_tolerance`` of the expected value
    are accepted. The default range includes the one-hour change around US
    daylight-saving transitions while still detecting substantially incomplete
    sessions.
    """
    required = {date_col, rv_col, observation_count_col}
    missing = required.difference(daily.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"daily data is missing required columns: {names}")
    if daily.empty:
        raise ValueError("daily data must contain at least one row")
    if expected_observations < 1:
        raise ValueError("expected_observations must be positive")
    if not 0 <= observation_tolerance < 1:
        raise ValueError("observation_tolerance must satisfy 0 <= value < 1")
    if max_gap_days < 1:
        raise ValueError("max_gap_days must be positive")

    dates = pd.to_datetime(daily[date_col], errors="coerce")
    rv = pd.to_numeric(daily[rv_col], errors="coerce").astype("float64")
    counts = pd.to_numeric(daily[observation_count_col], errors="coerce").astype("float64")

    invalid_dates = int(dates.isna().sum())
    valid_dates = dates[dates.notna()]
    dates_out_of_order = not valid_dates.is_monotonic_increasing
    duplicate_dates = int(valid_dates.duplicated().sum())
    unique_dates = valid_dates.drop_duplicates().sort_values().reset_index(drop=True)

    gaps: list[SessionGap] = []
    for previous, following in zip(unique_dates[:-1], unique_dates[1:], strict=True):
        days = int((following - previous) / pd.Timedelta(days=1))
        if days > max_gap_days:
            gaps.append(SessionGap(previous, following, days))

    years, counts_by_year = np.unique(
        unique_dates.dt.year.to_numpy(dtype="int64"),
        return_counts=True,
    )
    sessions_by_year = tuple(
        YearCoverage(int(year), int(sessions))
        for year, sessions in zip(years, counts_by_year, strict=True)
    )

    invalid_rv = int((~np.isfinite(rv) | (rv <= 0)).sum())
    invalid_counts_mask = ~np.isfinite(counts) | (counts <= 0)
    invalid_observation_counts = int(invalid_counts_mask.sum())
    lower = expected_observations * (1 - observation_tolerance)
    upper = expected_observations * (1 + observation_tolerance)
    unusual_counts = (~invalid_counts_mask) & ((counts < lower) | (counts > upper))
    unusual_observation_sessions = int(unusual_counts.sum())
    finite_counts = counts[np.isfinite(counts) & (counts > 0)]
    median_count = float(finite_counts.median()) if not finite_counts.empty else float("nan")

    passed = not any((
        duplicate_dates,
        dates_out_of_order,
        invalid_dates,
        invalid_rv,
        invalid_observation_counts,
        unusual_observation_sessions,
        gaps,
    ))
    return CoverageAudit(
        passed=passed,
        n_rows=len(daily),
        n_sessions=len(unique_dates),
        start_date=unique_dates.iloc[0] if len(unique_dates) else None,
        end_date=unique_dates.iloc[-1] if len(unique_dates) else None,
        sessions_by_year=sessions_by_year,
        median_observation_count=median_count,
        duplicate_dates=duplicate_dates,
        dates_out_of_order=dates_out_of_order,
        invalid_dates=invalid_dates,
        invalid_rv=invalid_rv,
        invalid_observation_counts=invalid_observation_counts,
        unusual_observation_sessions=unusual_observation_sessions,
        large_gaps=tuple(gaps),
    )
