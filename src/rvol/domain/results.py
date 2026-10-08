"""Immutable results returned by forecasting application services."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from rvol.domain.config import ExperimentConfig


def _require_name(value: str, field: str) -> None:
    if not value.strip():
        raise ValueError(f"{field} must not be empty")


@dataclass(frozen=True)
class ForecastRecord:
    """One model's prediction for one target session."""

    model: str
    origin_date: date
    target_date: date
    n_train: int
    actual_log_rv: float
    predicted_log_rv: float
    fit_date: date | None = None
    train_start_date: date | None = None
    train_end_date: date | None = None

    def __post_init__(self) -> None:
        _require_name(self.model, "model")
        if self.origin_date >= self.target_date:
            raise ValueError("origin_date must precede target_date")
        if self.n_train < 1:
            raise ValueError("n_train must be positive")
        fit_dates = (self.train_start_date, self.train_end_date, self.fit_date)
        if any(value is not None for value in fit_dates):
            if any(value is None for value in fit_dates):
                raise ValueError("fit date and training boundaries must be provided together")
            assert self.train_start_date is not None
            assert self.train_end_date is not None
            assert self.fit_date is not None
            if not (
                self.train_start_date <= self.train_end_date <= self.fit_date <= self.origin_date
            ):
                raise ValueError("training boundaries must precede fitting and forecasting")
        if not all(math.isfinite(value) for value in (self.actual_log_rv, self.predicted_log_rv)):
            raise ValueError("forecast values must be finite")
        try:
            variances = (math.exp(self.actual_log_rv), math.exp(self.predicted_log_rv))
        except OverflowError as error:
            message = "log forecast values must represent finite positive variances"
            raise ValueError(message) from error
        if not all(math.isfinite(value) and value > 0 for value in variances):
            raise ValueError("log forecast values must represent finite positive variances")

    @property
    def actual_rv(self) -> float:
        """Actual realized variance derived from its canonical log value."""
        return math.exp(self.actual_log_rv)

    @property
    def predicted_rv(self) -> float:
        """Predicted realized variance derived from its canonical log value."""
        return math.exp(self.predicted_log_rv)


@dataclass(frozen=True)
class ExperimentResult:
    """Forecast records from one experiment run."""

    records: tuple[ForecastRecord, ...] = ()
    config: ExperimentConfig | None = None
    strategy_anchor: date | None = None

    @property
    def models(self) -> tuple[str, ...]:
        return tuple(sorted({record.model for record in self.records}))

    @property
    def n_forecasts(self) -> int:
        return len(self.records)
