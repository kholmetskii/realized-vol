"""Immutable results returned by forecasting application services."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date


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
    actual_rv: float
    predicted_rv: float

    def __post_init__(self) -> None:
        _require_name(self.model, "model")
        if self.origin_date >= self.target_date:
            raise ValueError("origin_date must precede target_date")
        if self.n_train < 1:
            raise ValueError("n_train must be positive")
        values = (
            self.actual_log_rv,
            self.predicted_log_rv,
            self.actual_rv,
            self.predicted_rv,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("forecast values must be finite")
        if self.actual_rv <= 0 or self.predicted_rv <= 0:
            raise ValueError("variance values must be strictly positive")


@dataclass(frozen=True)
class ModelLossSummary:
    """Mean out-of-sample loss for one model and metric."""

    model: str
    metric: str
    n_obs: int
    mean_loss: float

    def __post_init__(self) -> None:
        _require_name(self.model, "model")
        _require_name(self.metric, "metric")
        if self.n_obs < 1:
            raise ValueError("n_obs must be positive")
        if not math.isfinite(self.mean_loss) or self.mean_loss < 0:
            raise ValueError("mean_loss must be finite and non-negative")


@dataclass(frozen=True)
class ExperimentResult:
    """Forecast records and loss summaries from one experiment run."""

    records: tuple[ForecastRecord, ...] = ()
    loss_summaries: tuple[ModelLossSummary, ...] = ()

    @property
    def models(self) -> tuple[str, ...]:
        names = {record.model for record in self.records}
        names.update(summary.model for summary in self.loss_summaries)
        return tuple(sorted(names))

    @property
    def n_forecasts(self) -> int:
        return len(self.records)
