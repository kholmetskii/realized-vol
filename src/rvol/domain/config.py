"""Immutable configuration for forecast experiments."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class ExperimentConfig:
    """Date bounds and training requirements for a walk-forward experiment."""

    min_train_size: int = 252
    forecast_start: date | None = None
    forecast_end: date | None = None

    def __post_init__(self) -> None:
        if self.min_train_size < 1:
            raise ValueError("min_train_size must be positive")
        if (
            self.forecast_start is not None
            and self.forecast_end is not None
            and self.forecast_start > self.forecast_end
        ):
            raise ValueError("forecast_start must not be after forecast_end")
