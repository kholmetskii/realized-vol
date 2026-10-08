"""Immutable configuration for forecast experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from rvol.domain.policies import WalkForwardStrategy


@dataclass(frozen=True)
class ExperimentConfig:
    """Date bounds, training requirements and execution strategy for one run."""

    min_train_size: int = 252
    forecast_start: date | None = None
    forecast_end: date | None = None
    strategy: WalkForwardStrategy = field(default_factory=WalkForwardStrategy)

    def __post_init__(self) -> None:
        if (
            isinstance(self.min_train_size, bool)
            or not isinstance(self.min_train_size, int)
            or self.min_train_size < 1
        ):
            raise ValueError("min_train_size must be a positive integer")
        capacity = self.strategy.training_window.max_size
        if capacity is not None and capacity < self.min_train_size:
            raise ValueError("training window size must be at least min_train_size")
        if (
            self.forecast_start is not None
            and self.forecast_end is not None
            and self.forecast_start > self.forecast_end
        ):
            raise ValueError("forecast_start must not be after forecast_end")
