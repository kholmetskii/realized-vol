"""Immutable configuration for forecast experiments."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Generic, TypeVar

Settings = TypeVar("Settings", covariant=True)


@dataclass(frozen=True)
class ExperimentConfig(Generic[Settings]):
    """Shared reporting bounds and typed settings for the selected executor."""

    execution: Settings
    forecast_start: date | None = None
    forecast_end: date | None = None

    def __post_init__(self) -> None:
        if (
            self.forecast_start is not None
            and self.forecast_end is not None
            and self.forecast_start > self.forecast_end
        ):
            raise ValueError("forecast_start must not be after forecast_end")
