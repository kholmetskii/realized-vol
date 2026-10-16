"""Immutable configuration for forecast experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from rvol.domain.identity import RunIdentity
from rvol.domain.settings import HoldoutSettings, WalkForwardSettings


@dataclass(frozen=True)
class ExperimentConfig:
    """Reporting bounds, instrument label and execution settings for a run."""

    execution: WalkForwardSettings | HoldoutSettings
    forecast_start: date | None = None
    forecast_end: date | None = None
    identity: RunIdentity = field(default_factory=RunIdentity)
    instrument: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.execution, (WalkForwardSettings, HoldoutSettings)):
            raise TypeError("execution must be WalkForwardSettings or HoldoutSettings")
        if self.instrument is not None and (
            not isinstance(self.instrument, str) or not self.instrument.strip()
        ):
            raise ValueError("instrument must be a non-empty string or None")
        if (
            self.forecast_start is not None
            and self.forecast_end is not None
            and self.forecast_start > self.forecast_end
        ):
            raise ValueError("forecast_start must not be after forecast_end")
