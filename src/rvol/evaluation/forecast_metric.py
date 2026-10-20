"""The interface every forecast loss metric implements."""

from __future__ import annotations

from typing import Protocol

from rvol.models import FloatArray


class ForecastMetric(Protocol):
    @property
    def name(self) -> str: ...

    def losses(self, actual: FloatArray, predicted: FloatArray) -> FloatArray: ...
