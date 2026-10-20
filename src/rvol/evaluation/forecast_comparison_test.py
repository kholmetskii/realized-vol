"""The interface every pairwise forecast comparison test implements."""

from __future__ import annotations

from typing import Protocol

from rvol.evaluation.comparison_outcome import ComparisonOutcome
from rvol.models import FloatArray


class ForecastComparisonTest(Protocol):
    @property
    def name(self) -> str: ...

    def compare(
        self, baseline_loss: FloatArray, candidate_loss: FloatArray
    ) -> ComparisonOutcome: ...
