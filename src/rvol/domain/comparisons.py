"""Generic statistical outcomes returned by injected comparison tests."""

from __future__ import annotations

import math
from dataclasses import dataclass

from rvol.domain.specifications import ComponentSpecification, ParameterValue


@dataclass(frozen=True)
class ComparisonOutcome:
    """Statistic, one-sided p-value and settings actually used by a test.

    Infinite statistics are allowed for degenerate tests with zero estimated
    sampling variance. Parameters are immutable, finite JSON scalar values.
    """

    statistic: float
    p_value: float
    parameters: tuple[tuple[str, ParameterValue], ...] = ()

    def __post_init__(self) -> None:
        if math.isnan(self.statistic):
            raise ValueError("comparison statistic must not be NaN")
        if not math.isfinite(self.p_value) or not 0 <= self.p_value <= 1:
            raise ValueError("comparison p_value must lie between zero and one")
        ComponentSpecification("comparison", "outcome", parameters=self.parameters)
