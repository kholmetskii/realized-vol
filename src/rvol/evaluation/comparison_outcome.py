"""The result of a statistical forecast comparison."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ComparisonOutcome:
    """A statistical test's value, p-value and resolved options (e.g. HAC lags)."""

    statistic: float
    p_value: float
    parameters: dict[str, str | int | float | bool | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if math.isnan(self.statistic):
            raise ValueError("comparison statistic must not be NaN")
        if not math.isfinite(self.p_value) or not 0 <= self.p_value <= 1:
            raise ValueError("comparison p_value must lie between zero and one")
        if any(not isinstance(key, str) or not key.strip() for key in self.parameters):
            raise ValueError("comparison parameter names must be non-empty strings")
        if any(
            not isinstance(v, (str, int, float, bool, type(None)))
            or isinstance(v, float)
            and not math.isfinite(v)
            for v in self.parameters.values()
        ):
            raise ValueError("comparison parameters must be finite JSON scalar values")
