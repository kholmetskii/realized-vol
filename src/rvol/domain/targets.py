"""Explicit meanings of session-based forecast targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from rvol.domain.specifications import ComponentSpecification

TargetKind = Literal["session", "cumulative"]


@dataclass(frozen=True)
class TargetHorizon:
    """Variance at session t+h, or the sum over sessions t+1 through t+h."""

    sessions: int = 1
    kind: TargetKind = "session"

    def __post_init__(self) -> None:
        if (
            isinstance(self.sessions, bool) or not isinstance(self.sessions, int)
            or self.sessions < 1
        ):
            raise ValueError("horizon sessions must be a positive integer")
        if self.kind not in ("session", "cumulative"):
            raise ValueError("target kind must be session or cumulative")

    @property
    def specification(self) -> ComponentSpecification:
        return ComponentSpecification(
            "forecast_target", f"{type(self).__module__}.{type(self).__qualname__}",
            parameters=(("sessions", self.sessions), ("kind", self.kind)),
        )
