"""Composable training windows and refit schedules for forecast experiments."""

from __future__ import annotations

from dataclasses import dataclass, field

from rvol.domain.contracts import RefitSchedule, TrainingWindowPolicy
from rvol.domain.specifications import ComponentSpecification, ExecutionSpecification


def _require_integer(value: int, name: str, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")


@dataclass(frozen=True)
class ExpandingWindow:
    """Select all eligible training observations available at the origin."""

    @property
    def specification(self) -> ComponentSpecification:
        return ComponentSpecification(
            name="expanding", implementation=f"{type(self).__module__}.{type(self).__qualname__}",
        )

    @property
    def max_size(self) -> None:
        return None

    def select(self, n_available: int) -> slice:
        _require_integer(n_available, "n_available", 0)
        return slice(0, n_available)


@dataclass(frozen=True)
class RollingWindow:
    """Select up to size of the most recent eligible training observations.

    Size counts observations rather than calendar days. Minimum training
    requirements are enforced by the runner before fitting a selected window.
    """

    size: int

    def __post_init__(self) -> None:
        _require_integer(self.size, "size", 1)

    @property
    def specification(self) -> ComponentSpecification:
        return ComponentSpecification(
            name="rolling", implementation=f"{type(self).__module__}.{type(self).__qualname__}",
            parameters=(("size", self.size),),
        )

    @property
    def max_size(self) -> int:
        return self.size

    def select(self, n_available: int) -> slice:
        _require_integer(n_available, "n_available", 0)
        return slice(max(0, n_available - self.size), n_available)


@dataclass(frozen=True)
class EveryNSessions:
    """Refit at origin indices 0, interval, 2 * interval, and so on.

    The anchor is the first eligible origin in the full dataset. Report date
    bounds do not reset this schedule. Every call is independent of prior calls.
    """

    interval: int = 1

    def __post_init__(self) -> None:
        _require_integer(self.interval, "interval", 1)

    @property
    def specification(self) -> ComponentSpecification:
        return ComponentSpecification(
            name="every_n_sessions",
            implementation=f"{type(self).__module__}.{type(self).__qualname__}",
            parameters=(("interval", self.interval),),
        )

    def should_refit(self, eligible_origin_index: int) -> bool:
        _require_integer(eligible_origin_index, "eligible_origin_index", 0)
        return eligible_origin_index % self.interval == 0


@dataclass(frozen=True)
class WalkForwardStrategy:
    """Select a training window and refit schedule independently.

    Defaults describe expanding history and refitting at every eligible origin.
    The runner owns eligibility, date filtering and fitted model state.
    """

    training_window: TrainingWindowPolicy = field(default_factory=ExpandingWindow)
    retrain: RefitSchedule = field(default_factory=EveryNSessions)

    @property
    def specification(self) -> ExecutionSpecification:
        return ExecutionSpecification(
            implementation=f"{type(self).__module__}.{type(self).__qualname__}",
            training_window=self.training_window.specification,
            retrain=self.retrain.specification,
        )
