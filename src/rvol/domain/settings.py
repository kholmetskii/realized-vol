"""Typed settings owned and validated by individual execution methods."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from rvol.domain.contracts import RefitSchedule, TrainingWindowPolicy
from rvol.domain.policies import EveryNSessions, ExpandingWindow
from rvol.domain.specifications import ComponentSpecification, ExecutionSpecification


def _validate_minimum(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("min_train_size must be a positive integer")


@dataclass(frozen=True)
class WalkForwardSettings:
    """Training requirements, history selection and refitting for walk-forward."""

    training_window: TrainingWindowPolicy = field(default_factory=ExpandingWindow)
    retrain: RefitSchedule = field(default_factory=EveryNSessions)
    min_train_size: int = 252

    def __post_init__(self) -> None:
        _validate_minimum(self.min_train_size)
        capacity = self.training_window.max_size
        if capacity is not None and capacity < self.min_train_size:
            raise ValueError("training window size must be at least min_train_size")

    @property
    def specification(self) -> ExecutionSpecification:
        return ExecutionSpecification(
            component=ComponentSpecification(
                name="walk_forward",
                implementation=f"{type(self).__module__}.{type(self).__qualname__}",
                parameters=(("min_train_size", self.min_train_size),),
            ),
            policies=(
                ("training_window", self.training_window.specification),
                ("retrain", self.retrain.specification),
            ),
        )


@dataclass(frozen=True)
class HoldoutSettings:
    """Fit once using labels available at train_end; optionally update state."""

    train_end: date
    min_train_size: int = 252
    update_observations: bool = False

    def __post_init__(self) -> None:
        if type(self.train_end) is not date:
            raise ValueError("train_end must be a date")
        _validate_minimum(self.min_train_size)
        if not isinstance(self.update_observations, bool):
            raise ValueError("update_observations must be a boolean")

    @property
    def specification(self) -> ExecutionSpecification:
        return ExecutionSpecification(ComponentSpecification(
            name="holdout",
            implementation=f"{type(self).__module__}.{type(self).__qualname__}",
            parameters=(
                ("train_end", self.train_end.isoformat()),
                ("min_train_size", self.min_train_size),
                ("update_observations", self.update_observations),
            ),
        ))
