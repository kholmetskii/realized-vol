"""Immutable descriptions of configured experiment components."""

from __future__ import annotations

import math
from dataclasses import dataclass

ParameterValue = str | int | float | bool | None


@dataclass(frozen=True)
class ComponentSpecification:
    """Identify an implementation, its ordered features and scalar settings.

    Parameter pairs use tuples so a saved specification cannot change when a
    caller mutates a configuration dictionary. Implementations own the values;
    reporting never infers settings from class names or model-specific rules.
    """

    name: str
    implementation: str
    feature_names: tuple[str, ...] = ()
    parameters: tuple[tuple[str, ParameterValue], ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.implementation.strip():
            raise ValueError("component name and implementation must not be empty")
        if len(self.feature_names) != len(set(self.feature_names)):
            raise ValueError("component feature names must be unique")
        if any(not name.strip() for name in self.feature_names):
            raise ValueError("component feature names must not be empty")
        names = [name for name, _ in self.parameters]
        if len(names) != len(set(names)) or any(not name.strip() for name in names):
            raise ValueError("component parameter names must be non-empty and unique")
        for _, value in self.parameters:
            if not isinstance(value, (str, int, float, bool, type(None))):
                raise ValueError("component parameters must be JSON scalar values")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("component parameters must be finite")


@dataclass(frozen=True)
class ExperimentSpecification:
    """Feature preparation and model settings selected for one experiment."""

    features: ComponentSpecification
    models: tuple[ComponentSpecification, ...]

    def __post_init__(self) -> None:
        names = [model.name for model in self.models]
        if not names or len(names) != len(set(names)):
            raise ValueError("model specifications must be non-empty and unique")
        required = {name for model in self.models for name in model.feature_names}
        missing = required.difference(self.features.feature_names)
        if missing:
            missing_names = ", ".join(sorted(missing))
            raise ValueError(f"feature specification is missing model inputs: {missing_names}")
