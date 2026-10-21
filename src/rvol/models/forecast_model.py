"""A forecasting model: a named forecasting rule applied to chosen regressors."""

from __future__ import annotations

from dataclasses import dataclass

from rvol.models.forecast_rule import ForecastRule


@dataclass(frozen=True)
class ForecastModel:
    """The triple that defines a model: its name, its rule and its regressors.

    Two models with the same rule differ only in their regressors; for example
    AR(1) and HAR are both least-squares regressions.
    """

    name: str
    rule: ForecastRule
    regressors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("a model needs a non-empty name")
        regressors = self.regressors
        if (
            not isinstance(regressors, tuple)
            or any(not isinstance(column, str) or not column.strip() for column in regressors)
            or len(set(regressors)) != len(regressors)
        ):
            raise ValueError("regressors must be a tuple of unique non-empty column names")
