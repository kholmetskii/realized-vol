"""Contracts and immutable tables for independently selectable diagnostics."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Protocol, runtime_checkable

from rvol.domain import ComponentSpecification, ExperimentResult, ForecastMetric
from rvol.forecasting.evaluation.evaluator import EvaluationResult

DiagnosticValue = str | int | float | bool | date | None


@dataclass(frozen=True)
class DiagnosticContext:
    experiment: ExperimentResult
    evaluation: EvaluationResult
    metrics: tuple[ForecastMetric, ...]
    candidate_model: str | None
    baseline_models: tuple[str, ...]

    @property
    def comparison_pairs(self) -> tuple[tuple[str, str], ...]:
        if self.candidate_model is None or not self.baseline_models:
            raise ValueError("comparison diagnostics require a candidate and baselines")
        names = {self.candidate_model, *self.baseline_models}
        missing = names.difference(self.experiment.models)
        if missing:
            raise ValueError(f"diagnostic references unknown models: {', '.join(sorted(missing))}")
        return tuple((baseline, self.candidate_model) for baseline in self.baseline_models)


@dataclass(frozen=True)
class DiagnosticResult:
    """A named CSV table and display templates consumed by a generic renderer.

    Display templates use Python format fields referring to CSV column names.
    Checks own presentation labels and numeric formats as well as calculations.
    """

    specification: ComponentSpecification
    title: str
    columns: tuple[str, ...]
    rows: tuple[tuple[DiagnosticValue, ...], ...]
    report_columns: tuple[tuple[str, str], ...]
    components: tuple[tuple[str, ComponentSpecification], ...] = ()

    def __post_init__(self) -> None:
        if re.fullmatch(r"[a-z][a-z0-9_]*", self.name) is None:
            raise ValueError("diagnostic name must be a safe lowercase file identifier")
        if not self.title.strip() or not self.columns or not self.report_columns:
            raise ValueError("diagnostic title, columns and report columns must not be empty")
        if len(set(self.columns)) != len(self.columns):
            raise ValueError("diagnostic column names must be unique")
        if {"run_id", "fold_id", "horizon_sessions", "target_kind"}.intersection(self.columns):
            raise ValueError("diagnostic columns must not override run identity or target metadata")
        if any(len(row) != len(self.columns) for row in self.rows):
            raise ValueError("diagnostic rows must match their columns")
        names = [name for name, _ in self.components]
        if len(names) != len(set(names)):
            raise ValueError("diagnostic component names must be unique")

    @property
    def name(self) -> str:
        return self.specification.name

    def records(self) -> tuple[dict[str, DiagnosticValue], ...]:
        return tuple(dict(zip(self.columns, row, strict=True)) for row in self.rows)


@runtime_checkable
class DiagnosticCheck(Protocol):
    @property
    def specification(self) -> ComponentSpecification: ...

    def run(self, context: DiagnosticContext) -> DiagnosticResult: ...


def validate_checks(checks: Sequence[DiagnosticCheck]) -> tuple[DiagnosticCheck, ...]:
    selected = tuple(checks)
    names = [check.specification.name for check in selected]
    if not names or len(names) != len(set(names)):
        raise ValueError("diagnostic checks must be non-empty and have unique names")
    return selected
