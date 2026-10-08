"""Composable robustness checks for fixed out-of-sample forecast results."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date

import numpy as np

from rvol.domain import (
    ComponentSpecification,
    ExperimentResult,
    ForecastComparisonTest,
    ForecastMetric,
    TargetHorizon,
)
from rvol.domain.identity import RunIdentity
from rvol.domain.specifications import ParameterValue
from rvol.evaluation.comparison import DieboldMarianoTest
from rvol.evaluation.diagnostics import (
    DiagnosticCheck,
    DiagnosticContext,
    DiagnosticResult,
    DiagnosticValue,
    validate_checks,
)
from rvol.evaluation.evaluator import ForecastEvaluator


def _specification(
    check: object, name: str, parameters: tuple[tuple[str, ParameterValue], ...] = (),
) -> ComponentSpecification:
    return ComponentSpecification(
        name, f"{type(check).__module__}.{type(check).__qualname__}", parameters=parameters,
    )


@dataclass(frozen=True)
class LagSensitivityCheck:
    hac_lags: tuple[int, ...] = tuple(range(6))
    test_factory: Callable[[int], ForecastComparisonTest] = DieboldMarianoTest

    def __post_init__(self) -> None:
        if not self.hac_lags or len(set(self.hac_lags)) != len(self.hac_lags) or any(
            isinstance(lag, bool) or not isinstance(lag, int) or lag < 0 for lag in self.hac_lags
        ):
            raise ValueError("hac_lags must be non-empty, unique, non-negative integers")

    @property
    def specification(self) -> ComponentSpecification:
        return _specification(self, "hac_sensitivity", (
            ("hac_lags", ",".join(map(str, self.hac_lags))),
        ))

    def run(self, context: DiagnosticContext) -> DiagnosticResult:
        rows = []
        components = []
        columns = (
            "metric", "baseline_model", "candidate_model", "n_obs", "baseline_mean_loss",
            "candidate_mean_loss", "mean_loss_difference", "test_name", "statistic", "p_value",
            "test_parameters", "hac_lags",
        )
        for lag in self.hac_lags:
            test = self.test_factory(lag)
            evaluation = ForecastEvaluator(
                metrics=context.metrics, comparison_pairs=context.comparison_pairs,
                comparison_test=test,
            ).evaluate(context.experiment)
            assert evaluation.comparison_test is not None
            components.append((f"lag_{lag}", evaluation.comparison_test))
            for comparison in evaluation.comparisons:
                row = asdict(comparison) | {
                    "test_parameters": json.dumps(dict(comparison.test_parameters), sort_keys=True),
                    "hac_lags": lag,
                }
                rows.append(tuple(row[column] for column in columns))
        rows.sort(key=lambda row: (row[0], row[1], row[2], row[-1]))
        return DiagnosticResult(
            self.specification, "Newey–West lag sensitivity", columns, tuple(rows),
            (("Metric", "{metric}"), ("Baseline", "{baseline_model}"),
             ("Candidate", "{candidate_model}"), ("NW lags", "{hac_lags}"),
             ("Statistic", "{statistic:.3f}"), ("p-value", "{p_value:.4g}")),
            components=tuple(components),
        )


@dataclass(frozen=True)
class MonthlyLossCheck:
    @property
    def specification(self) -> ComponentSpecification:
        return _specification(self, "monthly_losses")

    def run(self, context: DiagnosticContext) -> DiagnosticResult:
        values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        for loss in context.evaluation.losses:
            values[(loss.target_date.strftime("%Y-%m"), loss.metric, loss.model)].append(loss.loss)
        rows = tuple(
            (month, model, metric, len(losses), float(np.mean(losses)))
            for (month, metric, model), losses in sorted(values.items())
        )
        return DiagnosticResult(
            self.specification, "Monthly forecast losses",
            ("month", "model", "metric", "n_obs", "mean_loss"), rows,
            (("Month", "{month}"), ("Metric", "{metric}"), ("Model", "{model}"),
             ("n", "{n_obs}"), ("Mean loss", "{mean_loss:.6g}")),
        )


@dataclass(frozen=True)
class WinRateCheck:
    tie_rtol: float = 1e-5
    tie_atol: float = 1e-8

    def __post_init__(self) -> None:
        if any(not np.isfinite(value) or value < 0 for value in (self.tie_rtol, self.tie_atol)):
            raise ValueError("tie tolerances must be finite and non-negative")

    @property
    def specification(self) -> ComponentSpecification:
        return _specification(self, "win_rates", (
            ("tie_rtol", self.tie_rtol), ("tie_atol", self.tie_atol),
        ))

    def run(self, context: DiagnosticContext) -> DiagnosticResult:
        losses: dict[tuple[str, str], dict[date, float]] = defaultdict(dict)
        for loss in context.evaluation.losses:
            losses[(loss.metric, loss.model)][loss.target_date] = loss.loss
        rows = []
        for metric in context.metrics:
            for baseline, candidate_model in context.comparison_pairs:
                candidate_losses = losses[(metric.name, candidate_model)]
                baseline_losses = losses[(metric.name, baseline)]
                dates = sorted(candidate_losses)
                candidate = np.array([candidate_losses[day] for day in dates])
                benchmark = np.array([baseline_losses[day] for day in dates])
                tie_mask = np.isclose(candidate, benchmark, rtol=self.tie_rtol, atol=self.tie_atol)
                ties = int(np.count_nonzero(tie_mask))
                wins = int(np.count_nonzero((candidate < benchmark) & ~tie_mask))
                rows.append((
                    metric.name, baseline, candidate_model, len(dates),
                    wins, ties, wins / len(dates),
                ))
        rows.sort(key=lambda row: (row[0], row[1]))
        return DiagnosticResult(
            self.specification, "Candidate session-level win rates",
            ("metric", "baseline_model", "candidate_model", "n_obs", "candidate_wins", "ties",
             "candidate_win_rate"), tuple(rows),
            (("Metric", "{metric}"), ("Baseline", "{baseline_model}"),
             ("Candidate", "{candidate_model}"), ("Wins", "{candidate_wins}/{n_obs}"),
             ("Ties", "{ties}"), ("Win rate", "{candidate_win_rate:.1%}")),
        )


@dataclass(frozen=True)
class LargestErrorsCheck:
    top_errors: int = 5

    def __post_init__(self) -> None:
        if isinstance(self.top_errors, bool) or not isinstance(self.top_errors, int) \
                or self.top_errors < 1:
            raise ValueError("top_errors must be a positive integer")

    @property
    def specification(self) -> ComponentSpecification:
        return _specification(self, "largest_errors", (("top_errors", self.top_errors),))

    def run(self, context: DiagnosticContext) -> DiagnosticResult:
        rows: list[tuple[DiagnosticValue, ...]] = []
        for model in context.experiment.models:
            records = sorted(
                (record for record in context.experiment.records if record.model == model),
                key=lambda record: (
                    -abs(record.actual_log_rv - record.predicted_log_rv), record.target_date,
                ),
            )
            rows.extend((
                model, record.target_date, record.actual_log_rv, record.predicted_log_rv,
                abs(record.actual_log_rv - record.predicted_log_rv),
            ) for record in records[:self.top_errors])
        return DiagnosticResult(
            self.specification, "Largest absolute log-RV errors",
            ("model", "target_date", "actual_log_rv", "predicted_log_rv", "absolute_log_error"),
            tuple(rows),
            (("Model", "{model}"), ("Target", "{target_date}"),
             ("Actual log RV", "{actual_log_rv:.6f}"), ("Predicted", "{predicted_log_rv:.6f}"),
             ("Absolute error", "{absolute_log_error:.6f}")),
        )


@dataclass(frozen=True)
class RobustnessResult:
    checks: tuple[DiagnosticResult, ...]
    identity: RunIdentity = field(default_factory=RunIdentity)
    target_horizon: TargetHorizon = field(default_factory=TargetHorizon)

    def check(self, name: str) -> DiagnosticResult:
        for result in self.checks:
            if result.name == name:
                return result
        raise KeyError(name)


class ForecastRobustnessAnalyzer:
    """Run injected diagnostics against one shared evaluation without model fitting."""

    def __init__(
        self, checks: Sequence[DiagnosticCheck], *, metrics: Sequence[ForecastMetric] | None = None,
        candidate_model: str | None = None, baseline_models: Sequence[str] = (),
    ) -> None:
        self.checks = validate_checks(checks)
        self.evaluator = ForecastEvaluator(metrics=metrics)
        self.candidate_model = candidate_model
        self.baseline_models = tuple(baseline_models)
        if candidate_model is not None and not candidate_model.strip():
            raise ValueError("candidate_model must not be empty")
        if len(set(self.baseline_models)) != len(self.baseline_models):
            raise ValueError("baseline_models must be unique")
        if candidate_model in self.baseline_models:
            raise ValueError("candidate_model must not also be a baseline")

    def analyze(self, experiment: ExperimentResult) -> RobustnessResult:
        context = DiagnosticContext(
            experiment, self.evaluator.evaluate(experiment), self.evaluator.metrics,
            self.candidate_model, self.baseline_models,
        )
        results = []
        for check in self.checks:
            snapshot = check.specification
            result = check.run(context)
            if result.specification != snapshot:
                raise ValueError("diagnostic result must match its configured specification")
            results.append(result)
        return RobustnessResult(
            tuple(results), identity=experiment.identity, target_horizon=experiment.target_horizon,
        )
