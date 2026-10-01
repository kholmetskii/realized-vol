"""Robustness diagnostics for out-of-sample forecast comparisons."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np

from rvol.domain import ExperimentResult
from rvol.domain.contracts import ForecastMetric
from rvol.evaluation.evaluator import ForecastEvaluator, PairwiseComparison
from rvol.evaluation.metrics import LogMSEMetric, QLikeMetric


@dataclass(frozen=True)
class MonthlyLossSummary:
    """Mean loss for one model and calendar month."""

    month: str
    model: str
    metric: str
    n_obs: int
    mean_loss: float


@dataclass(frozen=True)
class WinRateSummary:
    """Share of target sessions where a candidate has lower loss."""

    metric: str
    baseline_model: str
    candidate_model: str
    n_obs: int
    candidate_wins: int
    ties: int
    candidate_win_rate: float


@dataclass(frozen=True)
class ForecastErrorRecord:
    """One unusually large forecast error measured on the log-RV scale."""

    model: str
    target_date: date
    actual_log_rv: float
    predicted_log_rv: float
    absolute_log_error: float


@dataclass(frozen=True)
class RobustnessResult:
    """Sensitivity tests and descriptive forecast diagnostics."""

    hac_comparisons: tuple[PairwiseComparison, ...]
    monthly_losses: tuple[MonthlyLossSummary, ...]
    win_rates: tuple[WinRateSummary, ...]
    largest_errors: tuple[ForecastErrorRecord, ...]


class ForecastRobustnessAnalyzer:
    """Run predefined checks without changing or refitting model specifications."""

    def __init__(
        self,
        *,
        candidate_model: str = "HAR",
        baseline_models: Sequence[str] = ("naive", "AR1", "historical_mean"),
        metrics: Sequence[ForecastMetric] | None = None,
        hac_lags: Sequence[int] = tuple(range(6)),
        top_errors: int = 5,
    ) -> None:
        baselines = tuple(baseline_models)
        lags = tuple(hac_lags)
        selected_metrics = (
            (QLikeMetric(), LogMSEMetric())
            if metrics is None
            else tuple(metrics)
        )
        if not candidate_model.strip():
            raise ValueError("candidate_model must not be empty")
        if not baselines or len(baselines) != len(set(baselines)):
            raise ValueError("baseline_models must be non-empty and unique")
        if candidate_model in baselines:
            raise ValueError("candidate_model must not also be a baseline")
        if not selected_metrics:
            raise ValueError("at least one metric is required")
        if not lags or len(lags) != len(set(lags)) or any(lag < 0 for lag in lags):
            raise ValueError("hac_lags must be non-empty, unique, and non-negative")
        if top_errors < 1:
            raise ValueError("top_errors must be positive")
        self.candidate_model = candidate_model
        self.baseline_models = baselines
        self.metrics = selected_metrics
        self.hac_lags = lags
        self.top_errors = top_errors

    def analyze(self, experiment: ExperimentResult) -> RobustnessResult:
        """Calculate sensitivity and descriptive checks for one fixed experiment."""
        pairs = tuple(
            (baseline, self.candidate_model)
            for baseline in self.baseline_models
        )
        evaluations = [
            ForecastEvaluator(
                metrics=self.metrics,
                comparison_pairs=pairs,
                hac_lags=lag,
            ).evaluate(experiment)
            for lag in self.hac_lags
        ]
        reference = evaluations[0]
        hac_comparisons = tuple(
            comparison
            for evaluation in evaluations
            for comparison in evaluation.comparisons
        )

        monthly_values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        losses_by_key: dict[tuple[str, str], dict[date, float]] = defaultdict(dict)
        for loss in reference.losses:
            month = loss.target_date.strftime("%Y-%m")
            monthly_values[(month, loss.metric, loss.model)].append(loss.loss)
            losses_by_key[(loss.metric, loss.model)][loss.target_date] = loss.loss
        monthly_losses = tuple(
            MonthlyLossSummary(
                month=month,
                metric=metric,
                model=model,
                n_obs=len(values),
                mean_loss=float(np.mean(values)),
            )
            for (month, metric, model), values in sorted(monthly_values.items())
        )

        win_rates: list[WinRateSummary] = []
        for metric in self.metrics:
            candidate_losses = losses_by_key[(metric.name, self.candidate_model)]
            for baseline in self.baseline_models:
                baseline_losses = losses_by_key[(metric.name, baseline)]
                if set(baseline_losses) != set(candidate_losses):
                    raise ValueError("candidate and baseline loss dates must align")
                dates = sorted(candidate_losses)
                candidate = np.array([candidate_losses[item] for item in dates])
                benchmark = np.array([baseline_losses[item] for item in dates])
                tie_mask = np.isclose(candidate, benchmark)
                ties = int(np.count_nonzero(tie_mask))
                wins = int(np.count_nonzero((candidate < benchmark) & ~tie_mask))
                win_rates.append(WinRateSummary(
                    metric=metric.name,
                    baseline_model=baseline,
                    candidate_model=self.candidate_model,
                    n_obs=len(dates),
                    candidate_wins=wins,
                    ties=ties,
                    candidate_win_rate=wins / len(dates),
                ))

        records_by_model = defaultdict(list)
        for record in experiment.records:
            records_by_model[record.model].append(record)
        largest_errors: list[ForecastErrorRecord] = []
        for model in sorted(records_by_model):
            ranked = sorted(
                records_by_model[model],
                key=lambda record: abs(record.actual_log_rv - record.predicted_log_rv),
                reverse=True,
            )
            largest_errors.extend(
                ForecastErrorRecord(
                    model=model,
                    target_date=record.target_date,
                    actual_log_rv=record.actual_log_rv,
                    predicted_log_rv=record.predicted_log_rv,
                    absolute_log_error=abs(record.actual_log_rv - record.predicted_log_rv),
                )
                for record in ranked[: self.top_errors]
            )

        return RobustnessResult(
            hac_comparisons=hac_comparisons,
            monthly_losses=monthly_losses,
            win_rates=tuple(win_rates),
            largest_errors=tuple(largest_errors),
        )
