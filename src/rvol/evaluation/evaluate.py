"""Score forecast tables: losses per metric, mean losses and pairwise tests."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from rvol.evaluation.diebold_mariano_test import DieboldMarianoTest
from rvol.evaluation.evaluation_result import EvaluationResult
from rvol.evaluation.forecast_comparison_test import ForecastComparisonTest
from rvol.evaluation.forecast_metric import ForecastMetric
from rvol.walkforward import CASE_COLUMNS, validated_forecasts

COMPARISON_COLUMNS = (
    "metric",
    "baseline_model",
    "candidate_model",
    "n_obs",
    "baseline_mean_loss",
    "candidate_mean_loss",
    "mean_loss_difference",
    "test_name",
    "statistic",
    "p_value",
    "test_parameters",
    "series",
)


def _validated_sample(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Require one series and a common sample for jointly evaluated models."""
    frame = validated_forecasts(forecasts)
    if frame["series"].nunique() != 1:
        raise ValueError("evaluate each series separately")
    aligned_columns = ["origin_date", "target_date", "actual"]
    groups = [group[aligned_columns].reset_index(drop=True) for _, group in frame.groupby("model")]
    if any(not group.equals(groups[0]) for group in groups[1:]):
        raise ValueError("all models must have identical dates and actual values")
    return frame


def evaluate(
    forecasts: pd.DataFrame,
    *,
    metrics: Sequence[ForecastMetric],
    comparison_pairs: Sequence[tuple[str, str]] = (),
    comparison_tests: Sequence[ForecastComparisonTest] | None = None,
) -> EvaluationResult:
    """Score every model with every metric and run every test on every pair.

    Pairs are (baseline, candidate). Without ``comparison_tests`` the one-sided
    Diebold-Mariano test is used.
    """
    frame = _validated_sample(forecasts)
    selected = tuple(metrics)
    names = [metric.name for metric in selected]
    if not names or len(set(names)) != len(names) or any(not name.strip() for name in names):
        raise ValueError("metrics must have non-empty unique names")
    pairs = tuple(comparison_pairs)
    models = set(frame["model"])
    if len(set(pairs)) != len(pairs) or any(
        base == candidate or {base, candidate}.difference(models) for base, candidate in pairs
    ):
        raise ValueError("comparison pairs must be unique distinct selected models")
    series = frame["series"].iloc[0]
    tests = (DieboldMarianoTest(),) if comparison_tests is None else tuple(comparison_tests)
    test_names = [test.name for test in tests]
    if (
        not test_names
        or len(set(test_names)) != len(test_names)
        or any(not name.strip() for name in test_names)
    ):
        raise ValueError("comparison tests must have non-empty unique names")
    losses, comparisons = [], []
    by_key: dict[tuple[str, str], np.ndarray] = {}
    for metric in selected:
        for model, group in frame.groupby("model", sort=True):
            actual = group["actual"].to_numpy(dtype="float64", copy=True)
            predicted = group["predicted"].to_numpy(dtype="float64", copy=True)
            values = np.asarray(metric.losses(actual, predicted), dtype="float64")
            if values.shape != (len(group),) or not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(
                    f"{metric.name} must return one finite non-negative loss per forecast"
                )
            by_key[(metric.name, str(model))] = values
            table = group[[*CASE_COLUMNS, "actual", "predicted"]].copy()
            table["metric"], table["loss"] = metric.name, values
            losses.append(table)
    for metric in selected:
        for baseline, candidate in pairs:
            base = by_key[(metric.name, baseline)]
            proposed = by_key[(metric.name, candidate)]
            for test in tests:
                outcome = test.compare(base.copy(), proposed.copy())
                comparisons.append(
                    {
                        "metric": metric.name,
                        "baseline_model": baseline,
                        "candidate_model": candidate,
                        "n_obs": len(base),
                        "baseline_mean_loss": float(np.mean(base)),
                        "candidate_mean_loss": float(np.mean(proposed)),
                        "mean_loss_difference": float(np.mean(base - proposed)),
                        "test_name": test.name,
                        "statistic": outcome.statistic,
                        "p_value": outcome.p_value,
                        "test_parameters": dict(outcome.parameters),
                        "series": series,
                    }
                )
    loss_table = pd.concat(losses, ignore_index=True)
    return EvaluationResult(
        loss_table,
        _summarize_losses(loss_table),
        pd.DataFrame(comparisons, columns=COMPARISON_COLUMNS),
    )


def _summarize_losses(losses: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (metric, model), group in losses.groupby(["metric", "model"], sort=False):
        mean = float(np.mean(group.loss.to_numpy(dtype="float64")))
        if not np.isfinite(mean):
            raise ValueError("mean loss must be finite")
        rows.append(
            {
                "model": model,
                "metric": metric,
                "n_obs": len(group),
                "mean_loss": mean,
                "series": group["series"].iloc[0],
            }
        )
    return pd.DataFrame(rows)
