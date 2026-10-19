"""Score forecast tables directly, without experiment objects or specifications."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from rvol.backtesting.results import CASE_COLUMNS, validated_forecasts
from rvol.evaluation.comparison import DieboldMarianoTest
from rvol.evaluation.contracts import ForecastComparisonTest, ForecastMetric
from rvol.evaluation.metrics import MSEMetric

META_COLUMNS = ("run_id", "fold_id")
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
    *META_COLUMNS,
)


class EvaluationResult:
    """Three ordinary tables: individual losses, means and requested comparisons."""

    def __init__(
        self,
        losses: pd.DataFrame,
        summaries: pd.DataFrame,
        comparisons: pd.DataFrame,
    ) -> None:
        self.losses = losses
        self.summaries = summaries
        self.comparisons = comparisons


def validated_evaluation_sample(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Require one run/fold and a common sample for jointly evaluated models."""
    frame = validated_forecasts(forecasts)
    if len(frame[list(META_COLUMNS)].drop_duplicates()) != 1:
        raise ValueError("evaluate each run/fold separately")
    aligned_columns = ["origin_date", "target_date", "actual"]
    groups = [group[aligned_columns].reset_index(drop=True) for _, group in frame.groupby("model")]
    if any(not group.equals(groups[0]) for group in groups[1:]):
        raise ValueError("all models must have identical dates and actual values")
    return frame


def evaluate(
    forecasts: pd.DataFrame,
    *,
    metrics: Sequence[ForecastMetric] | None = None,
    comparison_pairs: Sequence[tuple[str, str]] = (),
    comparison_test: ForecastComparisonTest | None = None,
) -> EvaluationResult:
    frame = validated_evaluation_sample(forecasts)
    selected = (MSEMetric(),) if metrics is None else tuple(metrics)
    names = [metric.name for metric in selected]
    if not names or len(set(names)) != len(names) or any(not name.strip() for name in names):
        raise ValueError("metrics must have non-empty unique names")
    pairs = tuple(comparison_pairs)
    models = set(frame["model"])
    if len(set(pairs)) != len(pairs) or any(
        base == candidate or {base, candidate}.difference(models) for base, candidate in pairs
    ):
        raise ValueError("comparison pairs must be unique distinct selected models")
    metadata = {name: frame[name].iloc[0] for name in META_COLUMNS}
    test = comparison_test if comparison_test is not None else DieboldMarianoTest()
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
                    **metadata,
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
                **{name: group[name].iloc[0] for name in META_COLUMNS},
            }
        )
    return pd.DataFrame(rows)


def validate_evaluation(forecasts: pd.DataFrame, evaluation: EvaluationResult) -> pd.DataFrame:
    """Check that an evaluation was computed from exactly these forecasts.

    Reports call this so a stale evaluation cannot be saved or plotted next to
    different forecasts. Losses, summaries and comparisons are not recomputed:
    evaluate() is tested on its own.
    """
    frame = validated_evaluation_sample(forecasts)
    columns = [*CASE_COLUMNS, "actual", "predicted"]
    losses = evaluation.losses
    if losses.empty or not {"metric", "loss", *columns}.issubset(losses.columns):
        raise ValueError("evaluation must contain one loss per forecast and metric")
    expected = frame[columns].reset_index(drop=True)
    for _, group in losses.groupby("metric", sort=False):
        recorded = group.sort_values(["model", "target_date"])[columns].reset_index(drop=True)
        try:
            pd.testing.assert_frame_equal(recorded, expected, check_dtype=False, check_exact=True)
        except AssertionError as error:
            raise ValueError("evaluation does not match the supplied forecasts") from error
    return frame
