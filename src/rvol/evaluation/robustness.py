"""Optional diagnostics calculated from already evaluated forecast tables."""

from __future__ import annotations

import json
from collections.abc import Sequence

import numpy as np
import pandas as pd

from rvol.evaluation.comparison import DieboldMarianoTest
from rvol.evaluation.evaluator import EvaluationResult, validate_evaluation


def analyze_robustness(
    forecasts: pd.DataFrame,
    evaluation: EvaluationResult,
    *,
    candidate_model: str | None = None,
    baseline_models: Sequence[str] | None = None,
    checks: Sequence[str] | None = None,
    hac_lags: Sequence[int] = tuple(range(6)),
    top_errors: int = 5,
    tie_rtol: float = 1e-5,
    tie_atol: float = 1e-8,
) -> dict[str, pd.DataFrame]:
    """Return named tables; this function never fits models or recomputes losses."""
    frame = validate_evaluation(forecasts, evaluation)
    models = set(frame.model)
    baselines = tuple(
        sorted(models - {candidate_model})
        if baseline_models is None and candidate_model is not None
        else (baseline_models or ())
    )
    if candidate_model is not None and candidate_model not in models:
        raise ValueError("unknown candidate model")
    if (
        len(set(baselines)) != len(baselines)
        or set(baselines).difference(models)
        or candidate_model in baselines
    ):
        raise ValueError("baselines must be unique selected models other than the candidate")
    selected = (
        tuple(checks)
        if checks is not None
        else (
            ("hac_sensitivity", "win_rates", "monthly_losses", "largest_errors")
            if candidate_model is not None and baselines
            else ("monthly_losses", "largest_errors")
        )
    )
    allowed = {"hac_sensitivity", "win_rates", "monthly_losses", "largest_errors"}
    if not selected or len(set(selected)) != len(selected) or set(selected).difference(allowed):
        raise ValueError("checks must be non-empty unique diagnostic names")
    if {"hac_sensitivity", "win_rates"}.intersection(selected) and (
        candidate_model is None or not baselines
    ):
        raise ValueError("comparison diagnostics require a candidate and baselines")
    if isinstance(top_errors, bool) or not isinstance(top_errors, int) or top_errors < 1:
        raise ValueError("top_errors must be a positive integer")
    if any(not np.isfinite(v) or v < 0 for v in (tie_rtol, tie_atol)):
        raise ValueError("tie tolerances must be finite and non-negative")
    lags = tuple(hac_lags)
    if "hac_sensitivity" in selected and (
        not lags
        or len(set(lags)) != len(lags)
        or any(isinstance(lag, bool) or not isinstance(lag, int) or lag < 0 for lag in lags)
    ):
        raise ValueError("hac_lags must be non-empty unique non-negative integers")
    losses = evaluation.losses
    result = {}
    metadata = frame[["run_id", "fold_id"]].iloc[0].to_dict()
    for name in selected:
        if name == "monthly_losses":
            table = (
                losses.assign(month=losses.target_date.dt.strftime("%Y-%m"))
                .groupby(
                    ["month", "metric", "model"],
                    sort=True,
                )
                .loss.agg(n_obs="size", mean_loss="mean")
                .reset_index()
            )
        elif name == "largest_errors":
            table = frame.assign(absolute_error=(frame.actual - frame.predicted).abs())
            table = (
                table.sort_values(
                    ["model", "absolute_error", "target_date"], ascending=[True, False, True]
                )
                .groupby("model", sort=False)
                .head(top_errors)
            )
            table = table[
                ["model", "target_date", "actual", "predicted", "absolute_error"]
            ].reset_index(drop=True)
        else:
            rows = []
            for metric, group in losses.groupby("metric", sort=True):
                pivot = group.pivot(
                    index="target_date", columns="model", values="loss"
                ).sort_index()
                for baseline in baselines:
                    candidate = pivot[candidate_model].to_numpy(dtype="float64")
                    benchmark = pivot[baseline].to_numpy(dtype="float64")
                    row = {
                        "metric": metric,
                        "baseline_model": baseline,
                        "candidate_model": candidate_model,
                        "n_obs": len(pivot),
                    }
                    if name == "win_rates":
                        ties = np.isclose(candidate, benchmark, rtol=tie_rtol, atol=tie_atol)
                        wins = int(np.count_nonzero((candidate < benchmark) & ~ties))
                        rows.append(
                            {
                                **row,
                                "candidate_wins": wins,
                                "ties": int(ties.sum()),
                                "candidate_win_rate": wins / len(pivot),
                            }
                        )
                    else:
                        for lag in lags:
                            test = DieboldMarianoTest(hac_lags=lag)
                            outcome = test.compare(benchmark, candidate)
                            rows.append(
                                {
                                    **row,
                                    "baseline_mean_loss": float(np.mean(benchmark)),
                                    "candidate_mean_loss": float(np.mean(candidate)),
                                    "mean_loss_difference": float(np.mean(benchmark - candidate)),
                                    "test_name": test.name,
                                    "statistic": outcome.statistic,
                                    "p_value": outcome.p_value,
                                    "test_parameters": json.dumps(
                                        outcome.parameters, sort_keys=True
                                    ),
                                    "hac_lags": lag,
                                }
                            )
            table = pd.DataFrame(rows)
        for key, value in metadata.items():
            table[key] = value
        result[name] = table
    return result


def cumulative_loss_advantage(
    evaluation: EvaluationResult, *, metric: str, candidate: str
) -> pd.DataFrame:
    """Running sum of each baseline's loss minus the candidate's, by target date.

    One column per baseline model, in name order. A rising line means the
    candidate keeps forecasting better than that baseline.
    """
    losses = (
        evaluation.losses.loc[evaluation.losses["metric"] == metric]
        .pivot(index="target_date", columns="model", values="loss")
        .sort_index()
    )
    if losses.empty:
        raise ValueError(f"evaluation has no {metric} losses")
    if candidate not in losses.columns:
        raise ValueError(f"evaluation has no losses for candidate {candidate!r}")
    baselines = [model for model in sorted(losses.columns) if model != candidate]
    return losses[baselines].sub(losses[candidate], axis=0).cumsum()
