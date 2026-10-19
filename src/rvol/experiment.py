"""The realized-variance forecast comparison: prepare, forecast, evaluate.

A notebook calls these functions in order:

    prepared = prepare(daily)
    forecasts = forecast(prepared, benchmark_models(), runner)
    comparison = compare(forecasts, evaluation_start=date(2025, 1, 1))
    results_table(comparison)

Models forecast next-session log RV. ``compare`` scores those log forecasts
with log-RV MSE and their Duan-smeared variance counterparts with QLIKE, on the
same dates, and runs robustness diagnostics for both scales.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import pandas as pd

from rvol.backtesting import WalkForwardRunner
from rvol.data.preparation import build_next_observation_targets, prepare_forecast_data
from rvol.data.volatility.features import VarianceFeatureBuilder
from rvol.evaluation import EvaluationResult, analyze_robustness, evaluate
from rvol.evaluation.volatility import LogMSEMetric, QLikeMetric, variance_forecasts
from rvol.models import AR1Model, EWMAModel, HARModel, HistoricalMeanModel, NaiveModel
from rvol.models.contracts import ForecastModel

HAR_FEATURES = ("log_rv_daily", "log_rv_weekly", "log_rv_monthly")

ModelSpec = tuple[ForecastModel, tuple[str, ...]]


def benchmark_models() -> tuple[ModelSpec, ...]:
    """The five models of the study, each with the feature columns it uses."""
    return (
        (HistoricalMeanModel(), ()),
        (NaiveModel(), ("log_rv_daily",)),
        (EWMAModel(), ()),
        (AR1Model(), ("log_rv_daily",)),
        (HARModel(), HAR_FEATURES),
    )


def prepare(daily: pd.DataFrame) -> pd.DataFrame:
    """Build every HAR feature and the next-session target from daily log RV.

    One table holds all features, so every model forecasts the same dates.
    """
    return prepare_forecast_data(
        VarianceFeatureBuilder(feature_names=HAR_FEATURES)(daily),
        build_next_observation_targets(daily, value_col="log_rv"),
    )


def forecast(
    prepared: pd.DataFrame,
    models: Sequence[ModelSpec],
    runner: WalkForwardRunner,
    *,
    run_id: str = "default",
) -> pd.DataFrame:
    """Run every model walk-forward and stack their log-RV forecasts."""
    if not models:
        raise ValueError("at least one model is required")
    runs = [
        runner.run(prepared, model=model, feature_columns=columns, run_id=run_id)
        for model, columns in models
    ]
    return pd.concat(runs, ignore_index=True)


@dataclass(frozen=True)
class ScaleResult:
    """Forecasts on one scale, their evaluation and robustness tables."""

    forecasts: pd.DataFrame
    evaluation: EvaluationResult
    robustness: dict[str, pd.DataFrame]


@dataclass(frozen=True)
class Comparison:
    """Log-scale and variance-scale results for the same forecast dates."""

    candidate: str
    log_rv: ScaleResult
    variance: ScaleResult


def compare(
    log_forecasts: pd.DataFrame,
    *,
    evaluation_start: date,
    candidate: str = "HAR",
    smearing_min_errors: int = 20,
) -> Comparison:
    """Score log forecasts with log-RV MSE and smeared variance forecasts with QLIKE.

    Forecasts before ``evaluation_start`` only supply errors for the smearing
    factor. Both scales are then scored on the same target dates, and every
    other model is tested against ``candidate`` with a one-sided DM test.
    """
    models = sorted(log_forecasts["model"].unique())
    if candidate not in models:
        raise ValueError(f"candidate {candidate!r} has no forecasts")
    pairs = [(model, candidate) for model in models if model != candidate]
    start = pd.Timestamp(evaluation_start)
    smeared = variance_forecasts(log_forecasts, min_errors=smearing_min_errors)

    def score(forecasts: pd.DataFrame, metric: LogMSEMetric | QLikeMetric) -> ScaleResult:
        window = (
            forecasts.loc[forecasts["target_date"] >= start]
            .sort_values(["model", "target_date"])
            .reset_index(drop=True)
        )
        if window.empty:
            raise ValueError("no forecasts on or after evaluation_start")
        evaluation = evaluate(window, metrics=(metric,), comparison_pairs=pairs)
        robustness = analyze_robustness(window, evaluation, candidate_model=candidate)
        return ScaleResult(window, evaluation, robustness)

    return Comparison(
        candidate=candidate,
        log_rv=score(log_forecasts, LogMSEMetric()),
        variance=score(smeared, QLikeMetric()),
    )


def results_table(comparison: Comparison) -> pd.DataFrame:
    """One row per model: mean losses, the candidate's advantage, DM p-values, win rate."""
    rows = []
    for scale, label in ((comparison.variance, "QLIKE"), (comparison.log_rv, "log-RV MSE")):
        means = scale.evaluation.summaries.set_index("model")["mean_loss"]
        tests = scale.evaluation.comparisons.set_index("baseline_model")["p_value"]
        advantage = 1 - means[comparison.candidate] / means
        rows.append(
            pd.DataFrame(
                {
                    label: means,
                    f"{label}: {comparison.candidate} lower by": advantage,
                    f"{label}: DM p-value": tests,
                }
            )
        )
    wins = comparison.variance.robustness["win_rates"].set_index("baseline_model")
    table = pd.concat(rows, axis=1)
    table[f"{comparison.candidate} wins (QLIKE)"] = wins["candidate_win_rate"]
    table.index.name = "model"
    return table.sort_values("QLIKE")
