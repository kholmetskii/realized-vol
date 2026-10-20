"""The realized-variance forecast comparison: prepare, forecast, evaluate.

Call these functions in order:

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

from rvol.data.preparation import build_next_observation_targets, prepare_forecast_data
from rvol.data.volatility import VarianceFeatureBuilder
from rvol.evaluation import (
    EvaluationResult,
    ForecastComparisonTest,
    ForecastMetric,
    evaluate,
)
from rvol.evaluation.robustness import Diagnostic, analyze_robustness, standard_diagnostics
from rvol.evaluation.volatility import LogMSEMetric, QLikeMetric, variance_forecasts
from rvol.models import (
    AR1Model,
    EWMAModel,
    ForecastModel,
    HARModel,
    HistoricalMeanModel,
    NaiveModel,
)
from rvol.walkforward import WalkForwardRunner

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
    asset: str = "default",
) -> pd.DataFrame:
    """Run every model walk-forward and stack their log-RV forecasts."""
    if not models:
        raise ValueError("at least one model is required")
    runs = [
        runner.run(prepared, model=model, feature_columns=columns, asset=asset)
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
    log_metrics: Sequence[ForecastMetric] | None = None,
    variance_metrics: Sequence[ForecastMetric] | None = None,
    comparison_tests: Sequence[ForecastComparisonTest] | None = None,
    diagnostics: Sequence[Diagnostic] | None = None,
) -> Comparison:
    """Score log forecasts and their smeared variance counterparts on the same dates.

    Forecasts before ``evaluation_start`` only supply errors for the smearing
    factor. Every other model is tested against ``candidate``. Defaults: log-RV
    MSE on the log scale, QLIKE on the variance scale, the one-sided DM test and
    the standard diagnostics for ``candidate``.
    """
    log_selected = (LogMSEMetric(),) if log_metrics is None else tuple(log_metrics)
    variance_selected = (QLikeMetric(),) if variance_metrics is None else tuple(variance_metrics)
    shared = {m.name for m in log_selected}.intersection(m.name for m in variance_selected)
    if shared:
        raise ValueError(f"metric names must differ between scales: {', '.join(sorted(shared))}")
    selected_diagnostics = (
        standard_diagnostics(candidate) if diagnostics is None else tuple(diagnostics)
    )
    models = sorted(log_forecasts["model"].unique())
    if candidate not in models:
        raise ValueError(f"candidate {candidate!r} has no forecasts")
    pairs = [(model, candidate) for model in models if model != candidate]
    start = pd.Timestamp(evaluation_start)
    smeared = variance_forecasts(log_forecasts, min_errors=smearing_min_errors)

    def score(forecasts: pd.DataFrame, metrics: tuple[ForecastMetric, ...]) -> ScaleResult:
        window = (
            forecasts.loc[forecasts["target_date"] >= start]
            .sort_values(["model", "target_date"])
            .reset_index(drop=True)
        )
        if window.empty:
            raise ValueError("no forecasts on or after evaluation_start")
        evaluation = evaluate(
            window, metrics=metrics, comparison_pairs=pairs, comparison_tests=comparison_tests
        )
        robustness = analyze_robustness(window, evaluation, selected_diagnostics)
        return ScaleResult(window, evaluation, robustness)

    return Comparison(
        candidate=candidate,
        log_rv=score(log_forecasts, log_selected),
        variance=score(smeared, variance_selected),
    )


def results_table(comparison: Comparison) -> pd.DataFrame:
    """One row per model: mean losses, the candidate's advantage, test p-values, win rate.

    Columns repeat for every metric (variance scale first) and every test; win
    rates come from the variance scale when that diagnostic was run. Rows are
    sorted by the first variance-scale metric.
    """
    candidate = comparison.candidate
    columns: dict[str, pd.Series] = {}
    for scale in (comparison.variance, comparison.log_rv):
        summaries = scale.evaluation.summaries
        tests = scale.evaluation.comparisons
        for metric in summaries["metric"].unique():
            means = summaries.loc[summaries["metric"] == metric].set_index("model")["mean_loss"]
            columns[metric] = means
            columns[f"{metric}: {candidate} lower by"] = 1 - means[candidate] / means
            tested = tests.loc[tests["metric"] == metric]
            for test_name in tested["test_name"].unique():
                p_values = tested.loc[tested["test_name"] == test_name]
                columns[f"{metric}: {test_name} p-value"] = p_values.set_index("baseline_model")[
                    "p_value"
                ]
    table = pd.DataFrame(columns)
    wins = comparison.variance.robustness.get("win_rates")
    if wins is not None:
        for metric, group in wins.groupby("metric", sort=False):
            table[f"{candidate} wins ({metric})"] = group.set_index("baseline_model")[
                "candidate_win_rate"
            ]
    table.index.name = "model"
    return table.sort_values(comparison.variance.evaluation.summaries["metric"].iloc[0])
