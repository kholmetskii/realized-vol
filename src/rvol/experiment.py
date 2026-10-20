"""The realized-variance forecast comparison: prepare, forecast, evaluate.

Call these functions in order:

    prepared = prepare(daily)
    forecasts = forecast(prepared, benchmark_models(), runner)
    comparison = compare(forecasts, evaluation_start=date(2025, 1, 1))
    results_table(comparison)

Models forecast next-session log RV. ``compare`` converts those forecasts to the
variance scale with Duan smearing and scores them with QLIKE and DM tests.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import pandas as pd

from rvol.data.preparation import (
    TrailingMeanFeatures,
    build_next_observation_targets,
    prepare_forecast_data,
)
from rvol.evaluation import (
    EvaluationResult,
    ForecastComparisonTest,
    ForecastMetric,
    QLikeMetric,
    evaluate,
    win_rates,
)
from rvol.models import (
    AR1Model,
    EWMAModel,
    ForecastModel,
    HARModel,
    HistoricalMeanModel,
    NaiveModel,
)
from rvol.walkforward import WalkForwardRunner, smeared_forecasts

# HAR inputs (Corsi, 2009): the current log RV and its weekly and monthly means.
HAR_WINDOWS = {"log_rv_daily": 1, "log_rv_weekly": 5, "log_rv_monthly": 22}
HAR_FEATURES = tuple(HAR_WINDOWS)

ModelSpec = tuple[ForecastModel, tuple[str, ...]]


def benchmark_models() -> tuple[ModelSpec, ...]:
    """The five models of the study, each with the feature columns it uses."""
    return (
        (HistoricalMeanModel(), ()),
        (NaiveModel(), ("log_rv_daily",)),
        (EWMAModel(decay=0.94), ()),  # RiskMetrics daily decay
        (AR1Model(), ("log_rv_daily",)),
        (HARModel(), HAR_FEATURES),
    )


def prepare(daily: pd.DataFrame) -> pd.DataFrame:
    """Build every HAR feature and the next-session target from daily log RV.

    One table holds all features, so every model forecasts the same dates.
    """
    return prepare_forecast_data(
        TrailingMeanFeatures(HAR_WINDOWS, value_col="log_rv", max_gap_days=7)(daily),
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
        runner.run(prepared, model=model, feature_columns=columns, series=asset)
        for model, columns in models
    ]
    return pd.concat(runs, ignore_index=True)


@dataclass(frozen=True)
class Comparison:
    """Variance-scale forecasts and their evaluation."""

    candidate: str
    forecasts: pd.DataFrame
    evaluation: EvaluationResult


def compare(
    log_forecasts: pd.DataFrame,
    *,
    evaluation_start: date,
    candidate: str = "HAR",
    smearing_min_errors: int = 20,
    metrics: Sequence[ForecastMetric] | None = None,
    comparison_tests: Sequence[ForecastComparisonTest] | None = None,
) -> Comparison:
    """Convert log forecasts to variance with Duan smearing and score them.

    Forecasts before ``evaluation_start`` only supply errors for the smearing
    factor. Every other model is tested against ``candidate``. Defaults: QLIKE
    and the one-sided DM test.
    """
    selected_metrics = (QLikeMetric(),) if metrics is None else tuple(metrics)
    models = sorted(log_forecasts["model"].unique())
    if candidate not in models:
        raise ValueError(f"candidate {candidate!r} has no forecasts")
    pairs = [(model, candidate) for model in models if model != candidate]
    smeared = smeared_forecasts(log_forecasts, min_errors=smearing_min_errors)
    window = (
        smeared.loc[smeared["target_date"] >= pd.Timestamp(evaluation_start)]
        .sort_values(["model", "target_date"])
        .reset_index(drop=True)
    )
    if window.empty:
        raise ValueError("no forecasts on or after evaluation_start")
    evaluation = evaluate(
        window, metrics=selected_metrics, comparison_pairs=pairs, comparison_tests=comparison_tests
    )
    return Comparison(candidate, window, evaluation)


def results_table(comparison: Comparison) -> pd.DataFrame:
    """One row per model: mean losses, the candidate's advantage, test p-values, win rate.

    Columns repeat for every metric and every test. Rows are sorted by the first
    metric.
    """
    candidate = comparison.candidate
    summaries = comparison.evaluation.summaries
    tests = comparison.evaluation.comparisons
    columns: dict[str, pd.Series] = {}
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
    wins = win_rates(comparison.evaluation.losses, candidate)
    for metric, group in wins.groupby("metric", sort=False):
        table[f"{candidate} wins ({metric})"] = group.set_index("baseline_model")[
            "candidate_win_rate"
        ]
    table.index.name = "model"
    return table.sort_values(summaries["metric"].iloc[0])
