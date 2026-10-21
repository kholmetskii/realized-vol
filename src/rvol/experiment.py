"""The realized-variance forecast comparison: prepare, forecast, evaluate.

Call these functions in order:

    prepared = prepare(daily)
    forecasts = forecast(prepared, benchmark_models(), start=..., end=...)
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
from rvol.evaluation import compare_models, forecast_losses, qlike, win_rates
from rvol.models import ExponentialSmoothing, ForecastModel, LinearRegression
from rvol.walkforward import smeared_forecasts, walk_forward

# HAR inputs (Corsi, 2009): the current log RV and its weekly and monthly means.
HAR_WINDOWS = {"log_rv_daily": 1, "log_rv_weekly": 5, "log_rv_monthly": 22}
HAR_FEATURES = tuple(HAR_WINDOWS)


def benchmark_models() -> tuple[ForecastModel, ...]:
    """The five models of the study: two forecasting rules with different regressors."""
    return (
        ForecastModel("historical_mean", LinearRegression()),  # no regressors: the mean
        ForecastModel("naive", ExponentialSmoothing(decay=0.0)),  # tomorrow equals today
        ForecastModel("EWMA", ExponentialSmoothing(decay=0.94)),  # RiskMetrics daily decay
        ForecastModel("AR1", LinearRegression(), ("log_rv_daily",)),
        ForecastModel("HAR", LinearRegression(), HAR_FEATURES),
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
    models: Sequence[ForecastModel],
    *,
    start: date | None = None,
    end: date | None = None,
    min_train_size: int = 252,
    asset: str = "default",
) -> pd.DataFrame:
    """Forecast next-session log RV walk-forward with every model and stack the results.

    Each model is refitted every session on at least ``min_train_size`` observed
    outcomes (252 sessions, about a year); ``start`` and ``end`` bound the target
    dates.
    """
    if not models:
        raise ValueError("at least one model is required")
    runs = [
        walk_forward(
            prepared, model, min_train_size=min_train_size, start=start, end=end, series=asset
        )
        for model in models
    ]
    return pd.concat(runs, ignore_index=True)


@dataclass(frozen=True)
class Comparison:
    """Variance-scale forecasts, their QLIKE losses and the per-model summary."""

    candidate: str
    forecasts: pd.DataFrame
    losses: pd.DataFrame
    summary: pd.DataFrame


def compare(
    log_forecasts: pd.DataFrame,
    *,
    evaluation_start: date,
    candidate: str = "HAR",
    smearing_min_errors: int = 20,
) -> Comparison:
    """Convert log forecasts to variance with Duan smearing and score them with QLIKE.

    Forecasts before ``evaluation_start`` only supply errors for the smearing
    factor. Every other model is tested against ``candidate`` with the one-sided
    Diebold-Mariano test.
    """
    if candidate not in set(log_forecasts["model"]):
        raise ValueError(f"candidate {candidate!r} has no forecasts")
    smeared = smeared_forecasts(log_forecasts, min_errors=smearing_min_errors)
    window = (
        smeared.loc[smeared["target_date"] >= pd.Timestamp(evaluation_start)]
        .sort_values(["model", "target_date"])
        .reset_index(drop=True)
    )
    if window.empty:
        raise ValueError("no forecasts on or after evaluation_start")
    losses = forecast_losses(window, qlike)
    return Comparison(candidate, window, losses, compare_models(losses, candidate))


def results_table(comparison: Comparison) -> pd.DataFrame:
    """One row per model, best first: mean QLIKE, the candidate's advantage,
    the DM p-value and the candidate's win rate."""
    candidate = comparison.candidate
    summary = comparison.summary.set_index("model")
    wins = win_rates(comparison.losses, candidate).set_index("baseline_model")
    table = pd.DataFrame(
        {
            "QLIKE": summary["mean_loss"],
            f"{candidate} lower by": 1 - summary.loc[candidate, "mean_loss"] / summary["mean_loss"],
            "DM p-value": summary["p_value"],
            f"{candidate} wins": wins["candidate_win_rate"],
        }
    )
    table.index.name = "model"
    return table.sort_values("QLIKE")
