"""The realized-variance forecast comparison: prepare, forecast, evaluate.

Call these functions in order:

    prepared = prepare(daily)
    forecasts = forecast(prepared, benchmark_models(), asset="EURUSD")
    comparison = compare(forecasts)
    results_table(comparison)

Models forecast next-session log RV. ``compare`` converts those forecasts to the
variance scale with Duan smearing and scores them with QLIKE and DM tests.

The constants below are the study design: every function uses them as its
defaults, so one asset's notebook and another's run the same study.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import pandas as pd

from rvol.data.preparation import (
    TrailingMeanFeatures,
    build_next_observation_targets,
    join_features,
    prepare_forecast_data,
)
from rvol.evaluation import compare_models, forecast_losses, qlike, win_rates
from rvol.models import ExponentialSmoothing, ForecastModel, LinearRegression
from rvol.walkforward import smeared_forecasts, walk_forward

#: The daily column the models forecast, one session ahead.
LOG_RV = "log_rv"

#: Consecutive sessions may be at most this many calendar days apart; a longer
#: gap is a data outage, not a weekend or holiday, and stops feature building.
MAX_GAP_DAYS = 7

#: HAR (Corsi, 2009) averages log RV over roughly a day, a week and a month of sessions.
HAR_WINDOWS = (1, 5, 22)
LOG_RV_MEANS = TrailingMeanFeatures(LOG_RV, HAR_WINDOWS, max_gap_days=MAX_GAP_DAYS)
HAR_FEATURES = LOG_RV_MEANS.columns

#: Every feature table the study builds; ``prepare`` joins them by origin.
FEATURE_TABLES = (LOG_RV_MEANS,)

#: Every fit uses at least this many observed outcomes, about one year of sessions.
MIN_TRAIN_SESSIONS = 252

#: The first and last target dates forecast. Forecasts before EVALUATION_START
#: only supply errors for the smearing factor; they are never scored.
FORECAST_START = date(2024, 10, 1)
EVALUATION_START = date(2025, 1, 1)
FORECAST_END = date(2026, 9, 30)

#: The model every other model is tested against.
CANDIDATE = "HAR"


def benchmark_models() -> tuple[ForecastModel, ...]:
    """The five models of the study: two forecasting rules with different regressors."""
    return (
        ForecastModel("historical_mean", LinearRegression()),  # no regressors: the mean
        ForecastModel("naive", ExponentialSmoothing(decay=0.0)),  # tomorrow equals today
        # 0.94 is RiskMetrics' daily decay for squared returns; here it smooths log RV.
        ForecastModel("EWMA", ExponentialSmoothing(decay=0.94)),
        ForecastModel("AR1", LinearRegression(), (LOG_RV_MEANS.column(1),)),
        ForecastModel("HAR", LinearRegression(), HAR_FEATURES),
    )


def prepare(daily: pd.DataFrame) -> pd.DataFrame:
    """Build every feature table and the next-session target from the daily table.

    One table holds all features, so every model forecasts the same dates.
    """
    return prepare_forecast_data(
        join_features([build(daily) for build in FEATURE_TABLES]),
        build_next_observation_targets(daily, value_col=LOG_RV),
    )


def forecast(
    prepared: pd.DataFrame,
    models: Sequence[ForecastModel],
    *,
    start: date | None = FORECAST_START,
    end: date | None = FORECAST_END,
    min_train_size: int = MIN_TRAIN_SESSIONS,
    asset: str = "default",
) -> pd.DataFrame:
    """Forecast next-session log RV walk-forward with every model and stack the results.

    Each model is refitted every session on at least ``min_train_size`` observed
    outcomes; ``start`` and ``end`` bound the target dates (None: no bound).
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
    evaluation_start: date = EVALUATION_START,
    candidate: str = CANDIDATE,
) -> Comparison:
    """Convert log forecasts to variance with Duan smearing and score them with QLIKE.

    Forecasts before ``evaluation_start`` only supply errors for the smearing
    factor. Every other model is tested against ``candidate`` with the one-sided
    Diebold-Mariano test.
    """
    if candidate not in set(log_forecasts["model"]):
        raise ValueError(f"candidate {candidate!r} has no forecasts")
    smeared = smeared_forecasts(log_forecasts)
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
