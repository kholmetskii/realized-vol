"""Forecast with every model, then score the forecasts on the evaluation period."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import pandas as pd

from rvol.evaluation import compare_models, forecast_losses, win_rates
from rvol.models import ForecastModel
from rvol.study.design import (
    CANDIDATE,
    EVALUATION_START,
    FORECAST_END,
    FORECAST_START,
    LOSS,
    LOSS_LABEL,
    MIN_TRAIN_SESSIONS,
)
from rvol.walkforward import smeared_forecasts, walk_forward


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
    """Variance-scale forecasts, their losses and the per-model summary."""

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
    """Convert log forecasts to variance with Duan smearing and score them with ``LOSS``.

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
    losses = forecast_losses(window, LOSS)
    return Comparison(candidate, window, losses, compare_models(losses, candidate))


def results_table(comparison: Comparison) -> pd.DataFrame:
    """One row per model, best first: mean loss, the candidate's advantage,
    the DM p-value and the candidate's win rate."""
    candidate = comparison.candidate
    summary = comparison.summary.set_index("model")
    wins = win_rates(comparison.losses, candidate).set_index("baseline_model")
    table = pd.DataFrame(
        {
            LOSS_LABEL: summary["mean_loss"],
            f"{candidate} lower by": 1 - summary.loc[candidate, "mean_loss"] / summary["mean_loss"],
            "DM p-value": summary["p_value"],
            f"{candidate} wins": wins["candidate_win_rate"],
        }
    )
    table.index.name = "model"
    return table.sort_values(LOSS_LABEL)
