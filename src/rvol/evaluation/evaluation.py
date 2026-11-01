"""Score log-scale forecasts on an evaluation period, and summarize the scores."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from rvol.evaluation.robustness import win_rates
from rvol.evaluation.scoring import Loss, compare_models, forecast_losses
from rvol.walkforward import smeared_forecasts


@dataclass(frozen=True)
class Evaluation:
    """Original-scale forecasts on the evaluation period, their losses, and the
    one-sided Diebold-Mariano test of the candidate against every other model."""

    candidate: str
    loss_name: str
    forecasts: pd.DataFrame
    losses: pd.DataFrame
    summary: pd.DataFrame


def evaluate(
    log_forecasts: pd.DataFrame,
    *,
    candidate: str,
    loss: Loss,
    evaluation_start: date,
    loss_name: str | None = None,
) -> Evaluation:
    """Smear log forecasts back to the original scale and score those from
    ``evaluation_start`` on.

    Earlier forecasts only supply errors for the smearing factor. Every other
    model is tested against ``candidate``.
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
    losses = forecast_losses(window, loss)
    name = loss_name if loss_name is not None else str(getattr(loss, "__name__", "loss"))
    return Evaluation(candidate, name, window, losses, compare_models(losses, candidate))


def results_table(evaluation: Evaluation) -> pd.DataFrame:
    """One row per model, best first: mean loss, the candidate's advantage,
    the DM p-value and the candidate's win rate."""
    candidate = evaluation.candidate
    summary = evaluation.summary.set_index("model")
    wins = win_rates(evaluation.losses, candidate).set_index("baseline_model")
    table = pd.DataFrame(
        {
            evaluation.loss_name: summary["mean_loss"],
            f"{candidate} lower by": 1 - summary.loc[candidate, "mean_loss"] / summary["mean_loss"],
            "DM p-value": summary["p_value"],
            f"{candidate} wins": wins["candidate_win_rate"],
        }
    )
    table.index.name = "model"
    return table.sort_values(evaluation.loss_name)
