"""Mean losses by calendar month of the target date."""

from __future__ import annotations

import pandas as pd

from rvol.evaluation import EvaluationResult
from rvol.evaluation.robustness.diagnostic import Diagnostic


class MonthlyLosses(Diagnostic):
    """Show whether an overall ranking rests on a few unusual months."""

    name = "monthly_losses"

    def run(self, forecasts: pd.DataFrame, evaluation: EvaluationResult) -> pd.DataFrame:
        losses = evaluation.losses
        return (
            losses.assign(month=losses.target_date.dt.strftime("%Y-%m"))
            .groupby(["month", "metric", "model"], sort=True)
            .loss.agg(n_obs="size", mean_loss="mean")
            .reset_index()
        )
