"""The forecasts and evaluation of one series under a study design."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from rvol.evaluation.evaluation import Evaluation


@dataclass(frozen=True)
class StudyRun:
    """One series' forecasting table, log forecasts and evaluation."""

    series: str
    data: pd.DataFrame
    forecasts: pd.DataFrame
    evaluation: Evaluation

    def losses(self) -> pd.DataFrame:
        """Evaluation-period losses, one column per model."""
        return self.evaluation.losses.pivot(index="target_date", columns="model", values="loss")
