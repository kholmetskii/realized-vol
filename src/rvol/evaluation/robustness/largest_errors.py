"""The largest forecast misses of each model."""

from __future__ import annotations

import pandas as pd

from rvol.evaluation import EvaluationResult
from rvol.evaluation.robustness.diagnostic import Diagnostic


class LargestErrors(Diagnostic):
    """The ``top`` largest |actual - predicted| per model, in the forecasts' scale.

    This ranks raw forecast errors, not metric losses: on variance forecasts it
    lists the largest variance misses, whatever metric was evaluated.
    """

    name = "largest_errors"

    def __init__(self, top: int = 5) -> None:
        if isinstance(top, bool) or not isinstance(top, int) or top < 1:
            raise ValueError("top must be a positive integer")
        self._top = top

    def run(self, forecasts: pd.DataFrame, evaluation: EvaluationResult) -> pd.DataFrame:
        table = forecasts.assign(absolute_error=(forecasts.actual - forecasts.predicted).abs())
        table = (
            table.sort_values(
                ["model", "absolute_error", "target_date"], ascending=[True, False, True]
            )
            .groupby("model", sort=False)
            .head(self._top)
        )
        return table[["model", "target_date", "actual", "predicted", "absolute_error"]].reset_index(
            drop=True
        )
