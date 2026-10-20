"""How often the candidate beats each baseline, forecast by forecast."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from rvol.evaluation import EvaluationResult
from rvol.evaluation.robustness._selection import (
    checked_selection,
    losses_by_metric,
    resolved_baselines,
)
from rvol.evaluation.robustness.diagnostic import Diagnostic


class WinRates(Diagnostic):
    """Share of target dates on which the candidate's loss is strictly lower.

    Losses equal within ``np.isclose(rtol, atol)`` count as ties, not wins.
    """

    name = "win_rates"

    def __init__(
        self,
        candidate: str,
        baselines: Sequence[str] | None = None,
        *,
        tie_rtol: float = 1e-5,
        tie_atol: float = 1e-8,
    ) -> None:
        self._baselines = checked_selection(candidate, baselines)
        if any(not np.isfinite(v) or v < 0 for v in (tie_rtol, tie_atol)):
            raise ValueError("tie tolerances must be finite and non-negative")
        self._candidate = candidate
        self._tie_rtol = tie_rtol
        self._tie_atol = tie_atol

    def run(self, forecasts: pd.DataFrame, evaluation: EvaluationResult) -> pd.DataFrame:
        baselines = resolved_baselines(forecasts, self._candidate, self._baselines)
        rows = []
        for metric, pivot in losses_by_metric(evaluation.losses):
            candidate = pivot[self._candidate].to_numpy(dtype="float64")
            for baseline in baselines:
                benchmark = pivot[baseline].to_numpy(dtype="float64")
                ties = np.isclose(candidate, benchmark, rtol=self._tie_rtol, atol=self._tie_atol)
                wins = int(np.count_nonzero((candidate < benchmark) & ~ties))
                rows.append(
                    {
                        "metric": metric,
                        "baseline_model": baseline,
                        "candidate_model": self._candidate,
                        "n_obs": len(pivot),
                        "candidate_wins": wins,
                        "ties": int(ties.sum()),
                        "candidate_win_rate": wins / len(pivot),
                    }
                )
        return pd.DataFrame(rows)
