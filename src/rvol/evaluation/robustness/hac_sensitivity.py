"""Whether a pairwise test's verdict depends on the chosen number of HAC lags."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from rvol.evaluation import DieboldMarianoTest, EvaluationResult, ForecastComparisonTest
from rvol.evaluation.robustness._selection import (
    checked_selection,
    losses_by_metric,
    resolved_baselines,
)
from rvol.evaluation.robustness.diagnostic import Diagnostic


class HacSensitivity(Diagnostic):
    """Repeat the candidate-versus-baseline test for every lag in ``lags``.

    ``make_test`` builds the test for one lag count; the default is the
    Diebold-Mariano test, so pass the same family of test you evaluated with.
    """

    name = "hac_sensitivity"

    def __init__(
        self,
        candidate: str,
        baselines: Sequence[str] | None = None,
        *,
        lags: Iterable[int] = range(6),
        make_test: Callable[[int], ForecastComparisonTest] = DieboldMarianoTest,
    ) -> None:
        self._baselines = checked_selection(candidate, baselines)
        selected = tuple(lags)
        if (
            not selected
            or len(set(selected)) != len(selected)
            or any(isinstance(lag, bool) or not isinstance(lag, int) or lag < 0 for lag in selected)
        ):
            raise ValueError("lags must be non-empty unique non-negative integers")
        self._candidate = candidate
        self._lags = selected
        self._make_test = make_test

    def run(self, forecasts: pd.DataFrame, evaluation: EvaluationResult) -> pd.DataFrame:
        baselines = resolved_baselines(forecasts, self._candidate, self._baselines)
        rows = []
        for metric, pivot in losses_by_metric(evaluation.losses):
            candidate = pivot[self._candidate].to_numpy(dtype="float64")
            for baseline in baselines:
                benchmark = pivot[baseline].to_numpy(dtype="float64")
                for lag in self._lags:
                    test = self._make_test(lag)
                    outcome = test.compare(benchmark.copy(), candidate.copy())
                    rows.append(
                        {
                            "metric": metric,
                            "baseline_model": baseline,
                            "candidate_model": self._candidate,
                            "n_obs": len(pivot),
                            "baseline_mean_loss": float(np.mean(benchmark)),
                            "candidate_mean_loss": float(np.mean(candidate)),
                            "mean_loss_difference": float(np.mean(benchmark - candidate)),
                            "test_name": test.name,
                            "statistic": outcome.statistic,
                            "p_value": outcome.p_value,
                            "test_parameters": json.dumps(outcome.parameters, sort_keys=True),
                            "hac_lags": lag,
                        }
                    )
        return pd.DataFrame(rows)
