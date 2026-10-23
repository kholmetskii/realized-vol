"""Robustness checks on a losses table from forecast_losses."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np
import pandas as pd

from rvol.evaluation.diebold_mariano import default_hac_lags, diebold_mariano


def win_rates(
    losses: pd.DataFrame,
    candidate: str,
    *,
    tie_rtol: float = 1e-5,
    tie_atol: float = 0.0,
) -> pd.DataFrame:
    """Share of target dates on which the candidate's loss is strictly lower.

    One row per baseline (every other model). Losses equal within
    ``np.isclose(rtol, atol)`` count as ties, not wins. The absolute tolerance
    defaults to zero because the losses' scale is unknown here.
    """
    if any(not np.isfinite(value) or value < 0 for value in (tie_rtol, tie_atol)):
        raise ValueError("tie tolerances must be finite and non-negative")
    rows = []
    for baseline, benchmark, proposed in _paired_losses(losses, candidate):
        ties = np.isclose(proposed, benchmark, rtol=tie_rtol, atol=tie_atol)
        wins = int(np.count_nonzero((proposed < benchmark) & ~ties))
        rows.append(
            {
                "baseline_model": baseline,
                "candidate_model": candidate,
                "n_obs": len(proposed),
                "candidate_wins": wins,
                "ties": int(ties.sum()),
                "candidate_win_rate": wins / len(proposed),
            }
        )
    return pd.DataFrame(rows)


def hac_sensitivity(
    losses: pd.DataFrame,
    candidate: str,
    lags: Iterable[int] | None = None,
) -> pd.DataFrame:
    """The one-sided DM test of the candidate against every baseline, for each HAC lag.

    A verdict that changes with the lag count depends on the Newey-West
    correction rather than on the forecasts. By default the lags run from 0 to
    the default lag count for each pair's sample size.
    """
    selected = None if lags is None else tuple(lags)
    if selected is not None and (
        not selected
        or len(set(selected)) != len(selected)
        or any(isinstance(lag, bool) or not isinstance(lag, int) or lag < 0 for lag in selected)
    ):
        raise ValueError("lags must be non-empty unique non-negative integers")
    rows = []
    for baseline, benchmark, proposed in _paired_losses(losses, candidate):
        pair_lags = range(default_hac_lags(len(benchmark)) + 1) if selected is None else selected
        for lag in pair_lags:
            statistic, p_value = diebold_mariano(benchmark, proposed, hac_lags=lag)
            rows.append(
                {
                    "baseline_model": baseline,
                    "candidate_model": candidate,
                    "hac_lags": lag,
                    "statistic": statistic,
                    "p_value": p_value,
                }
            )
    return pd.DataFrame(rows)


def _paired_losses(
    losses: pd.DataFrame, candidate: str
) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Yield (baseline, baseline losses, candidate losses) aligned by date."""
    models = set(losses["model"])
    if candidate not in models:
        raise ValueError(f"unknown candidate model {candidate!r}")
    if len(models) < 2:
        raise ValueError("robustness checks need at least one baseline model")
    table = losses.pivot(index="target_date", columns="model", values="loss").sort_index()
    proposed = table[candidate].to_numpy(dtype="float64")
    for baseline in sorted(models - {candidate}):
        yield baseline, table[baseline].to_numpy(dtype="float64"), proposed
