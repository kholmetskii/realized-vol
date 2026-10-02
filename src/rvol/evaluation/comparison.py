"""Statistical comparison of competing volatility forecasts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm


@dataclass(frozen=True)
class DieboldMarianoResult:
    """One-sided test of whether a candidate has lower expected loss."""

    n_obs: int
    mean_loss_difference: float
    statistic: float
    p_value: float
    hac_lags: int

def _loss_vector(values: ArrayLike, name: str) -> NDArray[np.float64]:
    vector = np.asarray(values, dtype="float64")
    if vector.ndim != 1 or len(vector) < 3:
        raise ValueError(f"{name} must contain at least three observations")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _default_hac_lags(n_obs: int) -> int:
    """Newey-West's common sample-size rule, bounded by available lags."""
    rule = int(np.floor(4 * (n_obs / 100) ** (2 / 9)))
    return min(rule, n_obs - 2)


def diebold_mariano(
    baseline_loss: ArrayLike,
    candidate_loss: ArrayLike,
    *,
    hac_lags: int | None = None,
) -> DieboldMarianoResult:
    """Test whether a candidate's expected loss is lower than a baseline's.

    The loss differential is ``baseline_loss - candidate_loss``. The reported
    p-value is one-sided for the alternative that its expectation is positive. A
    Bartlett-kernel Newey-West estimate allows the differential to be serially
    correlated.
    """
    baseline = _loss_vector(baseline_loss, "baseline_loss")
    candidate = _loss_vector(candidate_loss, "candidate_loss")
    if baseline.shape != candidate.shape:
        raise ValueError("baseline_loss and candidate_loss must have the same shape")

    n_obs = len(baseline)
    lags = _default_hac_lags(n_obs) if hac_lags is None else hac_lags
    if not 0 <= lags <= n_obs - 2:
        raise ValueError("hac_lags must satisfy 0 <= hac_lags <= n_obs - 2")

    differential = baseline - candidate
    mean_difference = float(np.mean(differential))
    centered = differential - mean_difference
    long_run_variance = float(centered @ centered / n_obs)
    for lag in range(1, lags + 1):
        covariance = float(centered[lag:] @ centered[:-lag] / n_obs)
        weight = 1.0 - lag / (lags + 1.0)
        long_run_variance += 2.0 * weight * covariance
    long_run_variance = max(long_run_variance, 0.0)

    tolerance = np.finfo("float64").eps * max(1.0, abs(mean_difference))
    if long_run_variance <= tolerance:
        if abs(mean_difference) <= tolerance:
            statistic, p_value = 0.0, 0.5
        elif mean_difference > 0:
            statistic, p_value = float("inf"), 0.0
        else:
            statistic, p_value = float("-inf"), 1.0
    else:
        standard_error = np.sqrt(long_run_variance / n_obs)
        statistic = mean_difference / standard_error
        p_value = float(norm.sf(statistic))

    return DieboldMarianoResult(
        n_obs=n_obs,
        mean_loss_difference=mean_difference,
        statistic=float(statistic),
        p_value=p_value,
        hac_lags=lags,
    )
