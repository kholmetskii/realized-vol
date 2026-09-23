"""Repeated-simulation validation for integrated-variance estimators."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

PathSimulator = Callable[[np.random.Generator], tuple[NDArray[np.float64], float]]
VarianceEstimator = Callable[[NDArray[np.float64]], float]


@dataclass(frozen=True)
class MonteCarloSummary:
    """Sampling performance of one estimator over repeated paths."""

    estimator: str
    n_replications: int
    mean_estimate: float
    mean_target: float
    bias: float
    bias_se: float
    rmse: float

    @property
    def relative_bias(self) -> float:
        """Bias divided by the average latent integrated variance."""
        return self.bias / self.mean_target


def evaluate_estimators(
    simulate: PathSimulator,
    estimators: Mapping[str, VarianceEstimator],
    *,
    n_replications: int,
    seed: int,
) -> tuple[MonteCarloSummary, ...]:
    """Estimate bias and RMSE against each path's known latent target.

    One seeded generator drives the full experiment, making the path/noise draws
    and summary statistics reproducible. All estimators see the same path in each
    replication, which makes comparisons paired rather than needlessly noisy.
    """
    if n_replications < 2:
        raise ValueError("n_replications must be at least two")
    if not estimators:
        raise ValueError("at least one estimator is required")

    rng = np.random.default_rng(seed)
    estimates = {name: np.empty(n_replications) for name in estimators}
    targets = np.empty(n_replications)

    for replication in range(n_replications):
        prices, target = simulate(rng)
        if not np.isfinite(target) or target <= 0:
            raise ValueError("simulation targets must be finite and strictly positive")
        targets[replication] = target
        for name, estimator in estimators.items():
            estimate = float(estimator(prices))
            if not np.isfinite(estimate) or estimate < 0:
                raise ValueError(f"estimator {name!r} returned an invalid variance")
            estimates[name][replication] = estimate

    summaries = []
    for name, values in estimates.items():
        errors = values - targets
        summaries.append(MonteCarloSummary(
            estimator=name,
            n_replications=n_replications,
            mean_estimate=float(np.mean(values)),
            mean_target=float(np.mean(targets)),
            bias=float(np.mean(errors)),
            bias_se=float(np.std(errors, ddof=1) / np.sqrt(n_replications)),
            rmse=float(np.sqrt(np.mean(errors**2))),
        ))
    return tuple(summaries)
