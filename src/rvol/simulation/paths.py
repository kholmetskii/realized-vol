"""Ready-made path generators for the simulation studies and illustrations."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from rvol.simulation.heston import simulate_heston
from rvol.simulation.monte_carlo import PathSimulator
from rvol.simulation.noise import add_iid_log_noise


def heston_paths(n_steps: int, dt: float, *, noise_sd: float = 0.0, **heston: Any) -> PathSimulator:
    """Return a simulator of (observed prices, true integrated variance) pairs.

    Each call draws one Heston path and, when ``noise_sd`` is positive, observes
    it with i.i.d. log-price noise from the same generator. Keyword arguments
    are passed to :func:`simulate_heston` (for example ``xi=0.0``).
    """
    if noise_sd < 0:
        raise ValueError("noise_sd must be nonnegative")

    def simulate(rng: np.random.Generator) -> tuple[NDArray[np.float64], float]:
        prices, _, target = simulate_heston(n_steps, dt, rng=rng, **heston)
        if noise_sd > 0:
            prices = add_iid_log_noise(prices, noise_sd, rng)
        return np.asarray(prices, dtype=np.float64), float(target)

    return simulate


def illustrative_path(n_observations: int, seed: int) -> NDArray[np.float64]:
    """A reproducible price path with ordinary moves and two visible shocks."""
    if n_observations < 4:
        raise ValueError("n_observations must be at least four")
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, 8e-4, n_observations - 1)
    returns[n_observations // 3] += 0.012
    returns[n_observations // 3 + 1] -= 0.009
    returns[2 * n_observations // 3] -= 0.010
    return 100 * np.exp(np.r_[0.0, np.cumsum(returns)])
