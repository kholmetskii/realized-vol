"""Heston paths with known integrated variance.

These exist for verification: every estimator is checked against a quantity we
know exactly before it is ever applied to real data.
"""

from __future__ import annotations

import numpy as np


def simulate_heston(
    n_steps: int,
    dt: float,
    s0: float = 1.0,
    v0: float = 0.04,
    kappa: float = 3.0,
    theta: float = 0.04,
    xi: float = 0.5,
    rho: float = -0.7,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Euler scheme with variance reflection.

    Returns (prices, variance path, integrated variance).
    """
    rng = rng or np.random.default_rng()
    z1 = rng.standard_normal(n_steps)
    z2 = rho * z1 + np.sqrt(1 - rho**2) * rng.standard_normal(n_steps)

    v = np.empty(n_steps + 1)
    log_s = np.empty(n_steps + 1)
    v[0], log_s[0] = v0, np.log(s0)

    sqrt_dt = np.sqrt(dt)
    for t in range(n_steps):
        vt = max(v[t], 0.0)
        sv = np.sqrt(vt)
        v[t + 1] = abs(vt + kappa * (theta - vt) * dt + xi * sv * sqrt_dt * z2[t])
        log_s[t + 1] = log_s[t] - 0.5 * vt * dt + sv * sqrt_dt * z1[t]

    integrated_var = float(np.sum(v[:-1]) * dt)
    return np.exp(log_s), v, integrated_var


def add_microstructure_noise(
    prices: np.ndarray, noise_sd: float, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Additive noise in logs — the simplest model of bid-ask bounce."""
    rng = rng or np.random.default_rng()
    return prices * np.exp(rng.normal(0.0, noise_sd, size=prices.shape))
