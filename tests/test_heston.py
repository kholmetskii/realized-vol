"""First test: the generator must reproduce what it claims."""

import numpy as np

from rvol.simulation.heston import add_microstructure_noise, simulate_heston


def test_integrated_variance_matches_realized_variance_without_noise():
    """With no noise, full-frequency RV should recover integrated variance."""
    rng = np.random.default_rng(0)
    n, dt = 100_000, 1 / 252 / 100_000
    prices, _, iv = simulate_heston(n, dt, xi=0.0, rng=rng)  # xi=0 -> near-constant variance

    rv = float(np.sum(np.diff(np.log(prices)) ** 2))
    assert abs(rv / iv - 1.0) < 0.05


def test_noise_inflates_realized_variance():
    """With noise, tick-scale RV should blow up."""
    rng = np.random.default_rng(1)
    n, dt = 50_000, 1 / 252 / 50_000
    prices, _, iv = simulate_heston(n, dt, rng=rng)
    noisy = add_microstructure_noise(prices, noise_sd=1e-4, rng=rng)

    rv_noisy = float(np.sum(np.diff(np.log(noisy)) ** 2))
    assert rv_noisy > 2 * iv
