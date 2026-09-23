import numpy as np

from rvol.estimators.realized import realized_variance, subsampled_variance
from rvol.simulation.heston import simulate_heston
from rvol.simulation.monte_carlo import MonteCarloSummary, evaluate_estimators
from rvol.simulation.noise import add_iid_log_noise

N_STEPS = 4_000
DT = 1 / 252 / N_STEPS
N_REPLICATIONS = 100


def clean_path(rng: np.random.Generator) -> tuple[np.ndarray, float]:
    prices, _, integrated_variance = simulate_heston(
        N_STEPS, DT, xi=0.0, rng=rng
    )
    return prices, integrated_variance


def noisy_path(rng: np.random.Generator) -> tuple[np.ndarray, float]:
    prices, _, integrated_variance = simulate_heston(
        N_STEPS, DT, xi=0.0, rng=rng
    )
    observed = add_iid_log_noise(prices, noise_sd=1e-4, rng=rng)
    return observed, integrated_variance


ESTIMATORS = {
    "naive": realized_variance,
    "subsampled-20": lambda prices: subsampled_variance(prices, step=20),
}


def by_name(summaries: tuple[MonteCarloSummary, ...]) -> dict[str, MonteCarloSummary]:
    return {summary.estimator: summary for summary in summaries}


def test_monte_carlo_is_reproducible_and_recovers_clean_integrated_variance():
    first = evaluate_estimators(
        clean_path, ESTIMATORS, n_replications=N_REPLICATIONS, seed=11
    )
    second = evaluate_estimators(
        clean_path, ESTIMATORS, n_replications=N_REPLICATIONS, seed=11
    )

    assert first == second
    for summary in first:
        assert abs(summary.relative_bias) < 0.03
        assert summary.bias_se > 0
        assert summary.rmse > 0


def test_subsampling_reduces_noise_bias_and_rmse():
    summaries = by_name(evaluate_estimators(
        noisy_path, ESTIMATORS, n_replications=N_REPLICATIONS, seed=12
    ))
    naive = summaries["naive"]
    subsampled = summaries["subsampled-20"]

    # The theoretical leading noise bias is 2 * N_STEPS * noise_sd**2,
    # approximately half the latent IV under this fixed scenario.
    assert naive.relative_bias > 0.45
    assert 0 < subsampled.bias < naive.bias / 5
    assert subsampled.rmse < naive.rmse / 5
