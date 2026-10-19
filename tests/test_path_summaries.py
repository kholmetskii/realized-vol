import numpy as np
import pandas as pd
import pytest

from rvol.estimators.variance import (
    cumulative_realized_variance,
    grid_realized_variances,
    realized_variance,
    subsampled_variance,
)
from rvol.evaluation import cumulative_loss_advantage, evaluate
from rvol.simulation.heston import cumulative_integrated_variance, simulate_heston
from rvol.simulation.monte_carlo import MonteCarloSummary


def path(n: int = 41, seed: int = 3) -> np.ndarray:
    returns = np.random.default_rng(seed).normal(0.0, 1e-3, n - 1)
    return 100 * np.exp(np.r_[0.0, np.cumsum(returns)])


def test_cumulative_rv_starts_at_zero_and_ends_at_realized_variance():
    prices = path()

    running = cumulative_realized_variance(prices)

    assert running.shape == prices.shape
    assert running[0] == 0
    assert np.all(np.diff(running) >= 0)
    assert running[-1] == pytest.approx(realized_variance(prices))


def test_grid_rvs_cover_every_origin_and_average_to_subsampled_rv():
    prices = path()

    grids = grid_realized_variances(prices, step=4)

    assert len(grids) == 4
    assert grids[0] == pytest.approx(realized_variance(prices, step=4, offset=0))
    assert grids.mean() == pytest.approx(subsampled_variance(prices, step=4))


def test_cumulative_iv_matches_the_simulated_integrated_variance():
    dt = 1 / 252 / 200
    _, variance, integrated = simulate_heston(200, dt, rng=np.random.default_rng(1))

    running = cumulative_integrated_variance(variance, dt)

    assert running[0] == 0
    assert running[-1] == pytest.approx(integrated, rel=1e-12)


@pytest.mark.parametrize("variance, dt", [([0.04], 0.1), ([0.04, -0.01], 0.1), ([0.04, 0.04], 0)])
def test_cumulative_iv_rejects_invalid_inputs(variance, dt):
    with pytest.raises(ValueError):
        cumulative_integrated_variance(np.array(variance), dt)


def test_relative_rmse_is_rmse_over_the_mean_target():
    summary = MonteCarloSummary("RV", 10, 1.1, 1.0, 0.1, 0.01, 0.2)
    assert summary.relative_rmse == pytest.approx(0.2)


def test_cumulative_loss_advantage_sums_baseline_minus_candidate_losses():
    dates = pd.bdate_range("2024-01-02", periods=5)
    actual = np.arange(1.0, 6.0)
    forecasts = pd.concat(
        [
            pd.DataFrame(
                {
                    "model": name,
                    "origin_date": dates - pd.offsets.BDay(),
                    "target_date": dates,
                    "actual": actual,
                    "predicted": actual + error,
                }
            )
            for name, error in (("HAR", 0.1), ("naive", 0.5), ("mean", 1.0))
        ],
        ignore_index=True,
    )
    evaluation = evaluate(forecasts)

    advantage = cumulative_loss_advantage(evaluation, metric="MSE", candidate="HAR")

    assert list(advantage.columns) == ["mean", "naive"]
    np.testing.assert_allclose(advantage["naive"], np.arange(1, 6) * (0.25 - 0.01))
    np.testing.assert_allclose(advantage["mean"], np.arange(1, 6) * (1.0 - 0.01))
    with pytest.raises(ValueError, match="candidate"):
        cumulative_loss_advantage(evaluation, metric="MSE", candidate="AR1")
