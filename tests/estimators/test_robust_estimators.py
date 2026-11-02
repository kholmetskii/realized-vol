"""Bipower variation, subsampled RV, the realized kernel and signature values.

Exact cases check each formula by hand; simulated one-second paths with a
known integrated variance check the property each estimator exists for.
"""

import math

import numpy as np
import pandas as pd
import pytest

from rvol.estimators import (
    PARZEN_BANDWIDTH_CONSTANT,
    bipower_variation,
    last_tick_sample,
    parzen_bandwidth,
    parzen_weight,
    realized_kernel,
    realized_variance,
    realized_variance_by_frequency,
    subsampled_realized_variance,
)

SECONDS = 86_400
#: Integrated variance of one simulated period: 10% annualized over 252 periods.
IV = 0.10**2 / 252
#: Noise variance set so that it adds 2 n w^2 = 4 IV to one-second RV.
NOISE_VARIANCE = 2 * IV / SECONDS
N_PATHS = 5


def efficient_log_prices(seed: int) -> np.ndarray:
    """Log prices of a constant-volatility diffusion observed every second."""
    steps = np.random.default_rng(seed).normal(0.0, math.sqrt(IV / SECONDS), SECONDS)
    return np.log(1.10) + np.concatenate([[0.0], np.cumsum(steps)])


def noisy_prices(seed: int) -> np.ndarray:
    """The same path observed with independent noise of variance NOISE_VARIANCE."""
    noise = np.random.default_rng(seed + 1000).normal(0.0, math.sqrt(NOISE_VARIANCE), SECONDS + 1)
    return np.exp(efficient_log_prices(seed) + noise)


def mean_over_paths(estimator, paths) -> float:
    return float(np.mean([estimator(path) for path in paths]))


# --- bipower variation ---


def test_bipower_variation_by_hand():
    # log returns +1, -2, 0, +3: adjacent products 2, 0, 0 over n = 4 returns
    prices = np.exp(np.cumsum([0.0, 1.0, -2.0, 0.0, 3.0]))

    assert bipower_variation(prices) == pytest.approx(math.pi / 2 * 4 / 3 * 2)


def test_bipower_variation_needs_two_returns():
    with pytest.raises(ValueError, match="at least 3"):
        bipower_variation(np.array([1.0, 1.1]))


def test_bipower_variation_recovers_the_integrated_variance():
    paths = [np.exp(efficient_log_prices(seed)) for seed in range(N_PATHS)]

    assert mean_over_paths(bipower_variation, paths) == pytest.approx(IV, rel=0.02)


def test_rv_minus_bipower_variation_estimates_the_squared_jump():
    jump = math.sqrt(IV)
    paths = []
    for seed in range(N_PATHS):
        log_prices = efficient_log_prices(seed)
        log_prices[SECONDS // 2 :] += jump
        paths.append(np.exp(log_prices))

    jumps = [realized_variance(path) - bipower_variation(path) for path in paths]

    assert np.mean(jumps) == pytest.approx(jump**2, rel=0.05)


# --- subsampled RV ---


def test_subsampled_rv_with_step_one_is_the_realized_variance():
    prices = np.exp(efficient_log_prices(0)[:1000])

    assert subsampled_realized_variance(prices, 1) == pytest.approx(realized_variance(prices))


def test_subsampled_rv_by_hand():
    # log prices 0, 1, 3, 6, 10: offset 0 uses 0, 3, 10 (9 + 49), offset 1 uses 1, 6 (25)
    prices = np.exp([0.0, 1.0, 3.0, 6.0, 10.0])

    assert subsampled_realized_variance(prices, 2) == pytest.approx((58 + 25) / 2)


@pytest.mark.parametrize("step", [0, -1, 1.5, True])
def test_subsampled_rv_rejects_an_invalid_step(step):
    with pytest.raises(ValueError, match="step"):
        subsampled_realized_variance(np.ones(10), step)


def test_subsampled_rv_needs_two_prices_on_every_offset_grid():
    with pytest.raises(ValueError, match="at least 6"):
        subsampled_realized_variance(np.ones(5), 3)


# --- realized kernel ---


def test_parzen_weight_values():
    assert parzen_weight(0.0) == 1.0
    assert parzen_weight(0.5) == pytest.approx(0.25)
    assert parzen_weight(1.0) == 0.0
    assert parzen_weight(1.5) == 0.0
    assert parzen_weight(-0.3) == parzen_weight(0.3)


def test_parzen_weight_is_continuous_at_one_half():
    below, above = parzen_weight(0.5 - 1e-9), parzen_weight(0.5 + 1e-9)

    assert below == pytest.approx(above, abs=1e-8)


def test_kernel_with_bandwidth_zero_is_the_realized_variance():
    prices = np.exp(efficient_log_prices(1)[:500])

    assert realized_kernel(prices, 0) == pytest.approx(realized_variance(prices))


def test_kernel_is_the_quadratic_form_of_its_weights():
    returns = np.random.default_rng(2).normal(0, 1e-3, 60)
    prices = np.exp(np.concatenate([[0.0], np.cumsum(returns)]))
    bandwidth = 7
    lags = np.subtract.outer(np.arange(60), np.arange(60))
    weights = np.vectorize(parzen_weight)(lags / (bandwidth + 1))
    quadratic_form = returns @ weights @ returns

    assert realized_kernel(prices, bandwidth) == pytest.approx(quadratic_form, rel=1e-10)


@pytest.mark.parametrize("bandwidth", [1, 2, 5, 20])
def test_kernel_is_never_negative(bandwidth):
    # alternating returns: the most negative first autocovariance there is
    prices = np.exp(np.tile([0.0, 1e-3], 50))

    assert realized_kernel(prices, bandwidth) >= 0.0


@pytest.mark.parametrize("bandwidth", [-1, 1.0, True])
def test_kernel_rejects_an_invalid_bandwidth(bandwidth):
    with pytest.raises(ValueError, match="bandwidth"):
        realized_kernel(np.exp(np.arange(10.0)), bandwidth)


def test_kernel_bandwidth_must_leave_a_return_beyond_it():
    with pytest.raises(ValueError, match="below the 9 returns"):
        realized_kernel(np.exp(np.arange(10.0)), 9)


def test_parzen_bandwidth_follows_the_rule():
    expected = math.ceil(PARZEN_BANDWIDTH_CONSTANT * 0.01**0.8 * 10_000**0.6)

    assert parzen_bandwidth(1e-4, 10_000) == expected
    # (144 / 0.269)^(1/5), the value the 2009 paper reports
    assert round(PARZEN_BANDWIDTH_CONSTANT, 4) == 3.5134


def test_parzen_bandwidth_grows_with_noise_and_with_returns():
    assert parzen_bandwidth(1e-3, 10_000) > parzen_bandwidth(1e-5, 10_000)
    assert parzen_bandwidth(1e-4, 100_000) > parzen_bandwidth(1e-4, 1_000)


@pytest.mark.parametrize(("ratio", "n"), [(0.0, 10), (-1.0, 10), (math.inf, 10), (1e-4, 0)])
def test_parzen_bandwidth_rejects_invalid_inputs(ratio, n):
    with pytest.raises(ValueError):
        parzen_bandwidth(ratio, n)


# --- behaviour under noise ---


@pytest.fixture(scope="module")
def noisy_paths():
    return [noisy_prices(seed) for seed in range(N_PATHS)]


def test_noise_biases_one_second_rv_by_two_n_noise_variances(noisy_paths):
    expected = IV + 2 * SECONDS * NOISE_VARIANCE

    assert mean_over_paths(realized_variance, noisy_paths) == pytest.approx(expected, rel=0.03)


def test_five_minute_subsampled_rv_removes_most_of_the_noise(noisy_paths):
    five_minutes = 300

    estimate = mean_over_paths(lambda p: subsampled_realized_variance(p, five_minutes), noisy_paths)

    assert estimate == pytest.approx(IV, rel=0.10)


def test_kernel_with_the_rule_bandwidth_removes_most_of_the_noise(noisy_paths):
    bandwidth = parzen_bandwidth(NOISE_VARIANCE / IV, SECONDS)

    estimate = mean_over_paths(lambda p: realized_kernel(p, bandwidth), noisy_paths)

    assert estimate == pytest.approx(IV, rel=0.10)


# --- signature values ---


def test_signature_values_are_rv_at_each_frequency(noisy_paths):
    index = pd.Timestamp("2024-01-15", tz="UTC") + pd.to_timedelta(np.arange(SECONDS + 1), "s")
    prices = pd.Series(noisy_paths[0], index=index)

    values = realized_variance_by_frequency(prices, ["1s", "1min", "5min"])

    assert list(values.index) == ["1s", "1min", "5min"]
    expected = realized_variance(last_tick_sample(prices, "1min").to_numpy())
    assert values["1min"] == pytest.approx(expected)
    # noise inflates the finest interval most: the rise a signature plot shows
    assert values["1s"] > values["1min"] > IV / 2


def test_signature_values_need_a_frequency():
    prices = pd.Series([1.0, 1.1], index=pd.date_range("2024-01-15", periods=2, freq="1s"))

    with pytest.raises(ValueError, match="frequencies"):
        realized_variance_by_frequency(prices, [])
