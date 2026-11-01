import numpy as np
import pandas as pd
import pytest

from rvol.estimators import last_tick_sample, realized_semivariances, realized_variance


def test_realized_variance_sums_squared_log_returns():
    # log prices 0, 1, ..., 5: five returns of 1
    assert realized_variance(np.exp(np.arange(6, dtype="float64"))) == pytest.approx(5.0)


def test_realized_variance_is_invariant_to_the_price_unit():
    prices = np.array([1.10, 1.12, 1.09, 1.11])

    assert realized_variance(100 * prices) == pytest.approx(realized_variance(prices))


def test_semivariances_split_squared_returns_by_sign():
    # log returns +1, -2, 0, +3
    prices = np.exp(np.cumsum([0.0, 1.0, -2.0, 0.0, 3.0]))

    assert realized_semivariances(prices) == pytest.approx((10.0, 4.0))


def test_semivariances_add_up_to_realized_variance():
    prices = np.exp(np.cumsum(np.random.default_rng(5).normal(0, 1e-4, 500)))

    rs_pos, rs_neg = realized_semivariances(prices)

    assert rs_pos + rs_neg == pytest.approx(realized_variance(prices), rel=1e-12)


def test_reversing_every_return_swaps_the_semivariances():
    log_prices = np.cumsum(np.random.default_rng(6).normal(0, 1e-4, 500))

    rs_pos, rs_neg = realized_semivariances(np.exp(log_prices))

    assert realized_semivariances(np.exp(-log_prices)) == pytest.approx((rs_neg, rs_pos), rel=1e-9)


@pytest.mark.parametrize("estimator", [realized_variance, realized_semivariances])
@pytest.mark.parametrize(
    "prices",
    [np.array([1.0]), np.ones((2, 2)), np.array([1.0, 0.0]), np.array([1.0, np.nan])],
)
def test_invalid_price_paths_are_rejected(estimator, prices):
    with pytest.raises(ValueError, match="prices"):
        estimator(prices)


def test_five_minute_rv_recovers_the_true_volatility():
    """A continuous path ticking every second, split into 24-hour periods."""
    sigma_annual, n_days, per_day = 0.10, 20, 24 * 60 * 60
    rng = np.random.default_rng(0)
    n = n_days * per_day
    steps = rng.normal(0, sigma_annual * np.sqrt(1 / 252 / per_day), n)
    start = pd.Timestamp("2024-01-15 22:00", tz="UTC")
    index = start + pd.to_timedelta(np.arange(n), unit="s")
    prices = pd.Series(np.exp(np.log(1.10) + np.cumsum(steps)), index=index)

    days = (prices.index - start) // pd.Timedelta(days=1)
    daily = prices.groupby(days).apply(
        lambda day: realized_variance(last_tick_sample(day, "5min").to_numpy())
    )

    assert len(daily) == n_days
    assert abs(np.sqrt(daily.mean() * 252) / sigma_annual - 1) < 0.15
