"""Session RV on a synthetic path with known volatility."""

import numpy as np
import pandas as pd
import pytest

from rvol.data.market import full_sessions
from rvol.estimators import daily_realized_variance

SIGMA_ANN = 0.10
N_DAYS = 20
PER_DAY = 24 * 60 * 60  # one tick per second


@pytest.mark.slow
def test_five_minute_rv_recovers_the_true_volatility():
    """A continuous path ticking every second, split into FX sessions."""
    rng = np.random.default_rng(0)
    n = N_DAYS * PER_DAY
    log_price = np.log(1.10) + np.cumsum(rng.normal(0, SIGMA_ANN * np.sqrt(1 / 252 / PER_DAY), n))
    start = pd.Timestamp("2024-01-15 22:00", tz="UTC")
    prices = pd.Series(np.exp(log_price), index=start + pd.to_timedelta(np.arange(n), unit="s"))

    daily = daily_realized_variance(full_sessions(prices), "5min")

    annualised = np.sqrt(daily.mean() * 252)
    assert len(daily) >= N_DAYS - 2
    assert abs(annualised / SIGMA_ANN - 1) < 0.15
