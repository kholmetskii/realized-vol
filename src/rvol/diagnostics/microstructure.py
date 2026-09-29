"""Statistical diagnostics for market-microstructure noise."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from rvol.features.realized import (
    daily_observation_count,
    daily_realized_variance,
)
from rvol.market.sessions import MIN_SESSION_HOURS, full_sessions


@dataclass(frozen=True)
class NoiseTest:
    """Result of a paired fine-versus-coarse realized-variance test."""

    fine: str
    coarse: str
    n_days: int
    mean_ratio: float
    se_log_ratio: float
    t_stat: float
    p_value: float
    implied_noise_bps: float

    def __str__(self) -> str:
        return (
            f"RV({self.fine}) / RV({self.coarse}) = {self.mean_ratio:.4f} "
            f"over {self.n_days} sessions\n"
            f"t = {self.t_stat:.2f}, one-sided p = {self.p_value:.2g}\n"
            f"implied noise sd: {self.implied_noise_bps:.3f} bps per observation"
        )


def noise_test(
    ticks: pd.DataFrame,
    price_col: str = "mid",
    fine: str = "1s",
    coarse: str = "5min",
    min_hours: float = MIN_SESSION_HOURS,
) -> NoiseTest:
    """Test whether fine sampling inflates RV, pairing observations by session."""
    prices = ticks.set_index("ts")[price_col].sort_index()
    prices = full_sessions(prices, min_hours)

    rv_fine = daily_realized_variance(prices, fine)
    rv_coarse = daily_realized_variance(prices, coarse)
    days = rv_fine.index.intersection(rv_coarse.index)
    rv_fine, rv_coarse = rv_fine[days], rv_coarse[days]

    log_ratio = np.log(rv_fine.to_numpy()) - np.log(rv_coarse.to_numpy())
    n_days = len(log_ratio)
    se = float(log_ratio.std(ddof=1) / np.sqrt(n_days))
    t_stat = float(log_ratio.mean() / se) if se > 0 else np.inf
    p_value = float(stats.t.sf(t_stat, df=n_days - 1))

    observations = daily_observation_count(prices, fine)[days].to_numpy()
    noise_variance = np.mean(
        (rv_fine.to_numpy() - rv_coarse.to_numpy()) / (2 * observations)
    )

    return NoiseTest(
        fine=fine,
        coarse=coarse,
        n_days=n_days,
        mean_ratio=float(np.exp(log_ratio.mean())),
        se_log_ratio=se,
        t_stat=t_stat,
        p_value=p_value,
        implied_noise_bps=float(np.sqrt(max(noise_variance, 0.0)) * 1e4),
    )
