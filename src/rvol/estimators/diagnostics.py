"""How realized variance depends on the sampling frequency.

The volatility signature averages daily RV at frequencies from 1 second to an
hour; the paired noise test compares fine and coarse RV session by session to
estimate how much microstructure noise inflates RV at fine sampling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from rvol.data.market.sessions import MIN_SESSION_HOURS, full_sessions
from rvol.estimators.realized import (
    daily_observation_count,
    daily_realized_variance,
)

FREQS: list[str] = [
    "1s",
    "2s",
    "5s",
    "10s",
    "15s",
    "30s",
    "1min",
    "2min",
    "5min",
    "10min",
    "15min",
    "30min",
    "60min",
]

TRADING_DAYS_PER_YEAR = 252


def freq_seconds(frequency: str) -> float:
    """Convert a pandas frequency string to seconds."""
    return pd.Timedelta(frequency).total_seconds()


def signature(
    ticks: pd.DataFrame,
    price_col: str = "mid",
    freqs: list[str] | None = None,
    min_hours: float = MIN_SESSION_HOURS,
) -> pd.DataFrame:
    """Summarise average daily RV across sampling frequencies.

    The result contains variance-scale estimates and annualised volatility.
    Market-specific preparation lives in ``data.market`` and ``data.volatility``;
    numerical RV itself lives in ``estimators.variance``.
    """
    frequencies = freqs or FREQS
    prices = ticks.set_index("ts")[price_col].sort_index()
    prices = full_sessions(prices, min_hours)

    rows = []
    for frequency in frequencies:
        daily = daily_realized_variance(prices, frequency)
        mean_rv = daily.mean()
        rows.append(
            {
                "freq": frequency,
                "seconds": freq_seconds(frequency),
                "mean_RV": mean_rv,
                "ann_vol_pct": np.sqrt(mean_rv * TRADING_DAYS_PER_YEAR) * 100,
                "n_days": len(daily),
                "se": daily.std(ddof=1) / np.sqrt(len(daily)),
            }
        )

    result = pd.DataFrame(rows)
    result["ann_vol_se"] = (
        result["se"]
        * np.sqrt(TRADING_DAYS_PER_YEAR)
        / (2 * np.sqrt(result["mean_RV"]))
        * 100
    )
    return result


class NoiseTest:
    """Result of a paired fine-versus-coarse realized-variance test."""

    def __init__(
        self,
        fine: str,
        coarse: str,
        n_days: int,
        mean_ratio: float,
        se_log_ratio: float,
        t_stat: float,
        p_value: float,
        implied_noise_bps: float,
    ) -> None:
        self.fine = fine
        self.coarse = coarse
        self.n_days = n_days
        self.mean_ratio = mean_ratio
        self.se_log_ratio = se_log_ratio
        self.t_stat = t_stat
        self.p_value = p_value
        self.implied_noise_bps = implied_noise_bps

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
    noise_variance = np.mean((rv_fine.to_numpy() - rv_coarse.to_numpy()) / (2 * observations))

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
