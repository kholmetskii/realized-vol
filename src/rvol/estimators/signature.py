"""Volatility signature: realized variance as a function of sampling frequency.

As sampling moves toward the tick, RV drifts upward — you are summing squared
microstructure noise rather than information. The plateau at moderate
frequencies is the reason the industry standard is five minutes.

Days are FX trading days, not calendar days: the week opens on Sunday evening
and each session runs 21:00 UTC to 21:00 UTC (17:00 New York, the market
convention). Splitting at midnight UTC instead cuts the New York afternoon in
half and turns each Sunday evening into a two-hour "day" whose tiny RV drags
the average down.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

# Frequency grid from 1 second to an hour
FREQS: list[str] = ["1s", "2s", "5s", "10s", "15s", "30s", "1min", "2min",
                    "5min", "10min", "15min", "30min", "60min"]

#: The session boundary: 17:00 in New York, the FX market convention.
SESSION_CLOSE_HOUR = 17

#: The zone that boundary is fixed in. It must follow US daylight saving —
#: 17:00 New York is 22:00 UTC in winter and 21:00 UTC in summer.
SESSION_TZ = "America/New_York"

#: A session shorter than this is a holiday or a half-open Sunday, not a day.
MIN_SESSION_HOURS = 12.0


def freq_seconds(freq: str) -> float:
    return pd.Timedelta(freq).total_seconds()


def trading_day(ts: pd.DatetimeIndex, close_hour: int = SESSION_CLOSE_HOUR,
                tz: str = SESSION_TZ) -> pd.DatetimeIndex:
    """Label each timestamp with the trading day it belongs to.

    A session runs from `close_hour` to `close_hour` in `tz` and is named after
    the local date it ends on, so Sunday evening and Monday afternoon share the
    label Monday. Fixing the boundary in UTC instead would put it an hour out
    for half the year, since New York moves and UTC does not.
    """
    idx = ts.tz_localize("UTC") if ts.tz is None else ts
    shifted = idx.tz_convert(tz) + pd.Timedelta(hours=24 - close_hour)
    return pd.DatetimeIndex(shifted.normalize().tz_localize(None))


def realized_variance(prices: pd.Series, freq: str,
                      origin: pd.Timestamp | str = "start_day") -> float:
    """RV for one day using last-tick sampling at step `freq`.

    `origin` anchors the sampling grid. Moving it shifts every bin boundary,
    which changes the answer — see `subsampled_variance`.
    """
    p = prices.resample(freq, origin=origin).last().dropna()
    if len(p) < 3:
        return np.nan
    r = np.diff(np.log(p.to_numpy()))
    return float(np.sum(r**2))


def grid_origins(prices: pd.Series, freq: str, n_grids: int) -> list[pd.Timestamp]:
    """`n_grids` equally spaced anchors covering one sampling interval."""
    step = pd.Timedelta(freq)
    base = prices.index[0].normalize()
    return [base + step * k / n_grids for k in range(n_grids)]


def subsampled_variance(prices: pd.Series, freq: str, n_grids: int = 12
                        ) -> float:
    """RV averaged over every shift of the sampling grid.

    One grid throws away all the ticks that fall inside its bins, and which
    ticks those are is arbitrary: on EURUSD, sliding a five-minute grid across
    its own width moves a single session's RV by tens of percent. Averaging
    over the shifts uses the whole sample and cuts that arbitrariness, at no
    cost in bias — each grid is itself an unbiased estimate.

    This is the RV_avg of Zhang, Mykland and Ait-Sahalia (2005); it still
    carries the noise bias of the underlying RV, which the two-scale estimator
    removes by combining RV_avg with the tick-level RV.
    """
    values = [
        realized_variance(prices, freq, origin=o)
        for o in grid_origins(prices, freq, n_grids)
    ]
    values = [v for v in values if not np.isnan(v)]
    return float(np.mean(values)) if values else np.nan


def sessions(s: pd.Series) -> pd.Series:
    """The trading day each observation belongs to, aligned with `s`."""
    return pd.Series(trading_day(pd.DatetimeIndex(s.index)), index=s.index)


def daily_rv(s: pd.Series, freq: str) -> pd.Series:
    """RV per trading day at one sampling frequency."""
    return s.groupby(sessions(s)).apply(
        lambda x: realized_variance(x, freq)
    ).dropna()


def daily_obs(s: pd.Series, freq: str) -> pd.Series:
    """How many sampled prices each trading day yields at `freq`."""
    return s.groupby(sessions(s)).apply(
        lambda x: float(len(x.resample(freq).last().dropna()))
    )


def full_sessions(s: pd.Series, min_hours: float = MIN_SESSION_HOURS
                  ) -> pd.Series:
    """Drop sessions that cover too little of the day to be comparable.

    A Sunday holds only the two hours after the weekly open; keeping it as a
    whole day would bias every average downward.
    """
    day = sessions(s)
    span = s.groupby(day).apply(
        lambda x: (x.index[-1] - x.index[0]).total_seconds() / 3600.0
    )
    keep = set(span[span >= min_hours].index)
    return s[day.isin(keep)]


def signature(ticks: pd.DataFrame, price_col: str = "mid",
              freqs: list[str] | None = None,
              min_hours: float = MIN_SESSION_HOURS) -> pd.DataFrame:
    """Average daily RV across a grid of sampling frequencies.

    Averages per trading day first, so that long days do not dominate, and
    keeps only sessions covering at least `min_hours` hours.
    """
    freqs = freqs or FREQS
    s = ticks.set_index("ts")[price_col].sort_index()
    s = full_sessions(s, min_hours)

    rows = []
    for f in freqs:
        daily = daily_rv(s, f)
        rv = daily.mean()
        rows.append({
            "freq": f,
            "seconds": freq_seconds(f),
            "mean_RV": rv,
            "ann_vol_pct": np.sqrt(rv * 252) * 100,   # annualised, 252 days
            "n_days": len(daily),
            "se": daily.std(ddof=1) / np.sqrt(len(daily)),
        })
    out = pd.DataFrame(rows)

    # Standard error of the annualised figure, by the delta method:
    # d/dRV of sqrt(252 RV) = sqrt(252) / (2 sqrt(RV)).
    out["ann_vol_se"] = (
        out["se"] * np.sqrt(252) / (2 * np.sqrt(out["mean_RV"])) * 100
    )
    return out


@dataclass(frozen=True)
class NoiseTest:
    """Whether fine sampling inflates RV, judged day by day."""

    fine: str
    coarse: str
    n_days: int
    mean_ratio: float          # geometric mean of RV(fine) / RV(coarse)
    se_log_ratio: float
    t_stat: float
    p_value: float             # one-sided: is the ratio above one?
    implied_noise_bps: float   # noise standard deviation per observation

    def __str__(self) -> str:
        return (
            f"RV({self.fine}) / RV({self.coarse}) = {self.mean_ratio:.4f} "
            f"over {self.n_days} sessions\n"
            f"t = {self.t_stat:.2f}, one-sided p = {self.p_value:.2g}\n"
            f"implied noise sd: {self.implied_noise_bps:.3f} bps per observation"
        )


def noise_test(ticks: pd.DataFrame, price_col: str = "mid",
               fine: str = "1s", coarse: str = "5min",
               min_hours: float = MIN_SESSION_HOURS) -> NoiseTest:
    """Test whether RV at `fine` sampling exceeds RV at `coarse`, pairing by day.

    Comparing the two averages across days is weak: a volatile March raises
    both, so the shared variation swells the standard errors. Pairing removes
    it — each session contributes one ratio, and the question becomes whether
    those ratios sit above one.

    The test is on log ratios, which are symmetric around zero and closer to
    normal than the ratios themselves.

    The implied noise follows from E[RV_n] = IV + 2*n*omega^2: the gap between
    the two estimates, divided by twice the number of fine observations, is an
    estimate of omega^2. It is reported in basis points and should land near
    the half-spread for a quote series, and below it for mid prices.
    """
    s = ticks.set_index("ts")[price_col].sort_index()
    s = full_sessions(s, min_hours)

    rv_fine = daily_rv(s, fine)
    rv_coarse = daily_rv(s, coarse)
    days = rv_fine.index.intersection(rv_coarse.index)
    rv_fine, rv_coarse = rv_fine[days], rv_coarse[days]

    log_ratio = np.log(rv_fine.to_numpy()) - np.log(rv_coarse.to_numpy())
    n = len(log_ratio)
    se = float(log_ratio.std(ddof=1) / np.sqrt(n))
    t_stat = float(log_ratio.mean() / se) if se > 0 else np.inf
    p_value = float(stats.t.sf(t_stat, df=n - 1))

    obs = daily_obs(s, fine)[days].to_numpy()
    omega_sq = np.mean((rv_fine.to_numpy() - rv_coarse.to_numpy()) / (2 * obs))
    implied = float(np.sqrt(max(omega_sq, 0.0)) * 1e4)

    return NoiseTest(
        fine=fine,
        coarse=coarse,
        n_days=n,
        mean_ratio=float(np.exp(log_ratio.mean())),
        se_log_ratio=se,
        t_stat=t_stat,
        p_value=p_value,
        implied_noise_bps=implied,
    )
