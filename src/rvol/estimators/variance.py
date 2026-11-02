"""Realized variance of a price path, observed or simulated, and its relatives.

Every estimator takes the prices of one period, in time order, and returns a
variance on that period's own scale, without annualization. With n + 1 prices
there are n log returns r_1, ..., r_n.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

#: Rate constant of the Parzen bandwidth rule, (k''(0)^2 / k^{0,0})^(1/5) with
#: k''(0) = -12 and k^{0,0} = 0.269 (Barndorff-Nielsen, Hansen, Lunde & Shephard, 2009).
PARZEN_BANDWIDTH_CONSTANT = (144 / 0.269) ** 0.2


def _log_prices(prices: ArrayLike, minimum: int = 2) -> np.ndarray:
    values = np.asarray(prices, dtype=np.float64)
    if values.ndim != 1 or len(values) < minimum:
        raise ValueError(f"prices must be a one-dimensional array with at least {minimum} values")
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("prices must be finite and strictly positive")
    return np.log(values)


def _log_returns(prices: ArrayLike, minimum: int = 2) -> np.ndarray:
    return np.diff(_log_prices(prices, minimum))


def realized_variance(prices: ArrayLike) -> float:
    """Sum of the squared log returns between consecutive prices.

    n prices give n - 1 returns. The result is on the variance scale of the
    path's own period, without annualization.
    """
    return float(np.sum(_log_returns(prices) ** 2))


def realized_semivariances(prices: ArrayLike) -> tuple[float, float]:
    """Realized variance split by the sign of each return: (positive, negative).

    RS+ sums the squared positive log returns and RS- the squared negative
    ones (Barndorff-Nielsen, Kinnebrock & Shephard, 2010). A zero return adds
    to neither, so RS+ + RS- equals the realized variance up to rounding.
    """
    returns = _log_returns(prices)
    squared = returns**2
    return float(np.sum(squared[returns > 0])), float(np.sum(squared[returns < 0]))


def bipower_variation(prices: ArrayLike) -> float:
    """Jump-robust variance: the scaled sum of products of adjacent absolute returns.

        BV = (pi / 2) * n / (n - 1) * sum_{i=2}^{n} |r_i| |r_{i-1}|

    (Barndorff-Nielsen & Shephard, 2004). For a Gaussian return E|r| equals
    sqrt(2 / pi) times its standard deviation, so pi / 2 undoes the product of
    two absolute moments, and n / (n - 1) rescales the n - 1 products to n
    returns. A jump enters RV as its square but enters only two products of BV,
    each times an ordinary return, so RV - BV estimates the squared jumps.
    """
    returns = np.abs(_log_returns(prices, minimum=3))
    n = len(returns)
    return float(np.pi / 2 * n / (n - 1) * np.sum(returns[1:] * returns[:-1]))


def subsampled_realized_variance(prices: ArrayLike, step: int) -> float:
    """Mean of the realized variances on the step offset grids of every step-th price.

    For prices on a fine grid, offset k in 0, ..., step - 1 uses prices k,
    k + step, k + 2 step, ... Each offset grid is a coarse RV with the small
    noise bias of a coarse grid; averaging the offsets uses every price and
    lowers the variance of a single coarse RV (Zhang, Mykland & Ait-Sahalia,
    2005). step = 1 gives the realized variance.

    An offset grid skips at most step - 1 fine returns at each end, a fraction
    of the path of order step / n.
    """
    if isinstance(step, bool) or not isinstance(step, int) or step < 1:
        raise ValueError("step must be a positive integer")
    log_prices = _log_prices(prices, minimum=2 * step)
    return float(np.mean([np.sum(np.diff(log_prices[k::step]) ** 2) for k in range(step)]))


def parzen_weight(x: float) -> float:
    """The Parzen kernel: 1 - 6x^2 + 6|x|^3 up to |x| = 1/2, then 2(1 - |x|)^3, 0 past 1."""
    a = abs(x)
    if a <= 0.5:
        return 1 - 6 * a**2 + 6 * a**3
    if a <= 1:
        return 2 * (1 - a) ** 3
    return 0.0


def realized_kernel(prices: ArrayLike, bandwidth: int) -> float:
    """Noise-robust variance: realized autocovariances weighted by the Parzen kernel.

        RK = gamma_0 + 2 sum_{h=1}^{H} k(h / (H + 1)) gamma_h,
        gamma_h = sum_{i=h+1}^{n} r_i r_{i-h},

    with H the bandwidth (Barndorff-Nielsen, Hansen, Lunde & Shephard, 2008,
    2009). Independent noise of variance w^2 adds about 2 n w^2 to gamma_0 and
    -n w^2 to gamma_1 and nothing beyond, so with k near 1 at short lags the two
    nearly cancel. RK equals r' K r with K_ij = k((i - j) / (H + 1)); the
    Parzen kernel has a non-negative Fourier transform, so K is positive
    semidefinite and RK is never negative. H = 0 gives the realized variance.

    The jittering of the end prices in the 2009 paper is left out: it removes
    a bias that the edges add, of the order of w^2.
    """
    if isinstance(bandwidth, bool) or not isinstance(bandwidth, int) or bandwidth < 0:
        raise ValueError("bandwidth must be a non-negative integer")
    returns = _log_returns(prices)
    if bandwidth >= len(returns):
        raise ValueError(f"bandwidth must be below the {len(returns)} returns")
    total = float(np.dot(returns, returns))
    for h in range(1, bandwidth + 1):
        total += 2 * parzen_weight(h / (bandwidth + 1)) * float(np.dot(returns[h:], returns[:-h]))
    return total


def parzen_bandwidth(noise_to_signal: float, n_returns: int) -> int:
    """Bandwidth of the Parzen realized kernel: ceil(c* xi^(4/5) n^(3/5)).

    noise_to_signal is xi^2, the noise variance over the integrated variance
    of the period, and n_returns the number of returns the kernel sees
    (Barndorff-Nielsen, Hansen, Lunde & Shephard, 2009). More noise or more
    returns need more lags to cancel it.
    """
    if not math.isfinite(noise_to_signal) or noise_to_signal <= 0:
        raise ValueError("noise_to_signal must be positive and finite")
    if isinstance(n_returns, bool) or not isinstance(n_returns, int) or n_returns < 1:
        raise ValueError("n_returns must be a positive integer")
    xi = math.sqrt(noise_to_signal)
    return math.ceil(PARZEN_BANDWIDTH_CONSTANT * xi**0.8 * n_returns**0.6)
