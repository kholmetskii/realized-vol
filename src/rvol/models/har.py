"""Heterogeneous autoregressive models of realized volatility."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.special import logsumexp

from rvol.domain.contracts import FloatArray
from rvol.models.linear import LinearPredictor, fit_ols

HAR_FEATURES = ("rv_daily", "rv_weekly", "rv_monthly")


def _feature_matrix(frame: pd.DataFrame) -> NDArray[np.float64]:
    missing = set(HAR_FEATURES).difference(frame.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"HAR data is missing required features: {names}")
    matrix = frame.loc[:, HAR_FEATURES].to_numpy(dtype="float64")
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("HAR features must be finite numeric values")
    return matrix


@dataclass(frozen=True)
class HarModel:
    """Fitted HAR slopes and a variance-corrected log-scale intercept."""

    intercept: float
    daily: float
    weekly: float
    monthly: float

    @property
    def coefficients(self) -> NDArray[np.float64]:
        return np.array([self.daily, self.weekly, self.monthly], dtype="float64")

    def predict(self, features: pd.DataFrame) -> NDArray[np.float64]:
        """Return the log of the variance forecast for each feature row."""
        matrix = _feature_matrix(features)
        return self.intercept + matrix @ self.coefficients


class HARForecaster:
    """Fit log HAR-RV by OLS and apply Duan's training-residual correction.

    Return the log of the variance-scale forecast: the OLS log prediction plus
    log(mean(exp(training residuals))). The correction estimates mean variance
    when the training residual distribution represents forecast uncertainty.
    It is recomputed from each expanding training window, without using the
    current or future forecast outcomes.
    """

    name = "HAR"
    feature_names = HAR_FEATURES

    def fit(self, features: FloatArray, target: FloatArray) -> LinearPredictor:
        fitted = fit_ols(features, target, n_features=len(self.feature_names))
        residuals = np.asarray(target, dtype="float64") - fitted.predict(features)
        # Stay in log space: exponentiating large residuals can overflow even
        # when the final corrected variance is representable.
        log_correction = float(logsumexp(residuals) - np.log(len(residuals)))
        return LinearPredictor(
            intercept=fitted.intercept + log_correction,
            coefficients=fitted.coefficients,
        )


def fit_har(features: pd.DataFrame, *, target_col: str = "target") -> HarModel:
    """Fit HAR-RV by OLS with the training-residual variance correction."""
    matrix = _feature_matrix(features)
    if target_col not in features:
        raise ValueError(f"HAR data is missing target column: {target_col}")
    target = pd.to_numeric(features[target_col], errors="coerce").to_numpy(dtype="float64")
    fitted = HARForecaster().fit(matrix, target)
    return HarModel(
        intercept=fitted.intercept,
        daily=fitted.coefficients[0],
        weekly=fitted.coefficients[1],
        monthly=fitted.coefficients[2],
    )
