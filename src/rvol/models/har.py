"""Heterogeneous autoregressive models of realized volatility."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

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
    """OLS coefficients for a log realized-variance HAR model."""

    intercept: float
    daily: float
    weekly: float
    monthly: float

    @property
    def coefficients(self) -> NDArray[np.float64]:
        return np.array([self.daily, self.weekly, self.monthly], dtype="float64")

    def predict(self, features: pd.DataFrame) -> NDArray[np.float64]:
        """Predict log realized variance for one or more feature rows."""
        matrix = _feature_matrix(features)
        return self.intercept + matrix @ self.coefficients


class HARForecaster:
    """Fit daily, weekly, and monthly log-RV components by OLS."""

    name = "HAR"
    feature_names = HAR_FEATURES

    def fit(self, features: FloatArray, target: FloatArray) -> LinearPredictor:
        return fit_ols(
            features,
            target,
            n_features=len(self.feature_names),
        )


def fit_har(features: pd.DataFrame, *, target_col: str = "target") -> HarModel:
    """Fit a log HAR-RV regression by ordinary least squares."""
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
