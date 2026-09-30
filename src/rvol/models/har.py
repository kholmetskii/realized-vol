"""Heterogeneous autoregressive models of realized volatility."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

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


def fit_har(features: pd.DataFrame, *, target_col: str = "target") -> HarModel:
    """Fit a log HAR-RV regression by ordinary least squares."""
    matrix = _feature_matrix(features)
    if target_col not in features:
        raise ValueError(f"HAR data is missing target column: {target_col}")
    target = pd.to_numeric(features[target_col], errors="coerce").to_numpy(dtype="float64")
    if len(target) < len(HAR_FEATURES) + 1:
        raise ValueError("HAR fitting requires at least four observations")
    if not np.all(np.isfinite(target)):
        raise ValueError("HAR target must contain finite numeric values")

    design = np.column_stack([np.ones(len(matrix)), matrix])
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    return HarModel(
        intercept=float(coefficients[0]),
        daily=float(coefficients[1]),
        weekly=float(coefficients[2]),
        monthly=float(coefficients[3]),
    )
