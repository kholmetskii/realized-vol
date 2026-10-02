"""Autoregressive benchmark models for realized volatility."""

from __future__ import annotations

from rvol.domain.contracts import FloatArray
from rvol.models.linear import LinearPredictor, fit_ols


class AR1Forecaster:
    """Regress next-session log RV on current-session log RV."""

    name = "AR1"
    feature_names = ("rv_daily",)

    def fit(self, features: FloatArray, target: FloatArray) -> LinearPredictor:
        return fit_ols(
            features,
            target,
            n_features=len(self.feature_names),
        )
