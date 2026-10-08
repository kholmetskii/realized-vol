"""Autoregressive benchmark models for realized volatility."""

from __future__ import annotations

from rvol.domain.contracts import FloatArray
from rvol.domain.specifications import ComponentSpecification
from rvol.models.linear import LinearPredictor, fit_ols


class AR1Forecaster:
    """Regress next-session log RV on current-session log RV.

    Return the OLS estimate of mean log variance. Exponentiating gives a
    geometric-mean variance forecast under the log regression model.
    """

    name = "AR1"
    feature_names = ("rv_daily",)

    @property
    def specification(self) -> ComponentSpecification:
        return ComponentSpecification(
            name=self.name,
            implementation=f"{type(self).__module__}.{type(self).__qualname__}",
            feature_names=self.feature_names,
        )

    def fit(self, features: FloatArray, target: FloatArray) -> LinearPredictor:
        return fit_ols(
            features,
            target,
            n_features=len(self.feature_names),
        )
