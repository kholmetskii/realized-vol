"""Heterogeneous autoregressive models of realized volatility."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from rvol.domain.contracts import FloatArray
from rvol.domain.specifications import ComponentSpecification
from rvol.models.linear import LinearPredictor, fit_ols

HAR_FEATURES = ("rv_daily", "rv_weekly", "rv_monthly")


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

    @property
    def specification(self) -> ComponentSpecification:
        return ComponentSpecification(
            name=self.name,
            implementation=f"{type(self).__module__}.{type(self).__qualname__}",
            feature_names=self.feature_names,
            parameters=(("variance_correction", "duan_smearing"),),
        )

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
