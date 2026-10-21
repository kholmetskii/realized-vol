"""The linear regression rule a + x @ b: least squares with an optional ridge penalty."""

from __future__ import annotations

import numpy as np

from rvol.models.arrays import FloatArray


class LinearRegression:
    """Predict a + x @ b, with (a, b) minimizing sum (y - a - x @ b)^2 + ridge * |b|^2.

    Special cases are choices of regressors and penalty, not separate rules:
    no regressors gives the mean of the training targets; ``ridge=0`` is ordinary
    least squares; a large ``ridge`` shrinks b towards zero, and the forecast
    towards the mean. The intercept is never penalized, and the penalty depends
    on the regressors' units, so they should share a scale.
    """

    def __init__(self, ridge: float = 0.0) -> None:
        if not np.isfinite(ridge) or ridge < 0:
            raise ValueError("ridge must be finite and non-negative")
        self.ridge = ridge

    def fit(self, features: FloatArray, target: FloatArray) -> LinearRegression:
        n_obs, n_regressors = features.shape
        if n_obs <= n_regressors:
            raise ValueError("linear regression needs more observations than regressors")
        if self.ridge == 0:
            # Least squares via SVD: numerically stable even for collinear regressors.
            design = np.column_stack([np.ones(n_obs), features])
            fitted, *_ = np.linalg.lstsq(design, target, rcond=None)
            self.intercept = float(fitted[0])
            self.coefficients = fitted[1:]
        else:
            # Centring removes the unpenalized intercept: b = (X'X + ridge I)^-1 X'y.
            x_mean = features.mean(axis=0)
            y_mean = float(target.mean())
            centered = features - x_mean
            gram = centered.T @ centered + self.ridge * np.eye(n_regressors)
            self.coefficients = np.linalg.solve(gram, centered.T @ (target - y_mean))
            self.intercept = float(y_mean - x_mean @ self.coefficients)
        return self

    def predict(self, features: FloatArray) -> FloatArray:
        return self.intercept + features @ self.coefficients
