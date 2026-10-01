"""Metric strategy objects backed by pure forecast-loss functions."""

from __future__ import annotations

from rvol.domain.contracts import FloatArray, ForecastScale
from rvol.evaluation.losses import log_squared_error, qlike


class QLikeMetric:
    """QLIKE evaluated on the positive variance scale."""

    name = "QLIKE"
    scale: ForecastScale = "variance"

    def losses(self, actual: FloatArray, predicted: FloatArray) -> FloatArray:
        return qlike(actual, predicted)


class LogMSEMetric:
    """Squared error evaluated on the log realized-variance scale."""

    name = "log-RV MSE"
    scale: ForecastScale = "log_variance"

    def losses(self, actual: FloatArray, predicted: FloatArray) -> FloatArray:
        return log_squared_error(actual, predicted)
