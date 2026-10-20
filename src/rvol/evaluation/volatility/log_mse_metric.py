"""Squared error of log realized variance, named for reports."""

from __future__ import annotations

from rvol.evaluation import MSEMetric


class LogMSEMetric(MSEMetric):
    """Squared error of log realized-variance values, without rescaling."""

    name = "log-RV MSE"
