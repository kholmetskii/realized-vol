"""Forecasting rules and the models built from them.

A rule says how a forecast is computed and estimated: LinearRegression (least
squares, optionally ridge-penalized) or ExponentialSmoothing (fixed weights). A
ForecastModel names a rule applied to chosen regressors; the historical mean,
AR(1) and HAR are all LinearRegression with different regressors.
"""

from rvol.models.arrays import FloatArray
from rvol.models.exponential_smoothing import ExponentialSmoothing
from rvol.models.forecast_model import ForecastModel
from rvol.models.forecast_rule import ForecastRule
from rvol.models.linear_regression import LinearRegression

__all__ = [
    "ExponentialSmoothing",
    "FloatArray",
    "ForecastModel",
    "ForecastRule",
    "LinearRegression",
]
