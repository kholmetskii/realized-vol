"""Realized-variance metrics and the conversion of log-RV forecasts to variance.

Each step works in one explicit scale. Models forecast log realized variance;
variance_forecasts converts that table to the variance scale; metrics then
score whatever scale they receive, without transforming it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from rvol.backtesting.results import validated_forecasts
from rvol.evaluation.metrics import MSEMetric
from rvol.models.contracts import FloatArray


def _finite_vector(values: ArrayLike, name: str) -> NDArray[np.float64]:
    vector = np.asarray(values, dtype="float64")
    if vector.ndim != 1 or len(vector) == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def qlike(actual_rv: ArrayLike, forecast_rv: ArrayLike) -> NDArray[np.float64]:
    """Return per-observation QLIKE losses for positive variance forecasts."""
    actual = _finite_vector(actual_rv, "actual_rv")
    forecast = _finite_vector(forecast_rv, "forecast_rv")
    if actual.shape != forecast.shape:
        raise ValueError("actual_rv and forecast_rv must have the same shape")
    if np.any(actual <= 0) or np.any(forecast <= 0):
        raise ValueError("QLIKE requires strictly positive actual and forecast variances")

    ratio = actual / forecast
    # The expression is theoretically non-negative; clipping removes tiny
    # negative values caused only by floating-point cancellation near ratio=1.
    return np.maximum(ratio - np.log(ratio) - 1.0, 0.0)


class QLikeMetric:
    """QLIKE on variance-scale values; pass the output of variance_forecasts."""

    name = "QLIKE"

    def losses(self, actual: FloatArray, predicted: FloatArray) -> FloatArray:
        return qlike(actual, predicted)


class LogMSEMetric(MSEMetric):
    """Squared error of log realized-variance values, without rescaling."""

    name = "log-RV MSE"


def variance_forecasts(forecasts: pd.DataFrame, *, min_errors: int = 20) -> pd.DataFrame:
    """Convert log-RV forecasts to variance forecasts with Duan smearing.

    exp(predicted) estimates the conditional median of RV, not its mean, so it
    is biased low. Each forecast is multiplied by the mean of exp(actual -
    predicted) over that model's earlier forecasts whose target_date is on or
    before the origin_date, so only errors observable at the origin are used.

    Rows with fewer than ``min_errors`` observable errors are dropped. The
    returned table keeps the forecast columns, now on the variance scale, and
    adds the applied ``smearing_factor``.
    """
    if isinstance(min_errors, bool) or not isinstance(min_errors, int) or min_errors < 1:
        raise ValueError("min_errors must be a positive integer")
    frame = validated_forecasts(forecasts)
    converted = []
    for _, group in frame.groupby(["run_id", "fold_id", "model"], dropna=False, sort=False):
        ordered = group.sort_values("target_date")
        exp_errors = np.exp(ordered["actual"] - ordered["predicted"]).to_numpy(dtype="float64")
        observable = np.searchsorted(
            ordered["target_date"].to_numpy(),
            ordered["origin_date"].to_numpy(),
            side="right",
        )
        totals = np.concatenate([[0.0], np.cumsum(exp_errors)])
        factor = totals[observable] / np.maximum(observable, 1)
        keep = observable >= min_errors
        result = ordered.loc[keep].copy()
        result["smearing_factor"] = factor[keep]
        result["actual"] = np.exp(result["actual"])
        result["predicted"] = np.exp(result["predicted"]) * result["smearing_factor"]
        converted.append(result)
    combined = pd.concat(converted, ignore_index=True)
    if combined.empty:
        raise ValueError("no forecast has enough observable errors for smearing")
    return combined.sort_values(["model", "target_date"]).reset_index(drop=True)
