"""The models compared, and the choice of the ridge penalty on the calibration window."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta

import pandas as pd

from rvol.evaluation import forecast_losses
from rvol.models import ExponentialSmoothing, ForecastModel, LinearRegression
from rvol.study.design import (
    EVALUATION_START,
    FORECAST_START,
    LOSS,
    LOSS_LABEL,
    MIN_TRAIN_SESSIONS,
    RIDGE_GRID,
)
from rvol.study.features import HAR_FEATURES, LOG_RV_MEANS, SHAR_FEATURES
from rvol.walkforward import smeared_forecasts, walk_forward


def benchmark_models() -> tuple[ForecastModel, ...]:
    """The models with fixed settings: two forecasting rules with different regressors."""
    return (
        ForecastModel("historical_mean", LinearRegression()),  # no regressors: the mean
        ForecastModel("naive", ExponentialSmoothing(decay=0.0)),  # tomorrow equals today
        # 0.94 is RiskMetrics' daily decay for squared returns; here it smooths log RV.
        ForecastModel("EWMA", ExponentialSmoothing(decay=0.94)),
        ForecastModel("AR1", LinearRegression(), (LOG_RV_MEANS.column(1),)),
        ForecastModel("HAR", LinearRegression(), HAR_FEATURES),
        ForecastModel("SHAR", LinearRegression(), SHAR_FEATURES),
    )


def ridge_har(penalty: float) -> ForecastModel:
    """HAR's regressors fitted with a ridge penalty on the slopes."""
    return ForecastModel("HAR-ridge", LinearRegression(ridge=penalty), HAR_FEATURES)


def ridge_calibration(
    prepared: pd.DataFrame,
    grid: Sequence[float] = RIDGE_GRID,
    *,
    start: date = FORECAST_START,
    end: date = EVALUATION_START,
    min_train_size: int = MIN_TRAIN_SESSIONS,
) -> pd.Series:
    """Mean loss of ridge HAR for each penalty, over targets in [start, end).

    The forecasts are made walk-forward and smeared exactly as in the study, so
    the score uses only targets before ``end``: by default the calibration
    window, which the evaluation never scores.
    """
    if not grid:
        raise ValueError("at least one penalty is required")
    scores = {}
    for penalty in grid:
        log_forecasts = walk_forward(
            prepared,
            ridge_har(penalty),
            min_train_size=min_train_size,
            start=start,
            end=end - timedelta(days=1),
        )
        losses = forecast_losses(smeared_forecasts(log_forecasts), LOSS)
        scores[penalty] = float(losses["loss"].mean())
    return pd.Series(scores, name=f"calibration {LOSS_LABEL}").rename_axis("ridge penalty")


def calibrated_models(calibration: pd.Series) -> tuple[ForecastModel, ...]:
    """The benchmark models and ridge HAR with the penalty of lowest calibration loss."""
    return (*benchmark_models(), ridge_har(float(calibration.idxmin())))


def study_models(prepared: pd.DataFrame) -> tuple[ForecastModel, ...]:
    """Every model of the study, with the ridge penalty chosen on ``prepared``."""
    return calibrated_models(ridge_calibration(prepared))
