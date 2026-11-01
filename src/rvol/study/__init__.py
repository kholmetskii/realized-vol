"""The realized-variance forecast study: prepare, forecast, evaluate.

Call these functions in order:

    prepared = prepare(daily)
    forecasts = forecast(prepared, study_models(prepared), asset="EURUSD")
    comparison = compare(forecasts)
    results_table(comparison)

Models forecast next-session log RV. ``compare`` converts those forecasts to the
variance scale with Duan smearing and scores them with ``LOSS`` and DM tests.

This is the one package that joins the generic forecasting tools to the
volatility data. ``design`` holds the study's fixed choices, ``features`` the
target and inputs, ``models`` the models compared, ``pipeline`` the steps.
"""

from rvol.study.design import (
    ASSETS,
    CANDIDATE,
    EVALUATION_START,
    FORECAST_END,
    FORECAST_START,
    LOSS,
    LOSS_LABEL,
    MAX_GAP_DAYS,
    MIN_TRAIN_SESSIONS,
    RIDGE_GRID,
)
from rvol.study.features import (
    FEATURE_TABLES,
    HAR_FEATURES,
    HAR_WINDOWS,
    LOG_RS_NEG,
    LOG_RS_POS,
    LOG_RV,
    LOG_RV_MEANS,
    SHAR_FEATURES,
    prepare,
)
from rvol.study.models import (
    benchmark_models,
    calibrated_models,
    ridge_calibration,
    ridge_har,
    study_models,
)
from rvol.study.pipeline import Comparison, compare, forecast, results_table

__all__ = [
    "ASSETS",
    "CANDIDATE",
    "EVALUATION_START",
    "FEATURE_TABLES",
    "FORECAST_END",
    "FORECAST_START",
    "HAR_FEATURES",
    "HAR_WINDOWS",
    "LOG_RS_NEG",
    "LOG_RS_POS",
    "LOG_RV",
    "LOG_RV_MEANS",
    "LOSS",
    "LOSS_LABEL",
    "MAX_GAP_DAYS",
    "MIN_TRAIN_SESSIONS",
    "RIDGE_GRID",
    "SHAR_FEATURES",
    "Comparison",
    "benchmark_models",
    "calibrated_models",
    "compare",
    "forecast",
    "prepare",
    "results_table",
    "ridge_calibration",
    "ridge_har",
    "study_models",
]
