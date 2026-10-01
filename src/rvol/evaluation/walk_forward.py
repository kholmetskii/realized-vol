"""Leakage-safe walk-forward volatility forecasts."""

from __future__ import annotations

from datetime import date

import pandas as pd

from rvol.application.forecast_experiment import WalkForwardExperiment
from rvol.domain.config import ExperimentConfig
from rvol.models.har import HAR_FEATURES, HARForecaster
from rvol.models.naive import NaiveForecaster

FORECAST_COLUMNS = [
    "origin_date",
    "target_date",
    "n_train",
    "actual_log_rv",
    "naive_log_prediction",
    "har_log_prediction",
    "actual_rv",
    "naive_rv_prediction",
    "har_rv_prediction",
]


def _as_date(value: str | date | pd.Timestamp | None) -> date | None:
    return None if value is None else pd.Timestamp(value).date()


def walk_forward_forecasts(
    features: pd.DataFrame,
    *,
    min_train_size: int = 252,
    forecast_start: str | date | pd.Timestamp | None = None,
    forecast_end: str | date | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Compatibility wrapper returning the original wide two-model table.

    New application code should use ``WalkForwardExperiment`` and its normalized
    ``ForecastRecord`` results directly.
    """
    if min_train_size < len(HAR_FEATURES) + 1:
        raise ValueError("min_train_size must be at least four")
    config = ExperimentConfig(
        min_train_size=min_train_size,
        forecast_start=_as_date(forecast_start),
        forecast_end=_as_date(forecast_end),
    )
    experiment = WalkForwardExperiment(
        models=(NaiveForecaster(), HARForecaster()),
        config=config,
    )
    result = experiment.run(features)
    by_model_and_date = {
        (record.model, record.target_date): record for record in result.records
    }
    target_dates = sorted({record.target_date for record in result.records})
    rows: list[dict[str, object]] = []
    for target_day in target_dates:
        naive = by_model_and_date[("naive", target_day)]
        har = by_model_and_date[("HAR", target_day)]
        rows.append({
            "origin_date": pd.Timestamp(har.origin_date),
            "target_date": pd.Timestamp(har.target_date),
            "n_train": har.n_train,
            "actual_log_rv": har.actual_log_rv,
            "naive_log_prediction": naive.predicted_log_rv,
            "har_log_prediction": har.predicted_log_rv,
            "actual_rv": har.actual_rv,
            "naive_rv_prediction": naive.predicted_rv,
            "har_rv_prediction": har.predicted_rv,
        })

    return pd.DataFrame(rows, columns=FORECAST_COLUMNS)
