"""Compose the standard realized-volatility forecast experiment."""

from __future__ import annotations

import pandas as pd

from rvol.application import WalkForwardExperiment
from rvol.domain import ExperimentConfig, ExperimentResult, Forecaster
from rvol.features.forecasting import build_har_features
from rvol.models import (
    AR1Forecaster,
    HARForecaster,
    HistoricalMeanForecaster,
    NaiveForecaster,
)


def standard_forecasters() -> tuple[Forecaster, ...]:
    """Return fresh instances of the models used by standard project workflows."""
    return (
        HistoricalMeanForecaster(),
        NaiveForecaster(),
        AR1Forecaster(),
        HARForecaster(),
    )


def run_standard_forecast_experiment(
    daily: pd.DataFrame,
    config: ExperimentConfig,
) -> ExperimentResult:
    """Build HAR features and run the standard model set on common windows."""
    features = build_har_features(daily)
    return WalkForwardExperiment(
        models=standard_forecasters(),
        config=config,
    ).run(features)
