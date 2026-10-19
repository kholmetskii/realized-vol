"""Deterministic data and an intentionally slow reference forecast calculation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from rvol.data.preparation import build_next_observation_targets, prepare_forecast_data
from rvol.data.volatility.features import VarianceFeatureBuilder
from rvol.models import (
    AR1Model,
    EWMAModel,
    HARModel,
    HistoricalMeanModel,
    NaiveModel,
)
from rvol.models.contracts import ForecastModel, UpdatablePredictor


def daily_sample(n: int = 85) -> pd.DataFrame:
    steps = np.arange(n)
    values = -10 + 0.3 * np.sin(steps / 4) + np.random.default_rng(71).normal(0, 0.04, n)
    return pd.DataFrame({"date": pd.bdate_range("2023-01-02", periods=n), "log_rv": values})


ALL_FEATURES = ("log_rv_daily", "log_rv_weekly", "log_rv_monthly")


def benchmark_models() -> tuple[tuple[ForecastModel, tuple[str, ...]], ...]:
    return (
        (HistoricalMeanModel(), ()),
        (NaiveModel(), ("log_rv_daily",)),
        (EWMAModel(), ()),
        (AR1Model(), ("log_rv_daily",)),
        (HARModel(), ALL_FEATURES),
    )


def prepared_variance(
    *,
    n: int = 85,
    feature_columns: tuple[str, ...] = ALL_FEATURES,
) -> pd.DataFrame:
    daily = daily_sample(n)
    return prepare_forecast_data(
        VarianceFeatureBuilder(feature_names=feature_columns)(daily),
        build_next_observation_targets(daily, value_col="log_rv"),
    )


def reference_predictions(
    data: pd.DataFrame,
    model: ForecastModel,
    *,
    feature_columns: tuple[str, ...] = (),
    min_train_size: int = 15,
    window_size: int | None = None,
    refit_every: int = 5,
) -> pd.DataFrame:
    """Rebuild fitted state from scratch for every forecast, using date filters."""
    eligible: list[Any] = []
    row: Any
    for row in data.itertuples():
        history = data.loc[data.target_date <= row.origin_date]
        if window_size is not None:
            history = history.tail(window_size)
        if len(history) >= min_train_size:
            eligible.append(row)
    rows = []
    for index, row in enumerate(eligible):
        anchor = eligible[index - index % refit_every].origin_date
        training = data.loc[data.target_date <= anchor]
        if window_size is not None:
            training = training.tail(window_size)
        columns = list(feature_columns)
        predictor = model.fit(training[columns].to_numpy(), training.target.to_numpy())
        observed = data.loc[(data.target_date > anchor) & (data.target_date <= row.origin_date)]
        if isinstance(predictor, UpdatablePredictor) and len(observed):
            predictor = predictor.update(observed[columns].to_numpy(), observed.target.to_numpy())
        current = data.loc[[row.Index], columns].to_numpy()
        rows.append(
            {
                "target_date": row.target_date,
                "predicted": predictor.predict(current)[0],
                "fit_date": anchor,
                "n_train": len(training),
            }
        )
    return pd.DataFrame(rows)


def lag_features(
    data: pd.DataFrame,
    *,
    date_col: str = "date",
    value_col: str = "value",
    lags: tuple[int, ...] = (0,),
) -> pd.DataFrame:
    """Current (lag_0) and past values keyed by origin date, for engine tests on plain series."""
    ordered = data.sort_values(date_col).reset_index(drop=True)
    result = pd.DataFrame({"origin_date": pd.to_datetime(ordered[date_col])})
    for lag in lags:
        result[f"lag_{lag}"] = ordered[value_col].astype("float64").shift(lag)
    return result.dropna().reset_index(drop=True)
