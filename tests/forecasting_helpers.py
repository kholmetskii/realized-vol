"""Deterministic data and an intentionally slow reference forecast calculation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rvol.data.preparation import (
    TrailingMeanFeatures,
    build_next_observation_targets,
    prepare_forecast_data,
)
from rvol.models import ExponentialSmoothing, ForecastModel, LinearRegression


def daily_sample(n: int = 85) -> pd.DataFrame:
    steps = np.arange(n)
    values = -10 + 0.3 * np.sin(steps / 4) + np.random.default_rng(71).normal(0, 0.04, n)
    return pd.DataFrame({"date": pd.bdate_range("2023-01-02", periods=n), "log_rv": values})


WINDOWS = {"log_rv_daily": 1, "log_rv_weekly": 5, "log_rv_monthly": 22}
ALL_FEATURES = tuple(WINDOWS)


def har_features(names: tuple[str, ...] = ALL_FEATURES) -> TrailingMeanFeatures:
    """The selected HAR inputs of the log_rv column, as in rvol.experiment."""
    return TrailingMeanFeatures(
        {name: WINDOWS[name] for name in names}, value_col="log_rv", max_gap_days=7
    )


def benchmark_models() -> tuple[ForecastModel, ...]:
    return (
        ForecastModel("historical_mean", LinearRegression()),
        ForecastModel("naive", ExponentialSmoothing(decay=0.0)),
        ForecastModel("EWMA", ExponentialSmoothing(decay=0.94)),
        ForecastModel("AR1", LinearRegression(), ("log_rv_daily",)),
        ForecastModel("HAR", LinearRegression(), ALL_FEATURES),
    )


def prepared_variance(
    *,
    n: int = 85,
    feature_columns: tuple[str, ...] = ALL_FEATURES,
) -> pd.DataFrame:
    daily = daily_sample(n)
    return prepare_forecast_data(
        har_features(feature_columns)(daily),
        build_next_observation_targets(daily, value_col="log_rv"),
    )


def reference_predictions(
    data: pd.DataFrame, model: ForecastModel, *, min_train_size: int = 15
) -> pd.DataFrame:
    """Refit from scratch for every forecast, selecting training rows by date."""
    columns = list(model.regressors)
    rows = []
    for row in data.itertuples():
        training = data.loc[data.target_date <= row.origin_date]
        if len(training) < min_train_size:
            continue
        fitted = model.rule.fit(training[columns].to_numpy(), training.target.to_numpy())
        current = data.loc[[row.Index], columns].to_numpy()
        rows.append(
            {
                "target_date": row.target_date,
                "predicted": fitted.predict(current)[0],
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
