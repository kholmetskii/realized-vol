"""Deterministic data and an intentionally slow reference forecast calculation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rvol import experiment
from rvol.data.preparation import (
    TrailingMeanFeatures,
    build_next_observation_targets,
    prepare_forecast_data,
)
from rvol.models import ForecastModel


def daily_sample(n: int = 85) -> pd.DataFrame:
    steps = np.arange(n)
    values = -10 + 0.3 * np.sin(steps / 4) + np.random.default_rng(71).normal(0, 0.04, n)
    return pd.DataFrame({"date": pd.bdate_range("2023-01-02", periods=n), "log_rv": values})


ALL_FEATURES = experiment.HAR_FEATURES
LATEST = experiment.LOG_RV_MEANS.column(1)
benchmark_models = experiment.benchmark_models


def har_features(names: tuple[str, ...] = ALL_FEATURES) -> TrailingMeanFeatures:
    """The study's log RV means restricted to the named columns, so the warm-up
    follows the longest window those columns use."""
    means = experiment.LOG_RV_MEANS
    windows = tuple(window for window in means.windows if means.column(window) in names)
    return TrailingMeanFeatures(means.value_col, windows, max_gap_days=means.max_gap_days)


def prepared_log_rv(
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
