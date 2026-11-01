"""Deterministic data and an intentionally slow reference forecast calculation."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np
import pandas as pd

from rvol import study
from rvol.data.preparation import (
    build_next_observation_targets,
    join_features,
    prepare_forecast_data,
)
from rvol.models import ForecastModel


def with_semivariances(daily: pd.DataFrame, seed: int = 72) -> pd.DataFrame:
    """Add log semivariances that split each session's RV at a random share."""
    share = np.random.default_rng(seed).uniform(0.3, 0.7, len(daily))
    return daily.assign(
        log_rs_pos=daily["log_rv"] + np.log(share),
        log_rs_neg=daily["log_rv"] + np.log1p(-share),
    )


def daily_sample(n: int = 85) -> pd.DataFrame:
    steps = np.arange(n)
    values = -10 + 0.3 * np.sin(steps / 4) + np.random.default_rng(71).normal(0, 0.04, n)
    daily = pd.DataFrame({"date": pd.bdate_range("2023-01-02", periods=n), "log_rv": values})
    return with_semivariances(daily)


ALL_FEATURES = tuple(column for table in study.FEATURE_TABLES for column in table.columns)
LATEST = study.LOG_RV_MEANS.column(1)
benchmark_models = study.benchmark_models


def study_features(names: tuple[str, ...] = ALL_FEATURES) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """The study's feature tables restricted to the named columns, so the
    warm-up follows the longest window those columns use."""
    builders = []
    for table in study.FEATURE_TABLES:
        windows = tuple(window for window in table.windows if table.column(window) in names)
        if windows:
            builders.append(dataclasses.replace(table, windows=windows))
    if not builders:  # no regressors: keep the full origin timeline
        builders.append(dataclasses.replace(study.LOG_RV_MEANS, windows=()))
    return lambda daily: join_features([build(daily) for build in builders])


def prepared_log_rv(
    *,
    n: int = 85,
    feature_columns: tuple[str, ...] = ALL_FEATURES,
) -> pd.DataFrame:
    daily = daily_sample(n)
    return prepare_forecast_data(
        study_features(feature_columns)(daily),
        build_next_observation_targets(daily, value_col=study.LOG_RV),
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
