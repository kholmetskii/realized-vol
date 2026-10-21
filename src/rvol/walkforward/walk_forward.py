"""Out-of-sample forecasts: refit at every origin on the outcomes known by then."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from rvol.models import ForecastModel

FORECAST_COLUMNS = (
    "model",
    "series",
    "origin_date",
    "target_date",
    "actual",
    "predicted",
    "n_train",
)


def walk_forward(
    data: pd.DataFrame,
    model: ForecastModel,
    *,
    min_train_size: int,
    start: date | None = None,
    end: date | None = None,
    series: str = "default",
) -> pd.DataFrame:
    """Forecast each target from a model fitted only on outcomes observed at its origin.

    ``data`` has one row per origin with ``origin_date``, ``target_date``,
    ``target`` and the model's regressor columns. At every origin the model's
    rule is refitted on all rows whose target_date is on or before that origin
    (an expanding window) and predicts the row's target. Origins with fewer than
    ``min_train_size`` training rows are skipped; ``start`` and ``end`` select
    target dates, inclusive, without changing any forecast.
    """
    columns = list(model.regressors)
    missing = {"origin_date", "target_date", "target", *columns}.difference(data.columns)
    if missing:
        raise ValueError(f"data is missing columns: {', '.join(sorted(missing))}")
    if (
        isinstance(min_train_size, bool)
        or not isinstance(min_train_size, int)
        or min_train_size < 1
    ):
        raise ValueError("min_train_size must be a positive integer")
    frame = data.sort_values("origin_date").reset_index(drop=True)
    origins = pd.DatetimeIndex(frame["origin_date"])
    targets = pd.DatetimeIndex(frame["target_date"])
    if (
        origins.hasnans
        or targets.hasnans
        or origins.has_duplicates
        or not (origins < targets).all()
        or not targets.is_monotonic_increasing
    ):
        raise ValueError("origins must be unique and precede their targets, which must be ordered")
    inputs = frame[columns].to_numpy(dtype="float64")
    outcomes = frame["target"].to_numpy(dtype="float64")
    if not (np.isfinite(inputs).all() and np.isfinite(outcomes).all()):
        raise ValueError("features and targets must be finite")

    rows = []
    for position, (origin, target) in enumerate(zip(origins, targets, strict=True)):
        if end is not None and target.date() > end:
            break
        if start is not None and target.date() < start:
            continue
        n_train = int(targets.searchsorted(origin, side="right"))
        if n_train < min_train_size:
            continue
        fitted = model.rule.fit(inputs[:n_train].copy(), outcomes[:n_train].copy())
        predicted = float(np.asarray(fitted.predict(inputs[position : position + 1].copy()))[0])
        if not np.isfinite(predicted):
            raise ValueError(f"{model.name} returned a non-finite prediction")
        rows.append(
            {
                "model": model.name,
                "series": series,
                "origin_date": origin,
                "target_date": target,
                "actual": outcomes[position],
                "predicted": predicted,
                "n_train": n_train,
            }
        )
    result = pd.DataFrame(rows, columns=list(FORECAST_COLUMNS))
    for column in ("origin_date", "target_date"):
        result[column] = pd.to_datetime(result[column])
    return result
