"""Leakage-safe walk-forward volatility forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rvol.models.har import HAR_FEATURES, fit_har

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


def _validated_features(features: pd.DataFrame) -> pd.DataFrame:
    required = {"origin_date", "target_date", "target", *HAR_FEATURES}
    missing = required.difference(features.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"forecast data is missing required columns: {names}")

    frame = features.loc[:, list(required)].copy()
    frame["origin_date"] = pd.to_datetime(frame["origin_date"], errors="coerce")
    frame["target_date"] = pd.to_datetime(frame["target_date"], errors="coerce")
    numeric = [*HAR_FEATURES, "target"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    valid = (
        frame["origin_date"].notna()
        & frame["target_date"].notna()
        & np.isfinite(frame[numeric]).all(axis=1)
    )
    if not valid.all():
        raise ValueError("forecast data contains invalid dates or non-finite values")
    if not (frame["origin_date"] < frame["target_date"]).all():
        raise ValueError("every forecast origin must precede its target date")

    frame = frame.sort_values("target_date").reset_index(drop=True)
    if frame["target_date"].duplicated().any():
        raise ValueError("forecast target dates must be unique")
    return frame


def walk_forward_forecasts(
    features: pd.DataFrame,
    *,
    min_train_size: int = 252,
    forecast_start: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Generate expanding-window naïve and HAR-RV forecasts.

    A training target is usable only when its target date is no later than the
    current forecast origin. This availability rule prevents the fitted model
    from seeing the outcome it is about to predict.
    """
    if min_train_size < len(HAR_FEATURES) + 1:
        raise ValueError("min_train_size must be at least four")
    frame = _validated_features(features)
    start = None if forecast_start is None else pd.Timestamp(forecast_start)
    rows: list[dict[str, object]] = []

    for _, current in frame.iterrows():
        if start is not None and current["target_date"] < start:
            continue
        training = frame[frame["target_date"] <= current["origin_date"]]
        if len(training) < min_train_size:
            continue

        model = fit_har(training)
        current_features = current.loc[list(HAR_FEATURES)].to_frame().T
        naive_log = float(current["rv_daily"])
        har_log = float(model.predict(current_features)[0])
        actual_log = float(current["target"])
        rows.append({
            "origin_date": current["origin_date"],
            "target_date": current["target_date"],
            "n_train": len(training),
            "actual_log_rv": actual_log,
            "naive_log_prediction": naive_log,
            "har_log_prediction": har_log,
            "actual_rv": float(np.exp(actual_log)),
            "naive_rv_prediction": float(np.exp(naive_log)),
            "har_rv_prediction": float(np.exp(har_log)),
        })

    return pd.DataFrame(rows, columns=FORECAST_COLUMNS)
