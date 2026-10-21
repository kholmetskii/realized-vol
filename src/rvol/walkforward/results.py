"""Forecast tables: one row per model, series and target date."""

from __future__ import annotations

import numpy as np
import pandas as pd

CASE_COLUMNS = (
    "model",
    "origin_date",
    "target_date",
    "series",
)


def validated_forecasts(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Copy and validate forecast records, allowing several series and samples."""
    required = {"model", "origin_date", "target_date", "actual", "predicted"}
    missing = required.difference(forecasts.columns)
    if not forecasts.columns.is_unique or missing:
        raise ValueError(f"forecast columns must be unique and include {sorted(required)}")
    if forecasts.empty:
        raise ValueError("forecasts must not be empty")
    frame = forecasts.copy()
    if "series" not in frame:
        frame["series"] = "default"
    for column in ("origin_date", "target_date"):
        frame[column] = pd.to_datetime(frame[column], errors="raise")
        if frame[column].isna().any():
            raise ValueError("forecast dates must be valid")
    for column in ("model", "series"):
        if (
            not frame[column]
            .map(lambda value: isinstance(value, str) and bool(value.strip()))
            .all()
        ):
            raise ValueError(f"{column} must contain non-empty strings")
    numbers = frame[["actual", "predicted"]].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numbers.to_numpy(dtype="float64")).all():
        raise ValueError("forecast values must be finite")
    frame[["actual", "predicted"]] = numbers
    if not (frame.origin_date < frame.target_date).all():
        raise ValueError("target dates must follow their forecast origins")
    if frame.duplicated(["series", "model", "target_date"]).any():
        raise ValueError("each model must have one forecast per target date within a series")
    frame = frame.sort_values(["model", "target_date"]).reset_index(drop=True)
    return frame
