"""Forecasts are DataFrames; these helpers validate their common columns."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

FORECAST_COLUMNS = (
    "model",
    "origin_date",
    "target_date",
    "n_train",
    "actual",
    "predicted",
    "fit_date",
    "train_start_date",
    "train_end_date",
    "series",
)
CASE_COLUMNS = (
    "model",
    "origin_date",
    "target_date",
    "series",
)


def forecast_frame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    result = pd.DataFrame(rows, columns=FORECAST_COLUMNS)
    for column in (
        "origin_date",
        "target_date",
        "fit_date",
        "train_start_date",
        "train_end_date",
    ):
        result[column] = pd.to_datetime(result[column])
    return result


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
    if "n_train" in frame and (
        not pd.to_numeric(frame.n_train, errors="raise")
        .map(lambda value: not isinstance(value, bool) and value >= 1 and int(value) == value)
        .all()
    ):
        raise ValueError("n_train must contain positive integers")
    fit_columns = {"fit_date", "train_start_date", "train_end_date"}
    present = fit_columns.intersection(frame.columns)
    if present and present != fit_columns:
        raise ValueError("fit date and training boundaries must be provided together")
    if present:
        for column in sorted(fit_columns):
            frame[column] = pd.to_datetime(frame[column], errors="raise")
        if (
            frame[list(fit_columns)].isna().any().any()
            or not (
                (frame.train_start_date <= frame.train_end_date)
                & (frame.train_end_date <= frame.fit_date)
                & (frame.fit_date <= frame.origin_date)
            ).all()
        ):
            raise ValueError("training boundaries must precede fitting and forecasting")
    frame = frame.sort_values(["model", "target_date"]).reset_index(drop=True)
    return frame
