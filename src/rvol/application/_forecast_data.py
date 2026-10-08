"""Shared DataFrame validation for forecast executors."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from rvol.domain.contracts import Forecaster


def validated_models(models: Sequence[Forecaster]) -> tuple[Forecaster, ...]:
    if not models:
        raise ValueError("at least one forecaster is required")
    names = [model.name for model in models]
    if len(names) != len(set(names)):
        raise ValueError("forecaster names must be unique")
    return tuple(models)


def validated_features(features: pd.DataFrame, models: Sequence[Forecaster]) -> pd.DataFrame:
    model_features = {
        feature_name
        for model in models
        for feature_name in model.feature_names
    }
    required = {"origin_date", "target_date", "target", *model_features}
    missing = required.difference(features.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"forecast data is missing required columns: {names}")

    columns = ["origin_date", "target_date", *sorted(model_features), "target"]
    frame = features.loc[:, columns].copy()
    frame["origin_date"] = pd.to_datetime(frame["origin_date"], errors="coerce")
    frame["target_date"] = pd.to_datetime(frame["target_date"], errors="coerce")
    numeric = [*sorted(model_features), "target"]
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
    if (
        not frame["origin_date"].is_monotonic_increasing
        or frame["origin_date"].duplicated().any()
    ):
        raise ValueError("forecast origin dates must be strictly increasing")
    return frame
