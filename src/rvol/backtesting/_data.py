"""DataFrame and option validation for walk-forward forecasting."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from rvol.models.contracts import ForecastModel


def validated_data(features: pd.DataFrame, feature_names: Sequence[str]) -> pd.DataFrame:
    model_features = set(feature_names)
    required = {"origin_date", "target_date", "target", *model_features}
    missing = required.difference(features.columns)
    if not features.columns.is_unique:
        raise ValueError("forecast data columns must be unique")
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"forecast data is missing required columns: {names}")
    columns = ["origin_date", "target_date", *sorted(model_features), "target"]
    frame = features.loc[:, columns].copy()
    for column in ("origin_date", "target_date"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
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
    if not frame["origin_date"].is_monotonic_increasing or frame["origin_date"].duplicated().any():
        raise ValueError("forecast origin dates must be strictly increasing")
    return frame


def validate_run(
    model: ForecastModel,
    feature_columns: Sequence[str],
    run_id: str,
    fold_id: str | None,
) -> None:
    for name, label in (("model name", model.name), ("run_id", run_id), ("fold_id", fold_id)):
        if (name != "fold_id" or label is not None) and (
            not isinstance(label, str) or not label.strip()
        ):
            raise ValueError(f"{name} must be a non-empty string")
    reserved = {
        "target",
        "origin_date",
        "target_date",
    }
    names = feature_columns
    if (
        isinstance(names, str)
        or any(not isinstance(n, str) or not n.strip() for n in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("feature columns must have unique non-empty names")
    if reserved.intersection(names):
        raise ValueError("model inputs must not include future targets or date columns")
