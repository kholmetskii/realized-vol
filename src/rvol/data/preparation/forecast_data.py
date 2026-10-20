"""Join features and targets into one ready forecasting table."""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_COLUMNS = ("target_date", "target")


def prepare_forecast_data(
    features: pd.DataFrame,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    """Join by origin without shifting labels across missing feature rows.

    Targets must retain the full observed timeline, including the final origin
    whose next outcome is unknown. Feature warm-up rows may be omitted.
    """
    for name, frame, required in (
        ("features", features, {"origin_date"}),
        ("targets", targets, {"origin_date", *TARGET_COLUMNS}),
    ):
        if not frame.columns.is_unique:
            raise ValueError(f"{name} columns must be unique")
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing required columns: {', '.join(sorted(missing))}")
    if set(TARGET_COLUMNS).intersection(features.columns):
        raise ValueError("feature data must not supply target columns")
    if set(targets.columns) != {"origin_date", *TARGET_COLUMNS}:
        raise ValueError("target data must supply only origin and target columns")
    inputs, outcomes = features.copy(), targets.copy()
    for name, frame in (("features", inputs), ("targets", outcomes)):
        frame["origin_date"] = pd.to_datetime(frame["origin_date"], errors="raise")
        if frame["origin_date"].isna().any() or frame["origin_date"].duplicated().any():
            raise ValueError(f"{name} must contain one valid row per origin date")
    feature_columns = [name for name in inputs.columns if name != "origin_date"]
    try:
        inputs[feature_columns] = inputs[feature_columns].apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError("feature data must contain only finite numerical values") from error
    if not np.isfinite(inputs[feature_columns].to_numpy(dtype="float64", na_value=np.nan)).all():
        raise ValueError("feature data must contain only finite numerical values")
    outcomes["target_date"] = pd.to_datetime(outcomes["target_date"], errors="raise")
    outcomes["target"] = pd.to_numeric(outcomes["target"], errors="raise")
    if not outcomes["target_date"].isna().equals(outcomes["target"].isna()):
        raise ValueError("target labels and dates must be available together")
    complete = outcomes.loc[outcomes["target_date"].notna()]
    if not np.isfinite(complete["target"].to_numpy(dtype="float64", na_value=np.nan)).all():
        raise ValueError("target data contains non-finite values")
    if not (complete["origin_date"] < complete["target_date"]).all():
        raise ValueError("every forecast origin must precede its target date")
    ordered = outcomes.sort_values("origin_date").reset_index(drop=True)
    if not ordered["target_date"].equals(ordered["origin_date"].shift(-1)):
        raise ValueError("targets must label the next observation on the full timeline")
    if not inputs["origin_date"].isin(outcomes["origin_date"]).all():
        raise ValueError("every feature origin must exist in the target timeline")
    combined = inputs.merge(outcomes, on="origin_date", how="left", validate="one_to_one")
    return (
        combined.loc[
            combined["target_date"].notna(),
            ["origin_date", "target_date", *feature_columns, "target"],
        ]
        .sort_values("origin_date")
        .reset_index(drop=True)
    )
