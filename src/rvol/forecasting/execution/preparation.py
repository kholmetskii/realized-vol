"""Align independently constructed features and future outcomes."""

from __future__ import annotations

import pandas as pd

from rvol.domain.targets import TargetHorizon
from rvol.forecasting.execution._data import target_horizon, validated_features

TARGET_COLUMNS = (
    "target_date", "target", "target_start_date", "horizon_sessions", "target_kind",
)


def prepare_forecast_data(
    features: pd.DataFrame, targets: pd.DataFrame, *,
    expected_horizon: TargetHorizon | None = None,
) -> pd.DataFrame:
    """Join by origin without changing the target timeline or availability dates.

    Builders may omit feature warm-up rows or other unavailable inputs, but
    targets must retain every origin, including incomplete trailing outcomes.
    Only rows with complete future outcomes are passed to an executor.
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
        raise ValueError("feature builders must not supply target columns")
    if set(targets.columns) != {"origin_date", *TARGET_COLUMNS}:
        raise ValueError("target builders must supply only origin and target columns")
    horizon = target_horizon(targets)
    if expected_horizon is not None and horizon != expected_horizon:
        raise ValueError("target metadata must match the configured target builder")
    inputs, outcomes = features.copy(), targets.copy()
    for name, frame in (("features", inputs), ("targets", outcomes)):
        frame["origin_date"] = pd.to_datetime(frame["origin_date"], errors="raise")
        if frame["origin_date"].isna().any() or frame["origin_date"].duplicated().any():
            raise ValueError(f"{name} must contain one valid row per origin date")
    for column in ("target_date", "target_start_date"):
        outcomes[column] = pd.to_datetime(outcomes[column], errors="raise")
    outcomes["target"] = pd.to_numeric(outcomes["target"], errors="raise")
    if not outcomes["target_date"].isna().equals(outcomes["target"].isna()):
        raise ValueError("target labels and end dates must be available together")
    # Apply the same interval/availability contract as the executors, including
    # outcomes at origins a feature builder omitted. No models are needed here.
    validated_features(outcomes.loc[outcomes["target_date"].notna()], ())
    if not inputs["origin_date"].isin(outcomes["origin_date"]).all():
        raise ValueError("every feature origin must exist in the target timeline")
    combined = inputs.merge(outcomes, on="origin_date", how="left", validate="one_to_one")
    combined = combined.loc[combined["target_date"].notna()]
    feature_columns = [name for name in inputs.columns if name != "origin_date"]
    result = combined.loc[:, [
        "origin_date", "target_date", *feature_columns, "target", "target_start_date",
        "horizon_sessions", "target_kind",
    ]].sort_values("origin_date").reset_index(drop=True)
    result.attrs["target_horizon"] = {"sessions": horizon.sessions, "kind": horizon.kind}
    return result
