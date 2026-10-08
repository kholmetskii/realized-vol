"""Shared DataFrame validation for forecast executors."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from rvol.domain.contracts import Forecaster
from rvol.domain.targets import TargetHorizon


def target_horizon(features: pd.DataFrame) -> TargetHorizon:
    """Read one explicit target definition, retaining one-session legacy inputs."""
    metadata_columns = {"horizon_sessions", "target_kind"}
    present = metadata_columns.intersection(features.columns)
    if not present:
        return TargetHorizon()
    if present != metadata_columns:
        raise ValueError("forecast target metadata requires horizon_sessions and target_kind")
    if features.empty:
        metadata = features.attrs.get("target_horizon")
        if not isinstance(metadata, dict) or set(metadata) != {"sessions", "kind"}:
            raise ValueError("empty forecast data must declare target_horizon metadata")
        return TargetHorizon(metadata["sessions"], metadata["kind"])
    values = features.loc[:, ["horizon_sessions", "target_kind"]].drop_duplicates()
    if len(values) != 1:
        raise ValueError("forecast data must contain one consistent target horizon")
    sessions = values["horizon_sessions"].iloc[0]
    if isinstance(sessions, bool) or not isinstance(sessions, (int, np.integer)):
        raise ValueError("horizon sessions must be a positive integer")
    return TargetHorizon(int(sessions), values["target_kind"].iloc[0])


def validated_models(models: Sequence[Forecaster]) -> tuple[Forecaster, ...]:
    if not models:
        raise ValueError("at least one forecaster is required")
    names = [model.name for model in models]
    if len(names) != len(set(names)):
        raise ValueError("forecaster names must be unique")
    return tuple(models)


def validated_features(features: pd.DataFrame, models: Sequence[Forecaster]) -> pd.DataFrame:
    horizon = target_horizon(features)
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
    if "target_start_date" in features:
        columns.append("target_start_date")
    elif horizon.kind == "cumulative" and horizon.sessions > 1:
        raise ValueError("cumulative forecast data must include target_start_date")
    frame = features.loc[:, columns].copy()
    frame["origin_date"] = pd.to_datetime(frame["origin_date"], errors="coerce")
    frame["target_date"] = pd.to_datetime(frame["target_date"], errors="coerce")
    frame["target_start_date"] = pd.to_datetime(
        frame.get("target_start_date", frame["target_date"]), errors="coerce",
    )
    numeric = [*sorted(model_features), "target"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    valid = (
        frame["origin_date"].notna()
        & frame["target_date"].notna()
        & frame["target_start_date"].notna()
        & np.isfinite(frame[numeric]).all(axis=1)
    )
    if not valid.all():
        raise ValueError("forecast data contains invalid dates or non-finite values")
    if not (frame["origin_date"] < frame["target_date"]).all():
        raise ValueError("every forecast origin must precede its target date")
    if not (
        (frame["origin_date"] < frame["target_start_date"])
        & (frame["target_start_date"] <= frame["target_date"])
    ).all():
        raise ValueError("target start must follow the origin and not exceed target end")
    if (horizon.kind == "session" or horizon.sessions == 1) and not (
        frame["target_start_date"] == frame["target_date"]
    ).all():
        raise ValueError("single-session targets must start and end on the target date")
    if horizon.kind == "cumulative" and horizon.sessions > 1 and not (
        frame["target_start_date"] < frame["target_date"]
    ).all():
        raise ValueError("multi-session cumulative targets must start before their end")

    frame = frame.sort_values("target_date").reset_index(drop=True)
    if frame["target_date"].duplicated().any():
        raise ValueError("forecast target dates must be unique")
    if (
        not frame["origin_date"].is_monotonic_increasing
        or frame["origin_date"].duplicated().any()
    ):
        raise ValueError("forecast origin dates must be strictly increasing")
    frame["horizon_sessions"] = horizon.sessions
    frame["target_kind"] = horizon.kind
    frame.attrs["target_horizon"] = {"sessions": horizon.sessions, "kind": horizon.kind}
    return frame
