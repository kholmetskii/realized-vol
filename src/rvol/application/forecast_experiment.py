"""Application service for model-agnostic walk-forward experiments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
import pandas as pd

from rvol.domain.config import ExperimentConfig
from rvol.domain.contracts import Forecaster
from rvol.domain.results import ExperimentResult, ForecastRecord


class WalkForwardExperiment:
    """Fit injected models on expanding windows without future leakage."""

    def __init__(
        self,
        models: Sequence[Forecaster],
        config: ExperimentConfig,
    ) -> None:
        if not models:
            raise ValueError("at least one forecaster is required")
        names = [model.name for model in models]
        if len(names) != len(set(names)):
            raise ValueError("forecaster names must be unique")
        self.models = tuple(models)
        self.config = config

    def _validated_features(self, features: pd.DataFrame) -> pd.DataFrame:
        model_features = {
            feature_name
            for model in self.models
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
        return frame

    def run(self, features: pd.DataFrame) -> ExperimentResult:
        """Return normalized forecast records for every model and target date."""
        frame = self._validated_features(features)
        records: list[ForecastRecord] = []

        for position in range(len(frame)):
            origin = cast(pd.Timestamp, frame["origin_date"].iloc[position])
            target_date = cast(pd.Timestamp, frame["target_date"].iloc[position])
            target_day = target_date.date()
            if self.config.forecast_start is not None and target_day < self.config.forecast_start:
                continue
            if self.config.forecast_end is not None and target_day > self.config.forecast_end:
                continue

            training = frame[frame["target_date"] <= origin]
            if len(training) < self.config.min_train_size:
                continue
            actual_log_rv = float(frame["target"].iloc[position])

            for model in self.models:
                feature_names = list(model.feature_names)
                training_features = training.loc[:, feature_names].to_numpy(dtype="float64")
                training_target = training["target"].to_numpy(dtype="float64")
                fitted = model.fit(training_features, training_target)
                current_features = frame.loc[[position], feature_names].to_numpy(dtype="float64")
                prediction = np.asarray(fitted.predict(current_features), dtype="float64")
                if prediction.shape != (1,) or not np.isfinite(prediction[0]):
                    raise ValueError(f"{model.name} must return one finite prediction per row")
                predicted_log_rv = float(prediction[0])
                records.append(ForecastRecord(
                    model=model.name,
                    origin_date=origin.date(),
                    target_date=target_day,
                    n_train=len(training),
                    actual_log_rv=actual_log_rv,
                    predicted_log_rv=predicted_log_rv,
                    actual_rv=float(np.exp(actual_log_rv)),
                    predicted_rv=float(np.exp(predicted_log_rv)),
                ))

        return ExperimentResult(records=tuple(records))
