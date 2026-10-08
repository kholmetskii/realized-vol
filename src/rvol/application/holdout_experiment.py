"""Chronological holdout execution with one fixed training cutoff."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
import pandas as pd

from rvol.application._forecast_data import target_horizon, validated_features, validated_models
from rvol.domain import (
    ExecutionMetadata,
    ExperimentConfig,
    ExperimentResult,
    Forecaster,
    ForecastRecord,
    HoldoutSettings,
    UpdatablePredictor,
)


class HoldoutExperiment:
    """Fit all models once, then predict chronologically with fresh features.

    Only labels with target_date <= train_end enter fitting. Optional state
    updates consume newly observed labels available at each forecast origin;
    fitted parameters and recorded training boundaries remain fixed.
    """

    def __init__(
        self, models: Sequence[Forecaster], config: ExperimentConfig[HoldoutSettings],
    ) -> None:
        if not isinstance(config.execution, HoldoutSettings):
            raise TypeError("holdout execution requires HoldoutSettings")
        self.models = validated_models(models)
        self.config = config

    def run(self, features: pd.DataFrame) -> ExperimentResult:
        frame = validated_features(features, self.models)
        horizon = target_horizon(frame)
        settings = self.config.execution
        execution = ExecutionMetadata(
            executor=f"{type(self).__module__}.{type(self).__qualname__}",
            settings=settings.specification,
            facts=(
                ("fit_cutoff", settings.train_end.isoformat()),
                ("observation_update_policy", (
                    "newly_available_targets_after_cutoff"
                    if settings.update_observations else "disabled"
                )),
            ),
        )
        targets = pd.DatetimeIndex(frame["target_date"])
        origins = pd.DatetimeIndex(frame["origin_date"])
        cutoff = pd.Timestamp(settings.train_end)
        n_train = int(targets.searchsorted(cutoff, side="right"))
        if n_train < settings.min_train_size:
            raise ValueError(
                f"holdout training contains {n_train} observations at {settings.train_end}; "
                f"requires at least {settings.min_train_size}"
            )
        positions = [
            position for position, (origin, target) in enumerate(zip(origins, targets, strict=True))
            if origin >= cutoff and target > cutoff
            and (self.config.forecast_start is None or target.date() >= self.config.forecast_start)
            and (self.config.forecast_end is None or target.date() <= self.config.forecast_end)
        ]
        if not positions:
            return ExperimentResult(
                config=self.config, execution=execution, identity=self.config.identity,
                target_horizon=horizon,
            )

        training = frame.iloc[:n_train]
        fitted_models = {
            model.name: model.fit(
                training.loc[:, list(model.feature_names)].to_numpy(dtype="float64"),
                training["target"].to_numpy(dtype="float64"),
            )
            for model in self.models
        }
        train_start = cast(pd.Timestamp, training["target_date"].iloc[0]).date()
        train_end = cast(pd.Timestamp, training["target_date"].iloc[-1]).date()
        last_available = n_train
        records: list[ForecastRecord] = []
        for position in positions:
            origin = origins[position]
            n_available = int(targets.searchsorted(origin, side="right"))
            if settings.update_observations and n_available > last_available:
                observations = frame.iloc[last_available:n_available]
                for model in self.models:
                    fitted = fitted_models[model.name]
                    if isinstance(fitted, UpdatablePredictor):
                        fitted_models[model.name] = fitted.update(
                            observations.loc[:, list(model.feature_names)].to_numpy(
                                dtype="float64",
                            ),
                            observations["target"].to_numpy(dtype="float64"),
                        )
            last_available = n_available
            for model in self.models:
                prediction = np.asarray(fitted_models[model.name].predict(
                    frame.loc[[position], list(model.feature_names)].to_numpy(dtype="float64"),
                ), dtype="float64")
                if prediction.shape != (1,) or not np.isfinite(prediction[0]):
                    raise ValueError(f"{model.name} must return one finite prediction per row")
                records.append(ForecastRecord(
                    model=model.name, origin_date=origin.date(),
                    target_date=targets[position].date(),
                    target_start_date=cast(
                        pd.Timestamp, frame["target_start_date"].iloc[position],
                    ).date(),
                    target_horizon=horizon,
                    n_train=n_train, actual_log_rv=float(frame["target"].iloc[position]),
                    predicted_log_rv=float(prediction[0]), fit_date=settings.train_end,
                    train_start_date=train_start, train_end_date=train_end,
                ))
        return ExperimentResult(
            records=tuple(records), config=self.config, execution=execution,
            identity=self.config.identity,
            target_horizon=horizon,
        )
