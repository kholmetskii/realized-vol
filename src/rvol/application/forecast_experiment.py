"""Application service for model-agnostic walk-forward experiments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import cast

import numpy as np
import pandas as pd

from rvol.domain.config import ExperimentConfig
from rvol.domain.contracts import FittedForecaster, Forecaster, UpdatablePredictor
from rvol.domain.results import ExperimentResult, ForecastRecord


@dataclass(frozen=True)
class _ExecutionStep:
    position: int
    origin_date: date
    target_date: date
    n_available: int
    training_slice: slice
    refit: bool


class WalkForwardExperiment:
    """Execute injected models under common window and refit policies."""

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
        if (
            not frame["origin_date"].is_monotonic_increasing
            or frame["origin_date"].duplicated().any()
        ):
            raise ValueError("forecast origin dates must be strictly increasing")
        return frame

    def _training_slice(self, n_available: int) -> slice:
        selected = self.config.strategy.training_window.select(n_available)
        if not isinstance(selected, slice):
            raise ValueError("training window must select a contiguous slice")
        start = 0 if selected.start is None else selected.start
        stop = n_available if selected.stop is None else selected.stop
        if (
            selected.step not in (None, 1)
            or isinstance(start, bool)
            or isinstance(stop, bool)
            or not isinstance(start, int)
            or not isinstance(stop, int)
            or not 0 <= start <= stop <= n_available
        ):
            raise ValueError("training window must select chronological, available observations")
        return slice(start, stop)

    def _execution_plan(self, frame: pd.DataFrame) -> list[_ExecutionStep]:
        """Anchor eligible origins before applying the reporting start date."""
        targets = pd.DatetimeIndex(frame["target_date"])
        origins = pd.DatetimeIndex(frame["origin_date"])
        steps: list[_ExecutionStep] = []
        for position, (origin, target) in enumerate(zip(origins, targets, strict=True)):
            if self.config.forecast_end is not None and target.date() > self.config.forecast_end:
                break
            n_available = int(targets.searchsorted(origin, side="right"))
            training_slice = self._training_slice(n_available)
            if training_slice.stop - training_slice.start < self.config.min_train_size:
                continue
            due = self.config.strategy.retrain.should_refit(len(steps))
            steps.append(_ExecutionStep(
                position=position,
                origin_date=origin.date(),
                target_date=target.date(),
                n_available=n_available,
                training_slice=training_slice,
                refit=due or not steps,
            ))
        return steps

    def run(self, features: pd.DataFrame) -> ExperimentResult:
        """Fit or update at each origin and report forecasts within date bounds."""
        frame = self._validated_features(features)
        plan = self._execution_plan(frame)
        anchor = plan[0].origin_date if plan else None
        first_output = next((
            index for index, step in enumerate(plan)
            if self.config.forecast_start is None or step.target_date >= self.config.forecast_start
        ), None)
        if first_output is None:
            return ExperimentResult(config=self.config, strategy_anchor=anchor)

        # Earlier fits are superseded by the last scheduled fit before output.
        # Start there, then consume every subsequent observation exactly once.
        first_step = max(index for index in range(first_output + 1) if plan[index].refit)
        records: list[ForecastRecord] = []
        fitted_models: dict[str, FittedForecaster] = {}
        fit_step: _ExecutionStep | None = None
        last_available = 0
        for step in plan[first_step:]:
            if step.refit:
                training = frame.iloc[step.training_slice]
                for model in self.models:
                    training_features = training.loc[:, list(model.feature_names)].to_numpy(
                        dtype="float64",
                    )
                    fitted_models[model.name] = model.fit(
                        training_features, training["target"].to_numpy(dtype="float64"),
                    )
                fit_step = step
            elif step.n_available > last_available:
                observations = frame.iloc[last_available:step.n_available]
                for model in self.models:
                    fitted = fitted_models[model.name]
                    if isinstance(fitted, UpdatablePredictor):
                        fitted_models[model.name] = fitted.update(
                            observations.loc[:, list(model.feature_names)].to_numpy(
                                dtype="float64",
                            ),
                            observations["target"].to_numpy(dtype="float64"),
                        )
            last_available = step.n_available

            if (
                self.config.forecast_start is not None
                and step.target_date < self.config.forecast_start
            ):
                continue
            assert fit_step is not None
            train_start = cast(
                pd.Timestamp, frame["target_date"].iloc[fit_step.training_slice.start],
            )
            train_end = cast(
                pd.Timestamp, frame["target_date"].iloc[fit_step.training_slice.stop - 1],
            )
            actual_log_rv = float(frame["target"].iloc[step.position])
            for model in self.models:
                current_features = frame.loc[[step.position], list(model.feature_names)].to_numpy(
                    dtype="float64",
                )
                prediction = np.asarray(
                    fitted_models[model.name].predict(current_features), dtype="float64",
                )
                if prediction.shape != (1,) or not np.isfinite(prediction[0]):
                    raise ValueError(f"{model.name} must return one finite prediction per row")
                records.append(ForecastRecord(
                    model=model.name,
                    origin_date=step.origin_date,
                    target_date=step.target_date,
                    n_train=fit_step.training_slice.stop - fit_step.training_slice.start,
                    actual_log_rv=actual_log_rv,
                    predicted_log_rv=float(prediction[0]),
                    fit_date=fit_step.origin_date,
                    train_start_date=train_start.date(),
                    train_end_date=train_end.date(),
                ))

        return ExperimentResult(records=tuple(records), config=self.config, strategy_anchor=anchor)
