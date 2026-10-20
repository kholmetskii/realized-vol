"""Execute planned model fitting, observation updates and forecasts."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from rvol.models import ForecastModel, Predictor, UpdatablePredictor
from rvol.walkforward._validation import validate_run, validated_data
from rvol.walkforward.results import forecast_frame
from rvol.walkforward.walk_forward_plan import WalkForwardPlan
from rvol.walkforward.walk_forward_planner import WalkForwardPlanner


class WalkForwardRunner:
    """Execute a planner's steps with fresh fitted state on every call."""

    def __init__(self, planner: WalkForwardPlanner | None = None) -> None:
        self._planner = WalkForwardPlanner() if planner is None else planner

    @property
    def planner(self) -> WalkForwardPlanner:
        return self._planner

    def plan(self, data: pd.DataFrame) -> WalkForwardPlan:
        """Return the steps and execution settings that run() uses for these dates.

        Forecast tables hold forecasts only; record plan.execution explicitly
        when you need to document how forecasts were produced.
        """
        frame = validated_data(data, ())
        return self._plan(frame)

    def _plan(self, frame: pd.DataFrame) -> WalkForwardPlan:
        return self.planner.build(
            pd.DatetimeIndex(frame["origin_date"]), pd.DatetimeIndex(frame["target_date"])
        )

    def run(
        self,
        data: pd.DataFrame,
        *,
        model: ForecastModel,
        feature_columns: Sequence[str] = (),
        asset: str = "default",
    ) -> pd.DataFrame:
        """Forecast a ready table using the explicitly ordered feature columns.

        Required columns are origin_date, target_date, target and feature_columns.
        Feature engineering, target construction and transformations happen before
        this call. Every model receives only numerical arrays in the supplied order.
        """
        validate_run(model, feature_columns, asset)
        frame = validated_data(data, feature_columns)
        origins = pd.DatetimeIndex(frame["origin_date"])
        targets = pd.DatetimeIndex(frame["target_date"])
        plan = self._plan(frame)
        predictor: Predictor | None = None
        fit_details: dict[str, object] = {}
        rows = []
        columns = list(feature_columns)
        for step in plan.steps:
            position = step.position
            if step.refit:
                training = frame.iloc[step.training_rows]
                predictor = model.fit(
                    training[columns].to_numpy(dtype="float64", copy=True),
                    training["target"].to_numpy(dtype="float64", copy=True),
                )
                fit_details = {
                    "fit_date": origins[position],
                    "train_start_date": training["target_date"].iloc[0],
                    "train_end_date": training["target_date"].iloc[-1],
                    "n_train": len(training),
                }
            elif isinstance(predictor, UpdatablePredictor):
                observations = frame.iloc[step.update_rows]
                if not observations.empty:
                    predictor = predictor.update(
                        observations[columns].to_numpy(dtype="float64", copy=True),
                        observations["target"].to_numpy(dtype="float64", copy=True),
                    )
            if not step.emit_forecast:
                continue
            assert predictor is not None
            prediction = np.asarray(
                predictor.predict(
                    frame.loc[[position], columns].to_numpy(dtype="float64", copy=True)
                ),
                dtype="float64",
            )
            if prediction.shape != (1,) or not np.isfinite(prediction[0]):
                raise ValueError(f"{model.name} must return one finite prediction per row")
            rows.append(
                {
                    "model": model.name,
                    "origin_date": origins[position],
                    "target_date": targets[position],
                    "actual": float(frame["target"].iloc[position]),
                    "predicted": float(prediction[0]),
                    "asset": asset,
                    **fit_details,
                }
            )
        return forecast_frame(rows)
