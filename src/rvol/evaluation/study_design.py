"""A forecast comparison's fixed choices, and how they are applied to a series."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import pandas as pd

from rvol.evaluation.evaluation import Evaluation, evaluate
from rvol.evaluation.scoring import Loss
from rvol.evaluation.study_run import StudyRun
from rvol.models import ForecastModel
from rvol.walkforward import walk_forward_models


@dataclass(frozen=True)
class StudyDesign:
    """Choices fixed before any forecast is scored.

    Targets from ``forecast_start`` to ``forecast_end`` are forecast walk-forward,
    each fit using at least ``min_train_size`` observed outcomes. Forecasts before
    ``evaluation_start`` form the calibration window: they supply errors for the
    smearing factor and may tune a model, and are never scored. From
    ``evaluation_start`` on, forecasts are scored with ``loss`` and every model is
    tested against ``candidate``.
    """

    forecast_start: date
    evaluation_start: date
    forecast_end: date
    min_train_size: int
    candidate: str
    loss: Loss
    loss_name: str

    def __post_init__(self) -> None:
        if not self.forecast_start <= self.evaluation_start <= self.forecast_end:
            raise ValueError(
                "dates must satisfy forecast_start <= evaluation_start <= forecast_end"
            )
        if (
            isinstance(self.min_train_size, bool)
            or not isinstance(self.min_train_size, int)
            or self.min_train_size < 1
        ):
            raise ValueError("min_train_size must be a positive integer")

    def forecast(
        self, data: pd.DataFrame, models: Sequence[ForecastModel], *, series: str
    ) -> pd.DataFrame:
        """Walk-forward log forecasts from every model over the forecast dates."""
        return walk_forward_models(
            data,
            models,
            min_train_size=self.min_train_size,
            start=self.forecast_start,
            end=self.forecast_end,
            series=series,
        )

    def evaluate(self, log_forecasts: pd.DataFrame) -> Evaluation:
        """Score the evaluation period and test every model against the candidate."""
        return evaluate(
            log_forecasts,
            candidate=self.candidate,
            loss=self.loss,
            evaluation_start=self.evaluation_start,
            loss_name=self.loss_name,
        )

    def run(self, data: pd.DataFrame, models: Sequence[ForecastModel], *, series: str) -> StudyRun:
        """Forecast and evaluate ``models`` on one series; the candidate must be among them."""
        if self.candidate not in {model.name for model in models}:
            raise ValueError(f"the candidate {self.candidate!r} must be among the models")
        forecasts = self.forecast(data, models, series=series)
        return StudyRun(series, data, forecasts, self.evaluate(forecasts))
