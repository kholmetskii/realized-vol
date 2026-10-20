"""Model-independent scheduling of chronological training and forecasts."""

from __future__ import annotations

from datetime import date
from typing import Literal

import pandas as pd

from rvol.walkforward.walk_forward_plan import WalkForwardPlan
from rvol.walkforward.walk_forward_step import WalkForwardStep


class WalkForwardPlanner:
    """Choose observable labels, training windows, refits and reporting bounds.

    Windows count prepared training pairs. Refits are anchored at the first
    eligible origin in the full history. Inclusive reporting bounds select
    target dates without changing that schedule.
    """

    def __init__(
        self,
        *,
        window: Literal["expanding", "rolling"] = "expanding",
        window_size: int | None = None,
        min_train_size: int = 252,
        refit_every: int = 1,
        forecast_start: date | None = None,
        forecast_end: date | None = None,
    ) -> None:
        self.window = window
        self.window_size = window_size
        self.min_train_size = min_train_size
        self.refit_every = refit_every
        self.forecast_start = forecast_start
        self.forecast_end = forecast_end
        self._validate_settings()

    def _validate_settings(self) -> None:
        if (
            isinstance(self.min_train_size, bool)
            or not isinstance(self.min_train_size, int)
            or self.min_train_size < 1
        ):
            raise ValueError("min_train_size must be a positive integer")
        for value in (self.forecast_start, self.forecast_end):
            if value is not None and type(value) is not date:
                raise ValueError("forecast bounds must be dates or None")
        if (
            self.forecast_start is not None
            and self.forecast_end is not None
            and self.forecast_start > self.forecast_end
        ):
            raise ValueError("forecast_start must not be after forecast_end")
        if self.window not in ("expanding", "rolling"):
            raise ValueError("window must be expanding or rolling")
        if (
            isinstance(self.refit_every, bool)
            or not isinstance(self.refit_every, int)
            or self.refit_every < 1
        ):
            raise ValueError("refit_every must be a positive integer")
        if self.window == "rolling":
            if (
                isinstance(self.window_size, bool)
                or not isinstance(self.window_size, int)
                or self.window_size < 1
            ):
                raise ValueError("rolling requires a positive integer window_size")
            if self.window_size < self.min_train_size:
                raise ValueError("window_size must be at least min_train_size")
        elif self.window_size is not None:
            raise ValueError("window_size only applies to rolling windows")

    def build(self, origins: pd.DatetimeIndex, targets: pd.DatetimeIndex) -> WalkForwardPlan:
        """Plan aligned, strictly increasing dates from validated input data."""
        self._validate_settings()
        steps: list[WalkForwardStep] = []
        last_available = 0
        for position, (origin, target) in enumerate(zip(origins, targets, strict=True)):
            if self.forecast_end is not None and target.date() > self.forecast_end:
                break
            available = int(targets.searchsorted(origin, side="right"))
            start = max(0, available - self.window_size) if self.window_size is not None else 0
            if available - start < self.min_train_size:
                continue
            steps.append(
                WalkForwardStep(
                    position=position,
                    training_rows=slice(start, available),
                    update_rows=slice(last_available, available),
                    refit=len(steps) % self.refit_every == 0,
                    emit_forecast=(
                        self.forecast_start is None or target.date() >= self.forecast_start
                    ),
                )
            )
            last_available = available
        execution: dict[str, object] = {
            "method": "walk_forward",
            "window": self.window,
            "window_size": self.window_size,
            "min_train_size": self.min_train_size,
            "refit_every": self.refit_every,
            "forecast_start": (
                None if self.forecast_start is None else self.forecast_start.isoformat()
            ),
            "forecast_end": None if self.forecast_end is None else self.forecast_end.isoformat(),
            "schedule_anchor": (
                None if not steps else origins[steps[0].position].date().isoformat()
            ),
            "observation_updates": "newly_available_targets_between_refits",
        }
        first_output = next((i for i, step in enumerate(steps) if step.emit_forecast), None)
        if first_output is None:
            return WalkForwardPlan((), execution)
        first_step = max(i for i in range(first_output + 1) if steps[i].refit)
        return WalkForwardPlan(tuple(steps[first_step:]), execution)
