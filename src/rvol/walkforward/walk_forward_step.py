"""One scheduled origin of a walk-forward plan."""

from __future__ import annotations


class WalkForwardStep:
    """Row selections and actions for one origin in an already validated table."""

    def __init__(
        self,
        *,
        position: int,
        training_rows: slice,
        update_rows: slice,
        refit: bool,
        emit_forecast: bool,
    ) -> None:
        self.position = position
        self.training_rows = training_rows
        self.update_rows = update_rows
        self.refit = refit
        self.emit_forecast = emit_forecast
