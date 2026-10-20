"""The executable output of WalkForwardPlanner."""

from __future__ import annotations

from rvol.walkforward.walk_forward_step import WalkForwardStep


class WalkForwardPlan:
    """Executable steps and a snapshot of the settings that produced them."""

    def __init__(
        self,
        steps: tuple[WalkForwardStep, ...],
        execution: dict[str, object],
    ) -> None:
        self.steps = steps
        self.execution = execution
