from datetime import date

import pandas as pd
import pytest

from rvol.walkforward import WalkForwardPlanner


@pytest.mark.parametrize(
    "window, expected_training",
    [
        ("expanding", [[0], [0], [0], [0, 1, 2, 3], [0, 1, 2, 3, 4]]),
        ("rolling", [[0], [0], [0], [1, 2, 3], [2, 3, 4]]),
    ],
)
def test_planner_handles_delayed_labels_and_multiple_new_observations(window, expected_training):
    origins = pd.DatetimeIndex(
        ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-07", "2026-01-08"]
    )
    targets = pd.DatetimeIndex(
        ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09"]
    )
    planner = WalkForwardPlanner(
        window=window,
        window_size=3 if window == "rolling" else None,
        min_train_size=1,
        refit_every=4,
    )

    plan = planner.build(origins, targets)

    positions = list(range(len(origins)))
    assert [step.position for step in plan.steps] == [1, 2, 3, 4, 5]
    assert [positions[step.training_rows] for step in plan.steps] == expected_training
    assert [positions[step.update_rows] for step in plan.steps] == [[0], [], [], [1, 2, 3], [4]]
    assert [step.refit for step in plan.steps] == [True, False, False, False, True]
    assert all(step.emit_forecast for step in plan.steps)
    for step in plan.steps:
        assert (targets[step.training_rows] <= origins[step.position]).all()
        assert (targets[step.update_rows] <= origins[step.position]).all()


def test_planner_retains_warmup_from_the_last_refit_and_inclusive_reporting_bounds():
    origins = pd.date_range("2026-10-01", periods=8)
    targets = pd.date_range("2026-10-02", periods=8)
    planner = WalkForwardPlanner(
        min_train_size=2,
        refit_every=3,
        forecast_start=date(2026, 10, 6),
        forecast_end=date(2026, 10, 8),
    )

    plan = planner.build(origins, targets)

    assert [step.position for step in plan.steps] == [2, 3, 4, 5, 6]
    assert [step.refit for step in plan.steps] == [True, False, False, True, False]
    assert [step.emit_forecast for step in plan.steps] == [False, False, True, True, True]
    assert plan.execution["schedule_anchor"] == "2026-10-03"
    assert [targets[step.position].date() for step in plan.steps if step.emit_forecast] == [
        date(2026, 10, 6),
        date(2026, 10, 7),
        date(2026, 10, 8),
    ]


@pytest.mark.parametrize(
    "settings, anchor",
    [
        ({"min_train_size": 10}, None),
        ({"min_train_size": 2, "forecast_start": date(2050, 1, 1)}, "2026-10-03"),
        ({"min_train_size": 2, "forecast_end": date(2026, 10, 2)}, None),
    ],
)
def test_plans_without_reportable_forecasts_preserve_execution_metadata(settings, anchor):
    plan = WalkForwardPlanner(**settings).build(
        pd.date_range("2026-10-01", periods=5),
        pd.date_range("2026-10-02", periods=5),
    )

    assert plan.steps == ()
    assert plan.execution["schedule_anchor"] == anchor
    assert plan.execution["min_train_size"] == settings["min_train_size"]


def test_planner_accepts_an_empty_validated_timeline():
    plan = WalkForwardPlanner().build(pd.DatetimeIndex([]), pd.DatetimeIndex([]))

    assert plan.steps == ()
    assert plan.execution["schedule_anchor"] is None


def test_replanning_applies_edited_settings_without_changing_existing_plans():
    origins = pd.date_range("2026-10-01", periods=8)
    targets = pd.date_range("2026-10-02", periods=8)
    planner = WalkForwardPlanner(min_train_size=2)
    original = planner.build(origins, targets)

    planner.window = "rolling"
    planner.window_size = 3
    changed = planner.build(origins, targets)

    assert original.execution["window"] == "expanding"
    assert original.steps[-1].training_rows == slice(0, 7)
    assert changed.execution["window"] == "rolling"
    assert changed.steps[-1].training_rows == slice(4, 7)

    planner.window_size = 1
    with pytest.raises(ValueError, match="at least min_train_size"):
        planner.build(origins, targets)


@pytest.mark.parametrize(
    "changes",
    [
        {"window": "unknown"},
        {"window": "rolling"},
        {"window_size": 25},
        {"window": "rolling", "window_size": 5},
        {"window": "rolling", "window_size": True},
        {"min_train_size": 0},
        {"min_train_size": True},
        {"min_train_size": 1.5},
        {"refit_every": 0},
        {"refit_every": True},
        {"refit_every": 1.5},
        {"forecast_start": "2024-01-01"},
        {"forecast_start": date(2024, 2, 1), "forecast_end": date(2024, 1, 1)},
    ],
)
def test_invalid_settings_are_rejected_when_constructing_a_planner(changes):
    with pytest.raises(ValueError):
        WalkForwardPlanner(**({"min_train_size": 15} | changes))
