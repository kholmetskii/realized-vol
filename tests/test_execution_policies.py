from dataclasses import FrozenInstanceError, replace
from typing import Any, cast

import pytest

from rvol.domain import (
    EveryNSessions,
    ExpandingWindow,
    RefitSchedule,
    RollingWindow,
    TrainingWindowPolicy,
    WalkForwardStrategy,
)


@pytest.mark.parametrize(
    "window, n_available, expected",
    [
        (ExpandingWindow(), 0, []),
        (ExpandingWindow(), 2, [0, 1]),
        (ExpandingWindow(), 6, [0, 1, 2, 3, 4, 5]),
        (RollingWindow(size=3), 0, []),
        (RollingWindow(size=3), 2, [0, 1]),
        (RollingWindow(size=3), 3, [0, 1, 2]),
        (RollingWindow(size=3), 6, [3, 4, 5]),
    ],
)
def test_windows_select_only_the_available_chronological_history(
    window: TrainingWindowPolicy, n_available: int, expected: list[int],
):
    history = list(range(10))

    selected = history[window.select(n_available)]

    assert isinstance(window, TrainingWindowPolicy)
    assert selected == expected
    assert history == list(range(10))


def test_rolling_window_counts_observations_across_calendar_gaps():
    sessions = ["2026-01-02", "2026-01-05", "2026-01-09", "2026-01-12"]

    assert sessions[RollingWindow(size=2).select(len(sessions))] == [
        "2026-01-09", "2026-01-12",
    ]


def test_refit_schedule_keeps_its_first_eligible_origin_anchor():
    schedule = EveryNSessions(5)

    assert isinstance(schedule, RefitSchedule)
    assert [index for index in range(15) if schedule.should_refit(index)] == [0, 5, 10]
    # A later reporting range continues to use the full dataset's origin indices.
    assert [index for index in range(7, 14) if schedule.should_refit(index)] == [10]
    assert schedule.should_refit(0)


def test_default_strategy_refits_every_origin_using_all_available_history():
    strategy = WalkForwardStrategy()

    assert list(range(8))[strategy.training_window.select(8)] == list(range(8))
    assert all(strategy.retrain.should_refit(index) for index in range(8))

    changed = replace(
        strategy, training_window=RollingWindow(3), retrain=EveryNSessions(5),
    )
    assert list(range(8))[changed.training_window.select(8)] == [5, 6, 7]
    assert not changed.retrain.should_refit(1)
    assert strategy.retrain.should_refit(1)
    with pytest.raises(FrozenInstanceError):
        cast(Any, strategy).retrain = EveryNSessions(10)


@pytest.mark.parametrize("policy", [RollingWindow, EveryNSessions])
@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_window_sizes_and_refit_intervals_require_positive_integers(policy, value):
    with pytest.raises(ValueError, match="integer of at least 1"):
        policy(value)


@pytest.mark.parametrize(
    "operation",
    [ExpandingWindow().select, RollingWindow(3).select, EveryNSessions(5).should_refit],
)
def test_policies_reject_negative_counts_and_origin_indices(operation):
    with pytest.raises(ValueError, match="integer of at least 0"):
        operation(-1)
