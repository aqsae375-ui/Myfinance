from datetime import date

import pytest

from src.advisor.goals import Goal
from src.advisor.projector import (
    GOAL_MET, INSUFFICIENT, NO_GOAL, OK, average_surplus, cumulative, months_between, months_to_reach, project,
    required_monthly,
)

AS_OF = date(2026, 1, 15)


# ---- formulas ------------------------------------------------------------------------------------

def test_average_surplus_needs_two_months():
    assert average_surplus([100.0, 200.0, 300.0]) == 200.0
    assert average_surplus([1788.0, -40.0]) == 874.0
    assert average_surplus([500.0]) is None and average_surplus([]) is None


def test_cumulative_is_surplus_times_months():
    assert cumulative(1251.3, 6) == 7507.8
    assert cumulative(1838.28, 12) == 22059.36
    assert cumulative(-50.0, 6) == -300.0


@pytest.mark.parametrize("remaining, surplus, expected", [
    (10000, 1251.3, 8), (10000, 1838.28, 6), (900, 300, 3), (901, 300, 4),
    (0, 300, 0), (-5, 300, 0), (100, 0, None), (100, -10, None), (0.3, 0.1, 3),
])
def test_months_to_reach(remaining, surplus, expected):
    assert months_to_reach(remaining, surplus) == expected


@pytest.mark.parametrize("start, end, expected", [
    (date(2026, 8, 28), date(2027, 3, 31), 7), (date(2026, 8, 28), date(2026, 9, 27), 0),
    (date(2026, 8, 28), date(2026, 9, 28), 1), (date(2026, 1, 31), date(2026, 2, 28), 0),
    (date(2026, 8, 28), date(2026, 8, 1), -1), (date(2026, 1, 15), date(2027, 1, 15), 12),
])
def test_months_between(start, end, expected):
    assert months_between(start, end) == expected


def test_required_monthly():
    assert required_monthly(10000, 7) == 1428.57
    assert required_monthly(0, 7) == 0.0 and required_monthly(-1, 0) == 0.0
    assert required_monthly(100, 0) is None and required_monthly(100, -2) is None


# ---- projection ----------------------------------------------------------------------------------

def test_target_goal_before_and_after():
    goal = Goal.by_date(6000, date(2026, 6, 15))  # 5 months left
    p = project([1000.0, 1000.0, 1000.0], 500.0, goal, AS_OF)
    assert p.status == OK
    assert (p.current_surplus, p.new_surplus, p.added_saving) == (1000.0, 1500.0, 500.0)
    assert p.cumulative_before == {6: 6000.0, 12: 12000.0}
    assert p.cumulative_after == {6: 9000.0, 12: 18000.0}
    assert p.plan_only == {6: 3000.0, 12: 6000.0}
    assert (p.remaining, p.months_left, p.required_monthly) == (6000.0, 5, 1200.0)
    assert (p.months_to_goal_before, p.on_track_before) == (6, False)
    assert (p.months_to_goal_after, p.on_track_after) == (4, True)
    assert p.numbers["months_to_goal_after"] == 4 and p.numbers["required_monthly"] == 1200.0


def test_saved_so_far_reduces_remaining():
    p = project([1000.0, 1000.0], 0.0, Goal.by_date(6000, date(2026, 6, 15), saved_so_far=2500), AS_OF)
    assert p.remaining == 3500.0 and p.months_to_goal_before == 4 and p.on_track_before


def test_target_goal_already_met():
    p = project([1000.0, 1000.0], 200.0, Goal.by_date(5000, date(2026, 6, 15), saved_so_far=5000), AS_OF)
    assert p.status == GOAL_MET
    assert p.remaining == 0.0 and p.required_monthly == 0.0
    assert p.months_to_goal_before == 0 and p.months_to_goal_after == 0
    # goal met needs no history
    assert project([], 0.0, Goal.by_date(100, date(2026, 6, 15), saved_so_far=150), AS_OF).status == GOAL_MET


def test_monthly_goal_met_and_not_met():
    met = project([1000.0, 1200.0], 0.0, Goal.monthly(1000), AS_OF)
    assert met.status == GOAL_MET and met.current_surplus == 1100.0 and met.on_track_before
    gap = project([1000.0, 1200.0], 500.0, Goal.monthly(1500), AS_OF)
    assert gap.status == OK and gap.required_monthly == 1500.0
    assert gap.on_track_before is False and gap.on_track_after is True
    assert gap.months_to_goal_before is None  # not meaningful for a monthly goal


def test_no_goal():
    p = project([800.0, 1200.0], 100.0, None, AS_OF)
    assert p.status == NO_GOAL and p.goal is None
    assert p.new_surplus == 1100.0 and p.cumulative_after[12] == 13200.0
    assert p.required_monthly is None and p.months_to_goal_before is None and p.on_track_before is None


def test_insufficient_data_one_month():
    p = project([1500.0], 100.0, Goal.by_date(6000, date(2026, 6, 15)), AS_OF)
    assert p.status == INSUFFICIENT
    assert p.current_surplus is None and p.new_surplus is None
    assert p.cumulative_before == {6: None, 12: None} and p.cumulative_after == {6: None, 12: None}
    assert p.plan_only == {6: 600.0, 12: 1200.0}   # the plan's own effect needs no history
    assert p.months_to_goal_before is None and p.required_monthly == 1200.0
    assert "Only 1 month(s) of history" in p.notes[0]
    assert project([1500.0], 0.0, None, AS_OF).status == INSUFFICIENT
    assert project([1500.0], 0.0, Goal.monthly(100), AS_OF).status == INSUFFICIENT


def test_deadline_passed_and_negative_surplus():
    late = project([1000.0, 1000.0], 0.0, Goal.by_date(5000, date(2025, 12, 1)), AS_OF)
    assert late.months_left < 0 and late.required_monthly is None
    assert late.on_track_before is False and late.on_track_after is False
    assert any("has passed" in n for n in late.notes)
    broke = project([-200.0, -100.0], 50.0, Goal.by_date(5000, date(2026, 12, 1)), AS_OF)
    assert broke.current_surplus == -150.0 and broke.new_surplus == -100.0
    assert broke.months_to_goal_before is None and broke.months_to_goal_after is None
    assert broke.on_track_after is False
