"""Before/after projection for a set of accepted opportunities. Pure arithmetic: no DB, no LLM.

Definitions (also returned in `Projection.method`):
- current surplus   = average monthly net saved (income - spending) over the months of history
- new surplus       = current surplus + sum of the accepted opportunities' monthly savings
- cumulative N mo.  = surplus * N (no interest, no inflation)
- months to goal    = ceil(remaining / surplus); never (None) if surplus <= 0
- months left       = whole calendar months from the as-of date to the goal's target date
- required monthly  = remaining / months left (target_by_date) or the goal amount (monthly_savings)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Sequence

from src.advisor.goals import BY_DATE, MONTHLY, Goal

MIN_HISTORY_MONTHS = 2   # one month is not a pattern: no "current path" is projected from it
HORIZONS = (6, 12)

METHOD = ("current surplus = average monthly (income - spending); new surplus = current + accepted savings; "
          "cumulative = surplus x months; months to goal = ceil(remaining / surplus)")

OK, NO_GOAL, GOAL_MET, INSUFFICIENT = "ok", "no_goal", "goal_met", "insufficient_data"


def _r(x: float) -> float:
    return round(float(x), 2)


# ---- formulas (unit-tested directly) -------------------------------------------------------------

def average_surplus(net_by_month: Sequence[float]) -> float | None:
    """Mean monthly net saved; None with fewer than MIN_HISTORY_MONTHS months."""
    if len(net_by_month) < MIN_HISTORY_MONTHS:
        return None
    return _r(sum(net_by_month) / len(net_by_month))


def cumulative(surplus: float, months: int) -> float:
    return _r(surplus * months)


def cumulative_path(surplus: float, months: int = 12) -> list[float]:
    """Cumulative saving at the end of month 0..`months` (0 first), for the before/after chart."""
    return [cumulative(surplus, n) for n in range(months + 1)]


def months_to_reach(remaining: float, surplus: float) -> int | None:
    """Whole months to save `remaining` at `surplus` a month: 0 if nothing remains, None if never."""
    if remaining <= 0:
        return 0
    if surplus <= 0:
        return None
    return math.ceil(round(remaining / surplus, 9))  # round first so 3.0000000001 doesn't become 4


def months_between(start: date, end: date) -> int:
    """Whole calendar months from `start` to `end` (negative if `end` is earlier):
    2026-08-28 -> 2027-03-31 is 7; 2026-08-28 -> 2026-09-27 is 0."""
    return (end.year - start.year) * 12 + (end.month - start.month) - (end.day < start.day)


def required_monthly(remaining: float, months_left: int) -> float | None:
    """Saving needed per month to hit the target on time; None when the deadline has passed."""
    if remaining <= 0:
        return 0.0
    if months_left <= 0:
        return None
    return _r(remaining / months_left)


# ---- projection ----------------------------------------------------------------------------------

@dataclass(frozen=True)
class Projection:
    status: str                         # ok | no_goal | goal_met | insufficient_data
    months_of_history: int
    added_saving: float                 # sum of accepted monthly savings
    current_surplus: float | None       # None with insufficient data
    new_surplus: float | None
    cumulative_before: dict[int, float | None]  # horizon (6, 12) -> Rs saved on the current path
    cumulative_after: dict[int, float | None]   # same, with the accepted opportunities
    plan_only: dict[int, float]         # horizon -> Rs the accepted opportunities alone add
    goal: Goal | None = None
    remaining: float | None = None      # target_by_date: amount - saved_so_far
    months_left: int | None = None      # target_by_date: months to the target date
    required_monthly: float | None = None
    months_to_goal_before: int | None = None  # target_by_date; None = never at this rate
    months_to_goal_after: int | None = None
    on_track_before: bool | None = None
    on_track_after: bool | None = None
    notes: tuple[str, ...] = ()
    method: str = METHOD
    numbers: dict[str, float] = field(default_factory=dict)  # every figure above, for verification

    @property
    def months_saved(self) -> int | None:
        """How many months sooner the goal is reached with the plan (target_by_date; None if not comparable)."""
        if self.months_to_goal_before is None or self.months_to_goal_after is None:
            return None
        return self.months_to_goal_before - self.months_to_goal_after


def project(net_by_month: Sequence[float], added_saving: float, goal: Goal | None, as_of: date) -> Projection:
    """Project the current path vs the path with `added_saving` more each month."""
    added = _r(added_saving)
    current = average_surplus(net_by_month)
    new = _r(current + added) if current is not None else None
    before = {h: cumulative(current, h) if current is not None else None for h in HORIZONS}
    after = {h: cumulative(new, h) if new is not None else None for h in HORIZONS}
    plan_only = {h: cumulative(added, h) for h in HORIZONS}
    notes: list[str] = []
    fields: dict = {}

    if current is None:
        notes.append(f"Only {len(net_by_month)} month(s) of history: at least {MIN_HISTORY_MONTHS} are needed "
                     f"to project your current path.")

    if goal is not None and goal.kind == BY_DATE:
        remaining = _r(goal.amount - goal.saved_so_far)
        months_left = months_between(as_of, goal.target_date)
        fields.update(remaining=max(remaining, 0.0), months_left=months_left,
                      required_monthly=required_monthly(remaining, months_left))
        if remaining > 0 and months_left <= 0:
            notes.append(f"The target date {goal.target_date.isoformat()} has passed.")
        if current is not None:
            mb, ma = months_to_reach(remaining, current), months_to_reach(remaining, new)
            fields.update(months_to_goal_before=mb, months_to_goal_after=ma,
                          on_track_before=mb is not None and mb <= months_left,
                          on_track_after=ma is not None and ma <= months_left)
        status = GOAL_MET if remaining <= 0 else (INSUFFICIENT if current is None else OK)
    elif goal is not None and goal.kind == MONTHLY:
        fields.update(required_monthly=goal.amount)
        if current is not None:
            fields.update(on_track_before=current >= goal.amount, on_track_after=new >= goal.amount)
        status = INSUFFICIENT if current is None else (GOAL_MET if current >= goal.amount else OK)
    else:
        status = INSUFFICIENT if current is None else NO_GOAL

    numbers = {"added_saving": added, "months_of_history": len(net_by_month)}
    for key, value in [("current_surplus", current), ("new_surplus", new),
                       *[(f"before_{h}m", v) for h, v in before.items()],
                       *[(f"after_{h}m", v) for h, v in after.items()],
                       *[(f"plan_only_{h}m", v) for h, v in plan_only.items()],
                       *[(k, v) for k, v in fields.items() if not isinstance(v, bool)]]:
        if value is not None:
            numbers[key] = value
    if goal is not None:
        numbers |= {"goal_amount": goal.amount, "saved_so_far": goal.saved_so_far}
    return Projection(status=status, months_of_history=len(net_by_month), added_saving=added,
                      current_surplus=current, new_surplus=new, cumulative_before=before,
                      cumulative_after=after, plan_only=plan_only, goal=goal, notes=tuple(notes),
                      numbers=numbers, **fields)
