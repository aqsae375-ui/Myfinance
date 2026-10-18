"""Savings goals: a target monthly saving, or a target amount by a date. Stored in the `goals`
table (schema in src/db.py); the newest row is the active goal."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date

MONTHLY = "monthly_savings"
BY_DATE = "target_by_date"
KINDS = (MONTHLY, BY_DATE)


@dataclass(frozen=True)
class Goal:
    kind: str                       # monthly_savings | target_by_date
    amount: float                   # Rs per month (monthly_savings) or the total target (target_by_date)
    name: str = "Savings goal"
    target_date: date | None = None  # required for target_by_date, must be None otherwise
    saved_so_far: float = 0.0       # already saved toward a target_by_date goal
    id: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"goal kind must be one of {KINDS}, got {self.kind!r}")
        if not self.amount > 0:
            raise ValueError("goal amount must be positive")
        if self.saved_so_far < 0:
            raise ValueError("saved_so_far cannot be negative")
        if (self.kind == BY_DATE) != (self.target_date is not None):
            raise ValueError("target_date is required for target_by_date goals and only for them")
        if self.kind == MONTHLY and self.saved_so_far:
            raise ValueError("saved_so_far only applies to target_by_date goals")

    @classmethod
    def monthly(cls, amount: float, name: str = "Monthly savings") -> Goal:
        return cls(MONTHLY, round(float(amount), 2), name)

    @classmethod
    def by_date(cls, amount: float, target_date: date | str, saved_so_far: float = 0.0,
                name: str = "Savings goal") -> Goal:
        if isinstance(target_date, str):
            target_date = date.fromisoformat(target_date)
        return cls(BY_DATE, round(float(amount), 2), name, target_date, round(float(saved_so_far), 2))

    def describe(self) -> str:
        if self.kind == MONTHLY:
            return f"{self.name}: save Rs {self.amount:,.2f} every month"
        return (f"{self.name}: Rs {self.amount:,.2f} by {self.target_date.isoformat()} "
                f"(saved so far Rs {self.saved_so_far:,.2f})")


def save_goal(conn: sqlite3.Connection, goal: Goal) -> Goal:
    """Store `goal` as the active goal (older goals are kept as history). Returns it with its id."""
    with conn:
        cur = conn.execute(
            "INSERT INTO goals (kind, name, amount, target_date, saved_so_far) VALUES (?, ?, ?, ?, ?)",
            (goal.kind, goal.name, goal.amount,
             goal.target_date.isoformat() if goal.target_date else None, goal.saved_so_far))
    return Goal(goal.kind, goal.amount, goal.name, goal.target_date, goal.saved_so_far, cur.lastrowid)


def load_goal(conn: sqlite3.Connection) -> Goal | None:
    """The active (newest) goal, or None if no goal has been set."""
    row = conn.execute("SELECT id, kind, name, amount, target_date, saved_so_far FROM goals "
                       "ORDER BY id DESC LIMIT 1").fetchone()
    if row is None:
        return None
    gid, kind, name, amount, target, saved = row
    return Goal(kind, amount, name, date.fromisoformat(target) if target else None, saved, gid)


def clear_goals(conn: sqlite3.Connection) -> None:
    with conn:
        conn.execute("DELETE FROM goals")
