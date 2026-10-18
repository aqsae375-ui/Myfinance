from datetime import date

import pytest

from src.advisor.goals import BY_DATE, MONTHLY, Goal, clear_goals, load_goal, save_goal
from src.db import connect


def test_constructors_and_validation():
    g = Goal.by_date(10000, "2027-03-31", saved_so_far=500, name="Emergency fund")
    assert g.kind == BY_DATE and g.target_date == date(2027, 3, 31) and g.saved_so_far == 500
    assert "Rs 10,000.00 by 2027-03-31" in g.describe()
    assert Goal.monthly(1500).kind == MONTHLY and "Rs 1,500.00 every month" in Goal.monthly(1500).describe()
    with pytest.raises(ValueError):
        Goal.monthly(0)
    with pytest.raises(ValueError):
        Goal("yearly", 100)
    with pytest.raises(ValueError):
        Goal(BY_DATE, 100)                              # missing date
    with pytest.raises(ValueError):
        Goal(MONTHLY, 100, target_date=date(2027, 1, 1))  # date on a monthly goal
    with pytest.raises(ValueError):
        Goal.by_date(100, "2027-01-01", saved_so_far=-1)


def test_db_round_trip_newest_goal_is_active():
    conn = connect(":memory:")
    assert load_goal(conn) is None
    first = save_goal(conn, Goal.monthly(1200))
    second = save_goal(conn, Goal.by_date(10000, "2027-03-31", saved_so_far=250, name="Emergency fund"))
    assert first.id is not None and second.id > first.id
    loaded = load_goal(conn)
    assert loaded == second and loaded.target_date == date(2027, 3, 31) and loaded.saved_so_far == 250
    clear_goals(conn)
    assert load_goal(conn) is None
    conn.close()


def test_db_rejects_invalid_rows():
    import sqlite3
    conn = connect(":memory:")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO goals (kind, name, amount) VALUES ('monthly_savings', 'x', -5)")
    conn.close()
