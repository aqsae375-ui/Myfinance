"""Unit tests for Stage 6 helpers: formatting, delta colors, KPI HTML, the new analytics views,
advisor.reproject, and the upload/override data actions (fake provider only)."""
from __future__ import annotations

import sqlite3
from datetime import date

import pytest

from src.advisor import Goal, build_plan, reproject
from src.advisor.projector import cumulative_path
from src.analytics import daily_spending, month_history, month_transactions, savings_rate, top_merchants
from src.analytics.monthly import change
from src.ui import format as fmt
from src.ui.components import delta_tone, kpi_html


# ---- format ----

@pytest.mark.parametrize("value, kwargs, expected", [
    (1234.4, {}, "Rs 1,234"), (-1234.4, {}, "-Rs 1,234"), (44.99, {"cents": True}, "Rs 44.99"),
    (-12.5, {"cents": True}, "-Rs 12.50"), (587.0, {"signed": True}, "+Rs 587"), (0.0, {"signed": True}, "Rs 0"),
    (None, {}, "—"), (-0.004, {"cents": True}, "Rs 0.00"),
])
def test_money(value, kwargs, expected):
    assert fmt.money(value, **kwargs) == expected


def test_pct_points_and_months():
    assert fmt.pct(28.37) == "28.4%"
    assert fmt.pct(-5, signed=True) == "-5.0%"
    assert fmt.pct(5, signed=True) == "+5.0%"
    assert fmt.pct(None) == "—"
    assert fmt.points(3.21) == "+3.2 pts" and fmt.points(-1) == "-1.0 pts"
    assert fmt.month_label("2026-08") == "August 2026" and fmt.month_label("2026-08", short=True) == "Aug 2026"
    assert fmt.month_name("2026-07") == "July"


# ---- delta colors ----

@pytest.mark.parametrize("metric, delta, tone", [
    ("total_spent", 100, "bad"), ("total_spent", -100, "good"),
    ("total_income", 100, "good"), ("total_income", -100, "bad"),
    ("net_saved", 5, "good"), ("savings_rate", -2.5, "bad"),
    ("total_spent", 0, "flat"), ("total_income", None, "flat"),
])
def test_delta_tone(metric, delta, tone):
    assert delta_tone(metric, delta) == tone


def test_kpi_html_first_month_and_zero_previous():
    first = kpi_html("total_spent", "Total spent", "Rs 10", change(10.0, None), None)
    assert 'data-tone="none"' in first and "No earlier month to compare" in first and "fa-pill" not in first
    # previous = 0: no % (never divide by zero), the Rs change is shown instead
    from_zero = kpi_html("total_income", "Income", "Rs 50", change(50.0, 0.0), "2026-03")
    assert 'data-tone="good"' in from_zero and "▲ Rs 50" in from_zero and "vs March" in from_zero
    up = kpi_html("total_spent", "Total spent", "Rs 120", change(120.0, 100.0), "2026-03")
    assert "▲ 20.0%" in up and 'fa-pill bad' in up
    rate = kpi_html("savings_rate", "Savings rate", "30%", change(30.0, 25.0), "2026-03", is_rate=True)
    assert "▲ 5.0 pts" in rate


# ---- analytics views ----

def test_savings_rate_never_divides_by_zero():
    assert savings_rate(0.0, -50.0) is None
    assert savings_rate(200.0, 50.0) == 25.0


def test_month_history_matches_monthly_summary(synthetic_df):
    from src.analytics import monthly_summary
    rows = month_history(synthetic_df)
    assert [r.month for r in rows] == ["2026-03", "2026-04", "2026-05", "2026-06", "2026-07", "2026-08"]
    first, may = rows[0], rows[2]
    assert first.spent.previous is None and first.rate.abs_change is None
    s = monthly_summary("2026-05", synthetic_df)
    assert (may.total_spent, may.total_income, may.net_saved) == (s.total_spent, s.total_income, s.net_saved)
    assert may.savings_rate == round(s.net_saved / s.total_income * 100, 2)
    assert may.rate.abs_change == round(may.savings_rate - rows[1].savings_rate, 2)
    assert may.rate.pct_change is None


def test_month_history_income_free_month(make_txns):
    df = make_txns([("2026-01-05", "SHOP", -10.0, "Shopping"), ("2026-02-05", "SHOP", -20.0, "Shopping")])
    rows = month_history(df)
    assert rows[1].savings_rate is None and rows[1].rate.abs_change is None
    assert rows[1].spent.abs_change == 10.0


def test_top_merchants_and_daily(synthetic_df):
    top = top_merchants("2026-07", 5, synthetic_df)
    assert len(top) == 5 and [t.amount for t in top] == sorted((t.amount for t in top), reverse=True)
    daily = daily_spending("2026-07", synthetic_df)
    assert len(daily.days) == 31 and daily.days[0].date == date(2026, 7, 1)
    assert daily.peak.amount == max(d.amount for d in daily.days)
    assert round(sum(d.amount for d in daily.days), 2) == month_history(synthetic_df)[4].total_spent


def test_daily_spending_with_no_debits(make_txns):
    df = make_txns([("2026-02-03", "SALARY", 100.0, "Income")])
    d = daily_spending("2026-02", df)
    assert len(d.days) == 28 and d.peak is None and all(x.amount == 0 for x in d.days)


def test_month_transactions_sources(synthetic_df):
    rows = month_transactions("2026-04", txns=synthetic_df)
    assert rows and rows[0].date >= rows[-1].date
    assert {r.source for r in rows} <= {"rule", "llm", "manual"}
    dining = month_transactions("2026-04", "Dining Out", synthetic_df)
    assert dining and all(r.category == "Dining Out" for r in dining)


# ---- advisor ----

def test_reproject_matches_build_plan(synthetic_df):
    goal = Goal.by_date(10000, "2027-03-31")
    base = build_plan(synthetic_df)
    ids = [i.opportunity.id for i in base.items]
    for accepted in ([], ids[:1], ids[1:3], ids):
        fast = reproject(base, accepted, goal)
        full = build_plan(synthetic_df, goal, accepted=accepted)
        assert fast.projection == full.projection
        assert [i.accepted for i in fast.items] == [i.accepted for i in full.items]
    assert reproject(base, ids).goal is None  # goal kept when not given
    with pytest.raises(ValueError):
        reproject(base, ["nope"])


def test_cumulative_path():
    assert cumulative_path(100.5, 3) == [0.0, 100.5, 201.0, 301.5]
    assert len(cumulative_path(-10)) == 13


# ---- data actions ----

def test_upload_pipeline_and_override(statement_csv, tmp_path, monkeypatch):
    monkeypatch.setenv("FINANCE_APP_DATA", str(tmp_path))
    from src.ui import data
    steps, needs = [], []
    content = statement_csv.read_bytes()
    result = data.ingest_upload(content, "statement.csv", "fake", step=lambda p, m: steps.append(p),
                                on_needs_llm=needs.append)
    # Generic merchant words (CAFE, POWER, FOODS, ...) are only hints since Stage 2d: those rows go to the LLM.
    assert result.rows_total == result.rows_added == 192 and result.by_rule + result.by_llm == 191
    assert steps[-1] == 1.0 and needs == [result.needs_llm]
    key = data.db_key(result.db_path)
    again = data.ingest_upload(content, "statement.csv", "fake")
    assert again.rows_added == 0 and data.db_key(result.db_path) == key   # idempotent re-upload

    conn = sqlite3.connect(result.db_path)
    txn_id = conn.execute("SELECT id FROM transactions WHERE category = 'Shopping' LIMIT 1").fetchone()[0]
    conn.close()
    data.set_category(result.db_path, txn_id, "Groceries")
    assert data.db_key(result.db_path) != key
    conn = sqlite3.connect(result.db_path)
    assert conn.execute("SELECT category, category_source FROM transactions WHERE id = ?",
                        (txn_id,)).fetchone() == ("Groceries", "manual")
    conn.close()


@pytest.mark.parametrize("content, message", [
    (b"", "empty"),
    (b"\xff\xfe\x00bad", "UTF-8"),
    (b"Date,Description\n2026-01-01,Coffee\n", "Missing column"),
    (b"Date,Description,Amount\n2026-01-01,Coffee,-3.50\n01-31-2026,Tea,-2\n", "line 3"),
])
def test_upload_errors_are_friendly(content, message):
    from src.ui.data import UploadError, parse_upload
    with pytest.raises(UploadError, match=message):
        parse_upload(content)


def test_rows_needing_llm():
    from src.categorizer import rows_needing_llm
    from src.db import connect, ingest
    from src.parser import normalize
    import pandas as pd
    conn = connect(":memory:")
    raw = pd.DataFrame({"date": ["2026-01-01", "2026-01-02"], "description": ["NETFLIX.COM", "ZQXW 9931 LLC"],
                        "amount": ["-15.99", "-40"]})
    ingest(conn, normalize(raw))
    assert rows_needing_llm(conn) == 1


def test_css_never_hides_the_sidebar_toggle(monkeypatch):
    """The "open sidebar" button (stExpandSidebarButton) lives inside stToolbar: hiding the toolbar
    made a collapsed sidebar impossible to reopen."""
    from src.ui import components
    captured = []
    monkeypatch.setattr(components.st, "markdown", lambda body, **kw: captured.append(body))
    components.inject_css("light")
    hidden = captured[0].split("{ display:none")[0]
    assert '[data-testid="stToolbar"]' not in hidden and "stExpandSidebarButton" not in hidden
    assert "stMainMenu" in hidden and "stAppDeployButton" in hidden
