"""Reference-design restyle: no sample content baked into the UI, a goal-driven Your Plan, data-driven Ask
suggestions, and edge cases (other statements, a single month, gaps, zero income). Fake provider only."""
from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from src import analytics  # noqa: E402
from src.analytics.common import prepare  # noqa: E402
from src.ui import format as fmt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "app.py")
TIMEOUT = 120
SAMPLE_STRINGS = ("15,000", "2027", "FITPRO", "3,394", "601.04", "596.10", "vs June", "Save Rs")


def test_no_sample_strings_in_the_ui_code():
    files = [ROOT / "app.py", *sorted((ROOT / "src" / "ui").glob("*.py"))]
    for path in files:
        text = path.read_text(encoding="utf-8")
        for s in SAMPLE_STRINGS:
            assert s not in text, f"{path.name} contains sample text {s!r}"


# ---- helpers ------------------------------------------------------------------------------------

def assert_clean(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def text_of(at: AppTest) -> str:
    return " ".join(m.value for m in at.markdown)


def _clear_goals(db: Path) -> None:
    conn = sqlite3.connect(db)
    try:
        conn.execute("DELETE FROM goals")
        conn.commit()
    finally:
        conn.close()


def open_app(db: Path, monkeypatch, view: str = "Dashboard") -> AppTest:
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.session_state["db_path"] = str(db)
    at.run()
    if view != "Dashboard":
        at.segmented_control(key="nav").set_value(view).run()
    return at


def upload(csv_bytes: bytes, name: str, tmp_path_factory) -> Path:
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("FINANCE_APP_DATA", str(tmp_path_factory.mktemp("refdata")))
        from src.ui.data import ingest_upload
        return ingest_upload(csv_bytes, name, "fake").db_path


@pytest.fixture(scope="module")
def sample_db(tmp_path_factory):
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("FINANCE_APP_DATA", str(tmp_path_factory.mktemp("sample")))
        from src.ui.data import ensure_sample_db
        return ensure_sample_db()


@pytest.fixture(scope="module")
def pk_db(tmp_path_factory):
    from tests.pk_fixture import FIXTURE
    return upload(FIXTURE.read_bytes(), "pk_statement.csv", tmp_path_factory)


@pytest.fixture(scope="module")
def one_month_db(tmp_path_factory):
    csv = ("Date,Description,Amount\n2026-05-01,PAYROLL DEPOSIT ACME,3000.00\n2026-05-03,NETFLIX.COM,-15.49\n"
           "2026-05-09,UBER TRIP,-22.00\n2026-05-20,KROGER #12,-80.25\n").encode()
    return upload(csv, "one_month.csv", tmp_path_factory)


def set_goal(at: AppTest, amount: float, deadline: dt.date | None = None) -> None:
    at.segmented_control(key="nav").set_value("Your Plan").run()
    at.number_input(key="goal_amount").set_value(amount)
    if deadline is not None:
        at.date_input(key="goal_date").set_value(deadline)
    at.button(key="goal-save").click().run()


# ---- (a) no goal: only the Set your goal form ----------------------------------------------------

def test_a_sample_without_goal_shows_only_the_goal_form(sample_db, monkeypatch):
    _clear_goals(sample_db)
    at = open_app(sample_db, monkeypatch, "Your Plan")
    assert_clean(at)
    text = text_of(at)
    assert "Set your goal" in text
    assert text.count("Set a goal to see this") >= 4          # 3 result cards + opportunities + chart
    assert not at.checkbox                                     # no ranked opportunities yet
    assert not at.get("plotly_chart")                          # no current-vs-plan chart
    assert at.number_input(key="goal_amount").value is None    # no made-up default amount


# ---- (b) with a goal: the goal-based cards -------------------------------------------------------

def test_b_with_a_goal_shows_goal_cards_and_can_be_edited_and_removed(sample_db, monkeypatch):
    _clear_goals(sample_db)
    at = open_app(sample_db, monkeypatch)
    deadline = dt.date.today().replace(year=dt.date.today().year + 2)
    set_goal(at, 20000, deadline)
    assert_clean(at)
    text = text_of(at)
    assert f"Save {fmt.money(20000)} by {deadline:%b} {deadline.day}, {deadline.year}" in text
    for label in ("Monthly surplus", "Months to goal", "12-month savings", "Ways to save, ranked",
                  "Current path vs your plan"):
        assert label in text
    assert "Set a goal to see this" not in text
    assert len(at.checkbox) >= 3 and at.get("plotly_chart")
    plan = analytics_plan(sample_db)
    assert f"vs {fmt.money(plan.projection.current_surplus)} now" in text   # from the advisor, not the PDF

    at.button(key="goal-edit").click().run()
    assert at.number_input(key="goal_amount").value == 20000
    at.number_input(key="goal_amount").set_value(25000)
    at.button(key="goal-save").click().run()
    assert f"Save {fmt.money(25000)} by" in text_of(at)
    at.button(key="goal-clear").click().run()
    assert_clean(at)
    assert "Set your goal" in text_of(at)


def analytics_plan(db: Path):
    from src.advisor import build_plan
    return build_plan(prepare(db_path=db), goal=None, provider=None)


def test_goal_validation_and_honest_unrealistic_message(sample_db, monkeypatch):
    from src.ui.plan import validate_goal
    today = dt.date(2026, 10, 10)
    assert validate_goal("target_by_date", None, today, 0, today) == "Enter a target amount above zero."
    assert validate_goal("target_by_date", 0, today, 0, today) == "Enter a target amount above zero."
    assert validate_goal("target_by_date", 500, today, 0, today) == "The deadline must be in the future."
    assert validate_goal("target_by_date", 500, None, 0, today) == "Pick a deadline."
    assert validate_goal("monthly_savings", 500, None, None, today) is None

    _clear_goals(sample_db)
    at = open_app(sample_db, monkeypatch)
    set_goal(at, 0)
    assert any("above zero" in e.value for e in at.error)
    # A huge target due next month: saved, but called out plainly.
    soon = dt.date.today() + dt.timedelta(days=40)
    at.number_input(key="goal_amount").set_value(5_000_000)
    at.date_input(key="goal_date").set_value(soon)
    at.button(key="goal-save").click().run()
    assert not at.exception
    assert 'data-testid="goal-honest"' in text_of(at)
    at.button(key="goal-clear").click().run()


# ---- (c) another statement shows its own months and numbers --------------------------------------

def test_c_pk_statement_shows_its_own_months_and_numbers(pk_db, monkeypatch):
    at = open_app(pk_db, monkeypatch)
    assert_clean(at)
    df = prepare(db_path=pk_db)
    months = analytics.all_months(df)
    assert list(at.selectbox(key="month").options) == [fmt.month_label(m) for m in months]
    assert at.selectbox(key="month").value == months[-1]
    latest = analytics.month_history(df, previous="data")[-1]
    text = text_of(at)
    assert f'<div class="value">{fmt.money(latest.total_spent)}</div>' in text
    assert f"vs {fmt.month_name(months[-2])}" in text
    assert f"{fmt.month_label(months[-1])}, by category" in text
    top = analytics.top_merchants(months[-1], 5, df)[0]
    assert top.merchant.title() in text

    at.segmented_control(key="nav").set_value("Ask").run()
    assert_clean(at)
    labels = [b.label for b in at.button if (b.key or "").startswith("suggest-")]
    assert labels and all(any(fmt.month_name(m) in q for m in months) or "subscription" in q for q in labels)


# ---- (d) a single month of data -------------------------------------------------------------------

@pytest.mark.parametrize("view", ["Dashboard", "Ask", "Your Plan"])
def test_d_single_month_does_not_crash(one_month_db, monkeypatch, view):
    at = open_app(one_month_db, monkeypatch, view)
    assert_clean(at)
    text = text_of(at)
    if view == "Dashboard":
        assert "Not enough history for a trend yet" in text
        assert "No earlier month to compare" in text
    if view == "Ask":
        assert not any("Compare" in b.label for b in at.button if (b.key or "").startswith("suggest-"))


def test_d_single_month_goal_says_it_cannot_project(one_month_db, monkeypatch):
    _clear_goals(one_month_db)
    at = open_app(one_month_db, monkeypatch)
    set_goal(at, 1000, dt.date.today() + dt.timedelta(days=400))
    assert_clean(at)
    assert "only 1 month of history" in text_of(at)
    at.button(key="goal-clear").click().run()


# ---- gaps and zero income -------------------------------------------------------------------------

def test_gap_compares_with_the_previous_month_in_the_data(make_txns):
    df = make_txns([("2026-01-05", "A", -10.0, "Other"), ("2026-03-05", "A", -20.0, "Other")])
    assert analytics.month_over_month("2026-03", df).previous_month is None          # calendar (default)
    mom = analytics.month_over_month("2026-03", df, previous="data")
    assert mom.previous_month == "2026-01" and mom.total_spent.abs_change == 10.0
    rows = analytics.month_history(df, previous="data")
    assert rows[1].spent.previous == 10.0


def test_gap_and_zero_income_statement_renders(tmp_path_factory, monkeypatch):
    csv = ("Date,Description,Amount\n2026-01-05,KROGER #1,-40.00\n2026-01-09,UBER TRIP,-12.00\n"
           "2026-03-05,KROGER #1,-55.00\n2026-03-12,NETFLIX.COM,-15.49\n").encode()
    db = upload(csv, "gap.csv", tmp_path_factory)
    at = open_app(db, monkeypatch)
    assert_clean(at)
    text = text_of(at)
    assert "vs January" in text                     # March compares with the previous month that has data
    assert f'<div class="value">{fmt.NA}</div>' in text   # savings rate with zero income: no fake value


# ---- Ask: collapsed per-answer traces; Transactions: data-driven filters, HTML ledger, Fix a category ----

def _ask(at: AppTest, question: str) -> None:
    at.text_input(key="ask-input").set_value(question)
    at.button(key="FormSubmitter:ask-form-Send").click().run()


def test_ask_has_no_trace_panel_and_one_collapsed_expander_per_answer(sample_db, monkeypatch):
    at = open_app(sample_db, monkeypatch, "Ask")
    assert_clean(at)
    assert "The exact steps behind the last answer" not in text_of(at)      # no always-visible panel
    assert not at.expander
    q1 = next(b for b in at.button if (b.key or "") == "suggest-0").label
    at.button(key="suggest-0").click().run()
    _ask(at, "Compare dining out in July and August")
    assert_clean(at)
    assert len(at.expander) == len(at.session_state["chat"]) == 2
    assert all(e.label.endswith("How I got this") and not e.proto.expanded for e in at.expander)
    # each answer keeps its own trace: the first expander shows the first question's tool call
    first = " ".join(m.value for m in at.expander[0].markdown)
    assert at.session_state["chat"][0]["question"] == q1
    assert at.session_state["chat"][0]["trace"].tool_calls[-1].name in first


def _filter(at: AppTest, month: str):
    return next(w for w in at.get("button_group") if f"txn-filter-{month}" in w.proto.id)


def test_filters_come_from_categories_in_the_data(pk_db, monkeypatch):
    df = prepare(db_path=pk_db)
    month = next(m for m in analytics.all_months(df)       # a month with both PK-only categories
                 if {"Cash Withdrawal", "Bank Fees"} <= {r.category for r in analytics.month_transactions(m, txns=df)})
    at = open_app(pk_db, monkeypatch)
    at.selectbox(key="month").set_value(month).run()
    assert_clean(at)
    rows = analytics.month_transactions(month, txns=df)
    present = {r.category for r in rows}
    options = list(_filter(at, month).options)
    assert options[0] == "All"
    cats = [o for o in options[1:] if not o.startswith("Needs review")]
    assert set(cats) == present and "Rent" not in cats and "Shopping" not in cats
    text = text_of(at)
    assert 'class="fa-ledger"' in text
    assert text.count("<tr data-txn=") == min(25, len(rows))
    assert f"{rows[0].date:%b} {rows[0].date.day}" in text                   # newest first, "Jul 25" format


def test_fix_a_category_saves_a_manual_override(tmp_path_factory, monkeypatch):
    from tests.pk_fixture import FIXTURE
    db = upload(FIXTURE.read_bytes(), "pk_fix.csv", tmp_path_factory)
    at = open_app(db, monkeypatch)
    month = at.selectbox(key="month").value
    row = analytics.month_transactions(month, txns=prepare(db_path=db))[0]
    new = "Shopping" if row.category != "Shopping" else "Health"
    at.selectbox(key="fix-txn").set_value(row.txn_id).run()
    at.selectbox(key=f"fix-cat-{row.txn_id}").set_value(new).run()
    at.button(key="fix-save").click().run()
    assert_clean(at)
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT category, category_source FROM transactions WHERE id = ?",
                        (row.txn_id,)).fetchone() == (new, "manual")
    conn.close()
    cell = text_of(at).split(f'<tr data-txn="{row.txn_id}">')[1].split("</tr>")[0]
    assert new in cell and ">Manual<" in cell


def test_emptied_filter_switches_back_to_all(tmp_path_factory, monkeypatch):
    """Filtering on a category, then re-labelling its only row: the filter goes back to All."""
    csv = ("Date,Description,Amount\n2026-06-01,PAYROLL DEPOSIT ACME,3000.00\n2026-06-03,NETFLIX.COM,-15.49\n"
           "2026-06-09,UBER TRIP,-22.00\n").encode()
    db = upload(csv, "filter.csv", tmp_path_factory)
    at = open_app(db, monkeypatch)
    _filter(at, "2026-06").set_value("Subscriptions").run()
    assert_clean(at)
    assert text_of(at).count("<tr data-txn=") == 1
    txn = at.selectbox(key="fix-txn").value
    at.selectbox(key=f"fix-cat-{txn}").set_value("Entertainment").run()
    at.button(key="fix-save").click().run()
    assert_clean(at)
    assert "Subscriptions" not in _filter(at, "2026-06").options
    assert _filter(at, "2026-06").value == "All"
    assert "No transactions in this category" not in text_of(at)
    assert text_of(at).count("<tr data-txn=") == 3
