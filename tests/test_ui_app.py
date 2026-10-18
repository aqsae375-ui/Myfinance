"""End-to-end UI tests with streamlit.testing.v1.AppTest (fake provider only, no network).
App-owned DBs go to a temp dir via FINANCE_APP_DATA, so data/finance.db is never touched."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from src.analytics import month_history  # noqa: E402
from src.analytics.common import prepare  # noqa: E402
from src.ui import format as fmt  # noqa: E402

TIMEOUT = 120
APP = str(Path(__file__).resolve().parent.parent / "app.py")
KPI_RE = re.compile(r'data-kpi="(\w+)" data-tone="(\w+)".*?<div class="value">([^<]+)</div>', re.S)


@pytest.fixture(scope="session")
def app_data(tmp_path_factory):
    """Temp app-data dir with the sample DB built once for the whole session."""
    with pytest.MonkeyPatch.context() as mp:
        d = tmp_path_factory.mktemp("appdata")
        mp.setenv("FINANCE_APP_DATA", str(d))
        mp.setenv("LLM_PROVIDER", "fake")
        from src.ui.data import ensure_sample_db
        path = ensure_sample_db()
        yield path


@pytest.fixture
def app(app_data, monkeypatch):
    monkeypatch.setenv("FINANCE_APP_DATA", str(app_data.parent))
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    at.button(key="load-sample").click().run()
    return at


def assert_clean(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def kpis(at: AppTest) -> dict[str, tuple[str, str]]:
    """metric -> (tone, displayed value), parsed from the KPI cards."""
    out = {}
    for m in at.markdown:
        if 'class="fa-kpi"' in m.value:
            metric, tone, value = KPI_RE.search(m.value).groups()
            out[metric] = (tone, value)
    return out


@pytest.fixture(scope="module")
def history(app_data):
    return {r.month: r for r in month_history(prepare(db_path=app_data))}


def test_starts_with_friendly_empty_state(app_data, monkeypatch):
    monkeypatch.setenv("FINANCE_APP_DATA", str(app_data.parent))
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    assert_clean(at)
    assert any("Upload a statement or load sample data" in m.value for m in at.markdown)
    assert not kpis(at)


def test_sample_data_shows_dashboard_for_latest_month(app, history):
    assert_clean(app)
    assert app.selectbox(key="month").value == max(history)
    cards = kpis(app)
    assert set(cards) == {"total_spent", "total_income", "net_saved", "savings_rate"}
    latest = history[max(history)]
    assert cards["total_spent"][1] == fmt.money(latest.total_spent)
    assert cards["total_income"][1] == fmt.money(latest.total_income)
    assert cards["savings_rate"][1] == fmt.pct(latest.savings_rate)


def test_first_month_kpis_have_no_comparison(app, history):
    first = min(history)
    app.selectbox(key="month").set_value(first).run()
    assert_clean(app)
    cards = kpis(app)
    assert len(cards) == 4
    assert all(tone == "none" for tone, _ in cards.values())
    kpi_html = [m.value for m in app.markdown if 'class="fa-kpi"' in m.value]
    assert all("No earlier month to compare" in h and "fa-pill" not in h for h in kpi_html)


@pytest.mark.parametrize("month", ["2026-07", "2026-08"])
def test_later_month_delta_colors(app, history, month):
    """Spent: up = bad (red), down = good. Income / net / rate: up = good, down = bad."""
    app.selectbox(key="month").set_value(month).run()
    assert_clean(app)
    cards = kpis(app)
    row = history[month]
    expect = {
        "total_spent": "bad" if row.spent.abs_change > 0 else "good",
        "total_income": "good" if row.income.abs_change > 0 else "bad",
        "net_saved": "good" if row.net.abs_change > 0 else "bad",
        "savings_rate": "good" if row.rate.abs_change > 0 else "bad",
    }
    assert {k: v[0] for k, v in cards.items()} == expect
    kpi_html = [m.value for m in app.markdown if 'class="fa-kpi"' in m.value]
    prev_name = fmt.month_name({"2026-07": "2026-06", "2026-08": "2026-07"}[month])
    assert all(f"vs {prev_name}" in h and "fa-pill" in h for h in kpi_html)


def test_spent_and_income_colors_move_in_opposite_directions(app, history):
    """July: spending fell (green); the same direction for income would be red."""
    july = history["2026-07"]
    assert july.spent.abs_change < 0
    app.selectbox(key="month").set_value("2026-07").run()
    assert kpis(app)["total_spent"][0] == "good"


def test_month_selector_changes_the_numbers(app, history):
    app.selectbox(key="month").set_value("2026-05").run()
    may = kpis(app)
    app.selectbox(key="month").set_value("2026-07").run()
    july = kpis(app)
    assert_clean(app)
    assert may["total_spent"][1] == fmt.money(history["2026-05"].total_spent)
    assert july["total_spent"][1] == fmt.money(history["2026-07"].total_spent)
    assert may["total_spent"] != july["total_spent"]


def test_dashboard_sections_render(app):
    text = " ".join(m.value for m in app.markdown)
    for title in ("Where your money went", "This month vs last month", "Spending trend", "What changed",
                  "merchants", "Daily spending", "Month history", "Transactions"):
        assert title in text
    assert 'tr class="selected"' in text            # history table highlights the selected month
    assert 'class="fa-merchant"' in text and 'class="fa-legend"' in text
    assert len(app.get("plotly_chart")) >= 8        # 4 sparklines + donut, comparison, trend, daily


def ask(at: AppTest, question: str) -> None:
    at.text_input(key="ask-input").set_value(question)
    at.button(key="FormSubmitter:ask-form-Send").click().run()


def test_ask_returns_traced_answer(app):
    app.segmented_control(key="nav").set_value("Ask").run()
    assert_clean(app)
    text = " ".join(m.value for m in app.markdown)
    assert "How I got this" in text and not app.expander           # hint only; no trace panel before a question
    ask(app, "How much did I spend on dining in May?")
    assert_clean(app)
    text = " ".join(m.value for m in app.markdown)
    assert "<b>Rs 1,535.37</b>" in text                          # answer bubble, amount in bold
    assert [e.label for e in app.expander] == ["✓ Verified · How I got this"]
    steps = " ".join(m.value for m in app.expander[0].markdown)
    assert "get_category_spending(" in steps and "Result returned by code" in steps and "Number check" in steps
    assert "1,535.37 ✓" in steps
    trace = app.session_state["chat"][-1]["trace"]
    assert trace.verified and not trace.fallback_used and trace.provider == "fake"


def test_ask_suggestions_come_from_the_data_and_refusal(app, history):
    app.segmented_control(key="nav").set_value("Ask").run()
    labels = [b.label for b in app.button if (b.key or "").startswith("suggest-")]
    latest, prev = fmt.month_name(max(history)), fmt.month_name(sorted(history)[-2])
    assert labels and any(latest in q for q in labels) and f"Compare {prev} with {latest}" in labels
    app.button(key="suggest-1").click().run()
    assert_clean(app)
    assert len(app.session_state["chat"]) == 1
    ask(app, "Which stocks should I buy?")
    assert app.expander[-1].label.startswith("Refused")


def _stat(at: AppTest, label: str) -> str:
    for m in at.markdown:
        if f'<div class="label plain">{label}</div>' in m.value and '<div class="value">' in m.value:
            return re.search(r'<div class="value">([^<]+)</div>', m.value).group(1)
    raise AssertionError(f"no stat {label!r}")


def set_goal(at: AppTest, amount: float, deadline: str) -> None:
    import datetime as dt
    at.segmented_control(key="nav").set_value("Your Plan").run()
    at.number_input(key="goal_amount").set_value(amount)
    at.date_input(key="goal_date").set_value(dt.date.fromisoformat(deadline))
    at.button(key="goal-save").click().run()


def test_accepting_an_opportunity_changes_the_projection(app):
    set_goal(app, 10000, "2027-03-31")
    assert_clean(app)
    assert app.session_state["accepted"] == {}  # nothing toggled yet: every suggestion accepted by default
    all_on = _stat(app, "Monthly surplus")
    app.checkbox[0].uncheck().run()
    assert_clean(app)
    assert _stat(app, "Monthly surplus") != all_on
    app.checkbox[0].check().run()
    assert _stat(app, "Monthly surplus") == all_on
    assert "vs Rs 1,251 now" in " ".join(m.value for m in app.markdown)
    app.button(key="goal-clear").click().run()


def test_goal_by_date_shows_months_to_goal(app):
    set_goal(app, 10000, "2027-03-31")
    assert_clean(app)
    assert _stat(app, "Months to goal") == "6"
    assert "▼ 2 months" in " ".join(m.value for m in app.markdown)
    app.button(key="goal-clear").click().run()
    assert any("Set your goal" in m.value for m in app.markdown)


def test_fix_this_opens_your_plan(app):
    fix = [b for b in app.button if (b.key or "").startswith("fix-")]
    assert fix
    fix[0].click().run()
    assert_clean(app)
    assert app.session_state["nav"] == "Your Plan"
    assert any("From your dashboard" in m.value for m in app.markdown)


@pytest.fixture(scope="module")
def pk_db(tmp_path_factory):
    """The synthetic Pakistani statement (12 months) ingested like an upload (rules + offline fake)."""
    from tests.pk_fixture import FIXTURE
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("FINANCE_APP_DATA", str(tmp_path_factory.mktemp("pkdata")))
        from src.ui.data import ingest_upload
        return ingest_upload(FIXTURE.read_bytes(), "pk_statement.csv", "fake").db_path


def test_twelve_month_statement_with_cash_note(pk_db, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.session_state["db_path"] = str(pk_db)
    at.run()
    assert_clean(at)
    assert len(at.selectbox(key="month").options) == 12
    assert at.selectbox(key="month").value == "2025-12"
    text = " ".join(m.value for m in at.markdown)
    assert "Cash spending is not tracked after withdrawal" in text
    assert "left your account as cash from Jan to Dec 2025" in text
    assert text.count("<tr class=") == 12                       # history table: one row per month
    spark = at.get("plotly_chart")[0].proto.spec
    assert spark.count('"2025-') >= 12                           # sparkline spans all 12 months
