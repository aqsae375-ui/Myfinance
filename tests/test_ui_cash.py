"""Cash wallet in the app (AppTest, fake provider only): the card, adding / quick-adding / editing / deleting
an entry, and the donut, KPI and transactions table following. Each test that writes gets its own DB."""
from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from src import analytics  # noqa: E402
from src.analytics.cash import UNACCOUNTED  # noqa: E402
from src.analytics.common import prepare  # noqa: E402
from src.ui import format as fmt  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = 120
STAT_RE = re.compile(r'data-cash="(\w+)"><div class="l">[^<]*</div><div class="v[^"]*">([^<]+)</div>')
KPI_RE = re.compile(r'data-kpi="total_spent".*?<div class="value">([^<]+)</div>', re.S)


def assert_clean(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def text_of(at: AppTest) -> str:
    return " ".join(m.value for m in at.markdown)


def stats(at: AppTest) -> dict[str, str]:
    return dict(STAT_RE.findall(text_of(at)))


def spent_kpi(at: AppTest) -> str:
    return KPI_RE.search(text_of(at)).group(1)


def donut(at: AppTest) -> dict[str, float]:
    chart = next(c for c in at.get("plotly_chart") if c.proto.id.endswith("donut") or "donut" in c.proto.id)
    trace = json.loads(chart.proto.spec)["data"][0]
    return dict(zip(trace["labels"], trace["values"]))


def open_app(db: Path, monkeypatch) -> AppTest:
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.session_state["db_path"] = str(db)
    return at.run()


@pytest.fixture
def pk_db(tmp_path, monkeypatch):
    from src.ui.data import ingest_upload
    from tests.pk_fixture import FIXTURE
    monkeypatch.setenv("FINANCE_APP_DATA", str(tmp_path))
    return ingest_upload(FIXTURE.read_bytes(), "pk_statement.csv", "fake").db_path


def prepare_entries_ids(db: Path) -> list[int]:
    from src.analytics.cash import load_cash_entries
    return sorted(load_cash_entries(db)["id"].astype(int))


def cash_month(db: Path) -> tuple[str, float]:
    """The latest month with a cash withdrawal, and how much was withdrawn."""
    df = prepare(db_path=db)
    month = max(m for m in analytics.all_months(df) if analytics.cash_summary(m, df).withdrawn > 0)
    return month, analytics.cash_summary(month, df).withdrawn


def test_card_hidden_when_the_data_has_no_withdrawals(tmp_path, monkeypatch):
    monkeypatch.setenv("FINANCE_APP_DATA", str(tmp_path))
    from src.ui.data import ensure_sample_db
    db = ensure_sample_db()
    assert not analytics.has_cash_withdrawals(prepare(db_path=db))
    at = open_app(db, monkeypatch)
    assert_clean(at)
    assert "Cash wallet" not in text_of(at) and not stats(at)
    assert not [w for w in at.text_input if (w.key or "").startswith("cash-")]


def test_card_shows_withdrawn_logged_unaccounted(pk_db, monkeypatch):
    month, withdrawn = cash_month(pk_db)
    at = open_app(pk_db, monkeypatch)
    at.selectbox(key="month").set_value(month).run()
    assert_clean(at)
    s = stats(at)
    assert (s["withdrawn"], s["logged"], s["unaccounted"]) == (fmt.money(withdrawn), fmt.money(0), fmt.money(withdrawn))
    assert "0% accounted for" in text_of(at) and "Wallet balance" not in str(s)
    assert "Unaccounted cash counts as spending until you log it." in text_of(at)   # in the (?) tooltip
    assert 'data-cash="empty">No cash logged yet. Try: chai 150<' in text_of(at)
    assert donut(at)[UNACCOUNTED] == withdrawn
    assert "Cash Withdrawal" not in donut(at)
    assert "Cash spending is not tracked after withdrawal" in text_of(at)


def test_add_entry_updates_card_donut_and_table_and_keeps_spent(pk_db, monkeypatch):
    month, withdrawn = cash_month(pk_db)
    logged = round(withdrawn / 4, 2)
    at = open_app(pk_db, monkeypatch)
    at.selectbox(key="month").set_value(month).run()
    spent_before, donut_before = spent_kpi(at), donut(at)

    assert at.date_input(key="cash-date").value == prepare(db_path=pk_db)["date"].max().date()  # latest date
    at.number_input(key="cash-amount").set_value(logged)
    at.selectbox(key="cash-cat").set_value("Groceries")
    at.date_input(key="cash-date").set_value(dt.date.fromisoformat(f"{month}-15"))
    at.text_input(key="cash-note").set_value("vegetables")
    at.button(key="cash-add").click().run()
    assert_clean(at)
    assert at.number_input(key="cash-amount").value is None        # the form is cleared

    s = stats(at)
    assert (s["logged"], s["unaccounted"]) == (fmt.money(logged), fmt.money(withdrawn - logged))
    assert "25% accounted for" in text_of(at)
    assert "No cash logged yet" not in text_of(at) and "Vegetables" in text_of(at).title()
    d = donut(at)
    assert d[UNACCOUNTED] == round(withdrawn - logged, 2)
    assert d["Groceries"] == round(donut_before.get("Groceries", 0.0) + logged, 2)
    assert spent_kpi(at) == spent_before                          # logged <= withdrawn: Spent unchanged
    text = text_of(at)
    assert '<span class="fa-src cash">Cash</span>' in text and "Cash Vegetables" in text
    assert "Groceries" in list(next(w for w in at.get("button_group")
                                    if f"txn-filter-{month}" in w.proto.id).options)


def test_quick_line_proposes_with_rules_then_confirm_edit_and_delete(pk_db, monkeypatch):
    month, withdrawn = cash_month(pk_db)
    at = open_app(pk_db, monkeypatch)
    at.selectbox(key="month").set_value(month).run()
    spent_before = spent_kpi(at)

    at.date_input(key="cash-date").set_value(dt.date.fromisoformat(f"{month}-10"))  # "More options" date
    at.text_input(key="cash-quick").set_value("chai 150")
    at.button(key="cash-quick-go").click().run()
    assert_clean(at)
    assert at.selectbox(key="cash-quick-cat").value == "Dining Out"
    assert f'data-cash="proposal">Dining Out - {fmt.money(150)}<' in text_of(at)
    at.button(key="cash-quick-save").click().run()
    assert_clean(at)
    assert stats(at)["logged"] == fmt.money(150)
    assert at.text_input(key="cash-quick").value == "" and 'data-cash="proposal"' not in text_of(at)
    assert donut(at)["Dining Out"] >= 150

    # no rule matches: the user must pick
    at.text_input(key="cash-quick").set_value("gift 300")
    at.button(key="cash-quick-go").click().run()
    assert at.selectbox(key="cash-quick-cat").value is None
    assert f"Pick a category for {fmt.money(300)}" in text_of(at)

    # edit the entry, then delete it: the card goes back to nothing logged
    eid = int(prepare_entries_ids(pk_db)[-1])
    at.button(key=f"cash-edit-{eid}").click().run()
    at.number_input(key=f"cash-edit-amount-{eid}").set_value(400.0)
    at.button(key="cash-edit-save").click().run()
    assert_clean(at)
    assert stats(at)["logged"] == fmt.money(400)
    at.button(key=f"cash-del-{eid}").click().run()
    assert_clean(at)
    assert stats(at)["logged"] == fmt.money(0) and stats(at)["unaccounted"] == fmt.money(withdrawn)
    assert spent_kpi(at) == spent_before
    assert '<span class="fa-src cash">' not in text_of(at)
    assert "No cash logged yet. Try: chai 150" in text_of(at)


def test_confirm_flow_change_category_and_pick_when_no_rule(pk_db, monkeypatch):
    month, _ = cash_month(pk_db)
    at = open_app(pk_db, monkeypatch)
    at.selectbox(key="month").set_value(month).run()
    at.date_input(key="cash-date").set_value(dt.date.fromisoformat(f"{month}-12"))
    at.text_input(key="cash-quick").set_value("chai 150")
    at.button(key="cash-quick-go").click().run()
    at.selectbox(key="cash-quick-cat").set_value("Groceries").run()        # "Change"
    assert f"Groceries - {fmt.money(150)}" in text_of(at)
    at.button(key="cash-quick-save").click().run()
    assert_clean(at)
    assert donut(at)["Groceries"] >= 150

    at.text_input(key="cash-quick").set_value("gift 300")
    at.button(key="cash-quick-go").click().run()
    at.button(key="cash-quick-save").click().run()                         # nothing picked yet
    assert any("Pick a spending category" in w.value for w in at.warning)
    at.selectbox(key="cash-quick-cat").set_value("Shopping")
    at.button(key="cash-quick-save").click().run()
    assert_clean(at)
    assert stats(at)["logged"] == fmt.money(450)
    shown = text_of(at)                                                     # recent list: newest first, max 3
    assert shown.index(">gift<") < shown.index(">chai<")


def test_recent_list_shows_only_the_last_three(pk_db, monkeypatch):
    from src.ui import data
    month, _ = cash_month(pk_db)
    for i in range(1, 5):
        data.add_cash(pk_db, 10 * i, "Shopping", dt.date.fromisoformat(f"{month}-0{i}"), f"item{i}")
    at = open_app(pk_db, monkeypatch)
    at.selectbox(key="month").set_value(month).run()
    assert_clean(at)
    deletes = [b for b in at.button if (b.key or "").startswith("cash-del-")]
    assert len(deletes) == 3
    assert "item1" not in text_of(at) and "item4" in text_of(at)


def test_logging_more_than_withdrawn_is_flagged(pk_db, monkeypatch):
    month, withdrawn = cash_month(pk_db)
    at = open_app(pk_db, monkeypatch)
    at.selectbox(key="month").set_value(month).run()
    at.number_input(key="cash-amount").set_value(withdrawn + 500)
    at.selectbox(key="cash-cat").set_value("Shopping")
    at.date_input(key="cash-date").set_value(dt.date.fromisoformat(f"{month}-20"))
    at.button(key="cash-add").click().run()
    assert_clean(at)
    assert stats(at)["unaccounted"] == fmt.money(0)
    assert "cash from outside the account" in text_of(at)
    assert UNACCOUNTED not in donut(at)


def test_bad_entry_shows_a_friendly_message(pk_db, monkeypatch):
    at = open_app(pk_db, monkeypatch)
    at.number_input(key="cash-amount").set_value(100.0)
    at.button(key="cash-add").click().run()                      # no category picked
    assert not at.exception
    assert any("Pick a spending category" in w.value for w in at.warning)
