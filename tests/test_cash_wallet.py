"""Cash wallet: the formula (src/analytics/cash.py), entry storage + quick line (src/cash_wallet.py) and the
agent's get_cash_summary tool. Offline: fake provider only."""
from __future__ import annotations

import pandas as pd
import pytest

from src import analytics
from src.analytics.cash import OUTSIDE_FLAG, UNACCOUNTED, apply_cash, cash_history, cash_summary, prepare_entries
from src.cash_wallet import (CashEntryError, add_entry, delete_entry, load_entries, parse_quick_line,
                             propose_category, update_entry)
from src.db import connect, ingest, load_transactions

CASH = "Cash Withdrawal"


def entries(*rows) -> pd.DataFrame:
    """rows of (date, amount, category[, note])."""
    return pd.DataFrame([{"id": i, "date": r[0], "amount": r[1], "category": r[2], "note": r[3] if len(r) > 3 else ""}
                         for i, r in enumerate(rows, start=1)])


@pytest.fixture
def txns(make_txns):
    return make_txns([
        ("2026-01-01", "SALARY", 50000.0, "Income"),
        ("2026-01-05", "ATM CASH WITHDRAWAL", -6000.0, CASH),
        ("2026-01-20", "ATM CASH WITHDRAWAL", -4000.0, CASH),
        ("2026-01-10", "KROGER", -2500.0, "Groceries"),
        ("2026-02-01", "SALARY", 50000.0, "Income"),
        ("2026-02-03", "KROGER", -3000.0, "Groceries"),          # February: no withdrawals
        ("2026-03-01", "SALARY", 50000.0, "Income"),
        ("2026-03-02", "ATM CASH WITHDRAWAL", -5000.0, CASH),
    ])


def spent(df, month):
    return analytics.monthly_summary(month, df).total_spent


def cats(df, month):
    return {c.category: c.amount for c in analytics.monthly_summary(month, df).categories}


# ---- formula ------------------------------------------------------------------------------------

def test_no_entries_unaccounted_equals_withdrawals(txns):
    c = cash_summary("2026-01", txns)
    assert (c.withdrawn, c.logged, c.unaccounted, c.outside, c.accounted_pct) == (10000.0, 0.0, 10000.0, 0.0, 0.0)
    adj = apply_cash(txns)
    for m in ("2026-01", "2026-02", "2026-03"):
        assert spent(adj, m) == spent(txns, m)
    assert cats(adj, "2026-01") == {"Groceries": 2500.0, UNACCOUNTED: 10000.0}


def test_logged_less_than_withdrawn(txns):
    ent = entries(("2026-01-07", 1500.0, "Dining Out", "chai"), ("2026-01-25", 2500.0, "Groceries"))
    c = cash_summary("2026-01", txns, ent)
    assert (c.withdrawn, c.logged, c.unaccounted, c.outside, c.accounted_pct) == (10000.0, 4000.0, 6000.0, 0.0, 40.0)
    adj = apply_cash(txns, ent)
    assert spent(adj, "2026-01") == spent(txns, "2026-01") == 12500.0      # Spent unchanged
    assert cats(adj, "2026-01") == {"Groceries": 5000.0, UNACCOUNTED: 6000.0, "Dining Out": 1500.0}
    # withdrawals are reduced earliest first: the 6000 row is used up by 4000 logged
    unacc = adj[(adj["category"] == UNACCOUNTED) & (adj["month"] == "2026-01")]
    assert sorted(unacc["amount"]) == [-4000.0, -2000.0]


def test_logged_more_than_withdrawn_counts_as_spending_and_is_flagged(txns):
    ent = entries(("2026-03-05", 7000.0, "Shopping"))
    c = cash_summary("2026-03", txns, ent)
    assert (c.withdrawn, c.logged, c.unaccounted, c.outside, c.accounted_pct) == (5000.0, 7000.0, 0.0, 2000.0, 100.0)
    assert OUTSIDE_FLAG in c.method
    adj = apply_cash(txns, ent)
    assert spent(adj, "2026-03") == spent(txns, "2026-03") + 2000.0
    assert cats(adj, "2026-03") == {"Shopping": 7000.0}


def test_entries_in_a_month_with_no_withdrawals(txns):
    ent = entries(("2026-02-10", 800.0, "Transport", "rickshaw"))
    c = cash_summary("2026-02", txns, ent)
    assert (c.withdrawn, c.logged, c.unaccounted, c.outside, c.accounted_pct) == (0.0, 800.0, 0.0, 800.0, None)
    assert spent(apply_cash(txns, ent), "2026-02") == 3800.0


def test_wallet_balance_is_cumulative(txns):
    ent = entries(("2026-01-07", 3000.0, "Dining Out"), ("2026-02-10", 800.0, "Transport"))
    balances = {c.month: c.wallet_balance for c in cash_history(txns, ent)}
    assert balances == {"2026-01": 7000.0, "2026-02": 6200.0, "2026-03": 11200.0}


def test_income_and_other_months_untouched(txns):
    ent = entries(("2026-01-07", 3000.0, "Dining Out"))
    adj = apply_cash(txns, ent)
    for m in ("2026-01", "2026-02", "2026-03"):
        assert analytics.monthly_summary(m, adj).total_income == analytics.monthly_summary(m, txns).total_income
    assert cats(adj, "2026-03") == cats(apply_cash(txns), "2026-03")


# ---- regression on the PK fixture: no entries -> every figure is today's ---------------------------

@pytest.fixture(scope="module")
def pk():
    from src.categorizer import categorize_transactions
    from src.llm.fake_provider import FakeProvider
    from src.parser import parse_statement
    from tests.pk_fixture import FIXTURE
    conn = connect(":memory:")
    ingest(conn, parse_statement(FIXTURE))
    categorize_transactions(conn, provider=FakeProvider())
    df = load_transactions(conn)
    conn.close()
    return df


def test_pk_no_entries_spent_and_category_totals_unchanged(pk):
    adj = apply_cash(pk)
    assert (pk["category"] == CASH).any()
    for m in analytics.all_months(pk):
        before, after = analytics.monthly_summary(m, pk), analytics.monthly_summary(m, adj)
        assert (after.total_spent, after.total_income, after.net_saved) == \
               (before.total_spent, before.total_income, before.net_saved)
        old = {c.category: c.amount for c in before.categories}
        new = {c.category: c.amount for c in after.categories}
        assert new.pop(UNACCOUNTED, 0.0) == old.pop(CASH, 0.0) == cash_summary(m, pk).unaccounted
        assert new == old
    assert analytics.month_history(adj, previous="data") == analytics.month_history(pk, previous="data")


def test_pk_logged_within_withdrawals_keeps_spent(pk):
    month = next(m for m in analytics.all_months(pk) if cash_summary(m, pk).withdrawn >= 1000)
    ent = entries((f"{month}-15", 1000.0, "Groceries", "sabzi"))
    adj = apply_cash(pk, ent)
    assert spent(adj, month) == spent(pk, month)
    assert cats(adj, month)[UNACCOUNTED] == round(cash_summary(month, pk).withdrawn - 1000.0, 2)


# ---- storage: add / edit / delete, redacted notes -------------------------------------------------

def test_add_update_delete_entry(txns):
    conn = connect(":memory:")
    eid = add_entry(conn, 2500, "groceries", "2026-01-12", "weekly sabzi")
    e = load_entries(conn)
    assert list(e[["id", "date", "amount", "category"]].itertuples(index=False, name=None)) == \
           [(eid, "2026-01-12", 2500.0, "Groceries")]
    assert cash_summary("2026-01", txns, e).unaccounted == 7500.0
    update_entry(conn, eid, 4000, "Dining Out", "2026-01-13", "dinner")
    assert cash_summary("2026-01", txns, load_entries(conn)).unaccounted == 6000.0
    delete_entry(conn, eid)
    after = cash_summary("2026-01", txns, load_entries(conn))
    assert (after.logged, after.unaccounted) == (0.0, 10000.0)
    assert spent(apply_cash(txns, load_entries(conn)), "2026-01") == spent(txns, "2026-01")


def test_note_is_redacted_before_it_is_stored():
    conn = connect(":memory:")
    add_entry(conn, 500, "Transfers", "2026-01-12", "paid to Ahmed Khan 0300-1234567")
    note = load_entries(conn)["note"].iloc[0]
    assert "Ahmed" not in note and "1234567" not in note and "[NAME]" in note


@pytest.mark.parametrize("amount, category, when", [(0, "Groceries", "2026-01-01"), (-5, "Groceries", "2026-01-01"),
                                                    (10, "Income", "2026-01-01"), (10, "Cash Withdrawal", "2026-01-01"),
                                                    (10, "Nope", "2026-01-01"), ("abc", "Groceries", "2026-01-01")])
def test_invalid_entries_are_rejected(amount, category, when):
    with pytest.raises(CashEntryError):
        add_entry(connect(":memory:"), amount, category, when)


def test_old_db_gets_the_cash_table(tmp_path):
    import sqlite3
    path = tmp_path / "old.db"
    sqlite3.connect(path).executescript("CREATE TABLE goals (id INTEGER PRIMARY KEY);").close()
    conn = connect(path)
    assert load_entries(conn).empty
    conn.close()


# ---- quick line: keyword rules only --------------------------------------------------------------

@pytest.mark.parametrize("line, amount, note, category", [
    ("chai 150", 150.0, "chai", "Dining Out"),
    ("petrol Rs 2,000", 2000.0, "petrol", "Transport"),
    ("Rs. 450.50 pharmacy", 450.5, "pharmacy", "Health"),
    ("karyana 1200", 1200.0, "karyana", "Groceries"),
    ("gift for cousin 900", 900.0, "gift for cousin", None),
    ("350", 350.0, "", None),
])
def test_parse_quick_line(line, amount, note, category):
    q = parse_quick_line(line)
    assert (q.amount, q.note, q.category) == (amount, note, category)


def test_quick_line_without_amount_is_an_error():
    with pytest.raises(CashEntryError):
        parse_quick_line("chai")


def test_proposal_never_offers_income_or_cash():
    assert propose_category("salary") is None
    assert propose_category("atm cash withdrawal") is None


def test_cash_code_calls_no_model():
    import ast
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    for rel in ("src/cash_wallet.py", "src/analytics/cash.py"):
        tree = ast.parse((root / rel).read_text(encoding="utf-8"))
        mods = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)] + \
               [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        assert not [m for m in mods if m and m.startswith(("src.llm", "src.agent", "ollama"))], rel


# ---- agent tool ---------------------------------------------------------------------------------

def test_get_cash_summary_tool(txns):
    from src.agent.tools import run_tool
    ent = entries(("2026-01-07", 1500.0, "Dining Out", "secret note text"))
    r = run_tool("get_cash_summary", {"month": "2026-01"}, txns, ent)
    assert r.ok and r.source == "analytics.cash_summary"
    assert {k: r.data[k] for k in ("withdrawn", "logged", "unaccounted", "accounted_pct")} == \
           {"withdrawn": 10000.0, "logged": 1500.0, "unaccounted": 8500.0, "accounted_pct": 15.0}
    assert "secret" not in repr(r.data)                         # notes never reach a model
    assert set(r.txn_ids) == {2, 3}
    assert "error" in run_tool("get_cash_summary", {"month": "2025-01"}, txns, ent).data
    assert "error" in run_tool("get_cash_summary", {}, txns, ent).data


def test_existing_tools_ignore_cash_entries(txns):
    from src.agent.tools import run_tool
    ent = entries(("2026-01-07", 9999.0, "Dining Out"))
    for name, args in [("get_monthly_summary", {"month": "2026-01"}), ("get_month_over_month", {"month": "2026-03"}),
                       ("get_category_spending", {"category": "Dining Out", "month": "2026-01"})]:
        assert run_tool(name, args, txns, ent).data == run_tool(name, args, txns).data


def test_agent_answers_a_cash_question_with_verified_numbers(txns):
    from src.agent.chat import FinanceAgent
    from src.llm.fake_provider import FakeProvider
    ent = entries(("2026-01-07", 2500.0, "Groceries", "sabzi"))
    agent = FinanceAgent(FakeProvider(), txns=txns, cash=ent)
    reply = agent.ask("How much cash is unaccounted in January 2026?")
    t = reply.trace
    assert not t.refused and t.verified and not t.fallback_used
    call = t.model_tool_calls[-1]
    assert call.name == "get_cash_summary" and call.args == {"month": "2026-01"}
    assert "Rs 7,500.00" in reply.text and "25.0%" in reply.text
    assert "sabzi" not in reply.text


def test_agent_without_entries_reports_all_withdrawn_cash_unaccounted(txns):
    from src.agent.chat import FinanceAgent
    from src.llm.fake_provider import FakeProvider
    reply = FinanceAgent(FakeProvider(), txns=txns).ask("How much of my cash is accounted for in March 2026?")
    assert reply.trace.verified and reply.trace.model_tool_calls[-1].result["unaccounted"] == 5000.0


def test_prepare_entries_empty():
    assert prepare_entries(entries()).empty
