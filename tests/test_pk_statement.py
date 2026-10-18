"""Stage 2b: Pakistani bank statements. Uses only the synthetic fixture (tests/pk_fixture.py); the real
statement is never read here, except by the privacy guard at the bottom, which only checks that none of
its masked words leak into test files and reports a count, never the words."""
from __future__ import annotations

import csv
import io
import re
from pathlib import Path

import pandas as pd
import pytest

from src import analytics
from src.categorizer import categorize_transactions
from src.db import connect, ingest, load_transactions
from src.llm.fake_provider import FakeProvider
from src.parser import StatementParseError, parse_statement
from src.redaction import redact
from src.rules import find_rule, match_rule
from tests.pk_fixture import FAKE_PEOPLE, FIXTURE, ROWS, write_fixture

ROOT = Path(__file__).resolve().parent.parent
REAL_STATEMENTS = [ROOT / "data" / "raw" / n for n in ("statement_2025.csv", "real_statement_2025.csv")]


@pytest.fixture(scope="module")
def pk_df():
    """The fixture through the real pipeline: parse, redact, ingest, categorize (rules + offline fake)."""
    conn = connect(":memory:")
    ingest(conn, parse_statement(FIXTURE))
    report = categorize_transactions(conn, provider=FakeProvider())
    df = load_transactions(conn)
    conn.close()
    return df, report


# ---- fixture ----

def test_committed_fixture_matches_generator(tmp_path):
    assert write_fixture(tmp_path / "pk.csv").read_text(encoding="utf-8") == FIXTURE.read_text(encoding="utf-8")


def test_fixture_covers_every_row_type():
    expected = {c for *_, c in ROWS}
    assert expected >= {"Interest", "Tax", "Cash Withdrawal", "Bank Fees", "Transfers In", "Charity/Donations",
                        "Dining Out", "Groceries", None}
    assert {d.month for d, *_ in ROWS} == set(range(1, 13))


# ---- parser ----

def test_parser_ignores_value_date_and_balance():
    df = parse_statement(FIXTURE)
    assert list(df.columns) == ["date", "description", "amount", "type"]
    assert len(df) == len(ROWS)
    assert list(df["date"].dt.date) == [d for d, *_ in ROWS]  # posting Date, never Value Date
    assert list(df["amount"]) == [a for _, _, a, _ in ROWS]


def _rewrite(fmt_date, debit_credit=False) -> str:
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["Date", "Description", "Debit", "Credit", "Balance"] if debit_credit
               else ["Date", "Value Date", "Description", "Amount", "Balance"])
    for d, desc, amt, _ in ROWS:
        if debit_credit:
            w.writerow([fmt_date(d), desc, f"{-amt:,.2f}" if amt < 0 else "", f"{amt:,.2f}" if amt > 0 else "", "0"])
        else:
            w.writerow([fmt_date(d), fmt_date(d), desc, f"{amt:,.2f}", "1,000.00"])
    return out.getvalue()


@pytest.mark.parametrize("fmt_date, debit_credit", [
    (lambda d: f"{d.day}/{d.month}/{d.year}", False),           # D/M/YYYY, as real exports print it
    (lambda d: f"{d:%d/%m/%Y}", False),                         # DD/MM/YYYY
    (lambda d: f"{d:%d/%m/%y}", False),                         # DD/MM/YY
    (lambda d: f"{d:%d/%m/%y}", True),                          # separate Debit / Credit columns
])
def test_parser_date_formats_and_debit_credit(fmt_date, debit_credit):
    df = parse_statement(io.StringIO(_rewrite(fmt_date, debit_credit)))
    assert list(df["date"].dt.date) == [d for d, *_ in ROWS]
    assert list(df["amount"]) == [a for _, _, a, _ in ROWS]


def test_parser_reports_bad_lines():
    text = ("Date,Value Date,Description,Amount,Balance\n2025-01-01,,Payment of Profit,10,0\n"
            "2025-13-01,,BANK CHARGES,-5,0\n01/02/2025,,BANK CHARGES,abc,0\n")
    with pytest.raises(StatementParseError, match=r"line 3, line 4"):
        parse_statement(io.StringIO(text))


def test_two_digit_year_is_never_year_26():
    df = parse_statement(io.StringIO("Date,Description,Amount\n05/03/26,BANK CHARGES,-5\n"))
    assert df["date"].iloc[0] == pd.Timestamp("2026-03-05")


# ---- redaction ----

@pytest.mark.parametrize("text, secrets", [
    ("Money Received from ZARA FICTIONAL STAN(200201) IBFT from EASYPAISA", ["ZARA", "FICTIONAL", "200201"]),
    ("Money Received from OMAR SAMPLE KH STAN(200203) IBFT from JAZZCASH BANK", ["OMAR", "SAMPLE", " KH "]),
    ("RAAST PYMT STAN (500501) MEEZAN BANK LIMITED Raast P2P Fund transfer from HINA PLACEHOLDER",
     ["HINA", "PLACEHOLDER", "500501"]),
    ("1-Link ATM Cash Withdrawal STAN (100101) TESTVILLE MAIN BRANCH - II", ["TESTVILLE", "100101", "MAIN"]),
    ("Sent to Bilal Testname via Easypaisa", ["Bilal", "Testname"]),
    ("TRANSFER REF NO: TX99812", ["TX99812"]),
])
def test_redaction_masks_people_refs_and_addresses(text, secrets):
    out = redact(text)
    assert not any(s in f" {out} " for s in secrets), out


@pytest.mark.parametrize("text, kept", [
    ("POS Purchase CHEEZIOUS STAN (300301)", "POS Purchase CHEEZIOUS STAN [REF]"),
    ("POS Purchase LAYERS BAKESHOP STAN (1)", "POS Purchase LAYERS BAKESHOP STAN [REF]"),
    ("POS Purchase THE MONAL STAN (2)", "POS Purchase THE MONAL STAN [REF]"),
    ("POS Purchase KFC STAN (3)", "POS Purchase KFC STAN [REF]"),
    ("Money Received from ALFALAH STAN(4) IBFT from ALFALAH", "Money Received from ALFALAH STAN [REF] IBFT from ALFALAH"),
    ("Payment of Profit", "Payment of Profit"),
    ("Withholding Tax Debit", "Withholding Tax Debit"),
    ("Charges Taxes Plus FED STAN (5) LaunchGood Inc. to PKR 128", "Charges Taxes Plus FED STAN [REF] LaunchGood Inc. to PKR 128"),
    ("MasterCard Annual Fee 2025", "MasterCard Annual Fee 2025"),
    ("PAYMENT TO NETFLIX", "PAYMENT TO NETFLIX"),
])
def test_redaction_keeps_merchants_banks_and_transaction_words(text, kept):
    assert redact(text) == kept


@pytest.mark.parametrize("bank", ["EASYPAISA", "HBL", "JAZZCASH", "ALFALAH", "MEEZAN"])
def test_redaction_keeps_bank_names(bank):
    assert bank in redact(f"Money Received from {bank} STAN(1) IBFT from {bank}")


def test_no_fixture_person_survives_redaction():
    for _, desc, _, _ in ROWS:
        out = redact(desc)
        for person in FAKE_PEOPLE:
            assert not any(part in out.split() for part in person.split() if len(part) > 2), out
        assert not re.search(r"STAN\s*\(?\d", out), out


# ---- rules ----

@pytest.mark.parametrize("desc, amount, expected", [(d, a, c) for _, d, a, c in ROWS], ids=lambda v: str(v)[:30])
def test_rules_classify_every_row_type(desc, amount, expected):
    hit = find_rule(redact(desc), amount)  # a deciding rule or, for generic merchant words, a hint
    assert (hit.category if hit else None) == expected


@pytest.mark.parametrize("desc, expected, decides", [
    ("POS Purchase ATM CASH WITHDRAWAL STAN [REF]", "Cash Withdrawal", True),
    ("POS Purchase ZORBO BIRYANI STAN [REF]", "Dining Out", True),      # a word that names one kind of shop
    ("POS Purchase ZORBO MART STAN [REF]", "Groceries", False),         # generic: a hint for the LLM only
    ("POS Purchase ZORBO MANDI STAN [REF]", "Dining Out", False),       # usually an Arabian rice restaurant
    ("POS Purchase ZORBO SABZI MANDI STAN [REF]", "Groceries", True),   # a vegetable market
    ("POS Purchase ZORBO BAKERS STAN [REF]", "Dining Out", False),
    ("POS Purchase UNKNOWN BOUTIQUE (PAYSA) STAN [REF]", None, None),   # unknown merchant: left to the LLM / user
])
def test_pos_merchant_keywords(desc, expected, decides):
    hit = find_rule(desc, -100.0)
    assert (hit.category if hit else None) == expected
    assert (hit.decides if hit else None) == decides
    assert (match_rule(desc, -100.0) is not None) == bool(decides)


def test_pipeline_categories(pk_df):
    df, report = pk_df
    expected = [c or "Other" for *_, c in ROWS]  # fake LLM: follows rule hints; Other for the unknown merchant
    assert list(df["category"]) == expected
    to_llm = [(d, a) for _, d, a, _ in ROWS if not ((h := find_rule(redact(d), a)) and h.decides)]
    assert report.by_llm == len(to_llm) and report.by_rule == len(ROWS) - len(to_llm)
    assert report.hinted == len(to_llm) - 1  # every LLM row but the unknown merchant carried a rule hint


# ---- analytics ----

def test_income_is_interest_plus_transfers_in_and_spent_is_all_debits(pk_df):
    df, _ = pk_df
    for month in analytics.all_months(df):
        s = analytics.monthly_summary(month, df)
        rows = [(a, c) for d, _, a, c in ROWS if f"{d:%Y-%m}" == month]
        assert s.total_income == round(sum(a for a, c in rows if c in ("Interest", "Transfers In")), 2)
        assert s.total_spent == round(-sum(a for a, _ in rows if a < 0), 2)
        assert s.other_credits == 0.0


def test_refund_credit_is_not_income(make_txns):
    df = make_txns([("2026-01-02", "SALARY", 1000.0, "Income"), ("2026-01-03", "SHOP REFUND", 50.0, "Shopping"),
                    ("2026-01-04", "SHOP", -200.0, "Shopping")])
    s = analytics.monthly_summary("2026-01", df)
    assert (s.total_income, s.other_credits, s.total_spent, s.net_saved) == (1000.0, 50.0, 200.0, 800.0)


def test_profit_tax_and_atm_are_never_subscriptions_or_anomalies(pk_df):
    df, _ = pk_df
    subs = analytics.detect_subscriptions(df)
    assert not any(s.category in ("Interest", "Tax", "Cash Withdrawal", "Bank Fees") for s in subs)
    assert not any(s.possibly_unused for s in subs)
    assert not any(a.category in ("Interest", "Tax") for a in analytics.detect_anomalies(df))


def test_cash_share_and_insight(pk_df):
    df, _ = pk_df
    c = analytics.cash_share("2025-12", df)
    cash = round(sum(-a for _, _, a, cat in ROWS if cat == "Cash Withdrawal"), 2)
    spent = round(sum(-a for _, _, a, _ in ROWS if a < 0), 2)
    assert (c.cash, c.spent, c.start_month) == (cash, spent, "2025-01")
    assert c.pct_of_spend == round(cash / spent * 100, 2)
    ins = [i for i in analytics.generate_insights("2025-12", df, limit=10) if i.kind == "cash_share"]
    assert len(ins) == 1
    assert ins[0].text.startswith(f"Rs {cash:,.0f} ({c.pct_of_spend:.0f}% of your spending) left your account as cash "
                                  "from Jan to Dec 2025.")
    assert ins[0].numbers["cash"] == cash and set(ins[0].txn_ids) == set(c.txn_ids)


def test_no_cash_insight_without_cash_that_month(make_txns):
    df = make_txns([("2025-01-05", "ATM CASH WITHDRAWAL", -500.0, "Cash Withdrawal"),
                    ("2025-02-05", "SHOP", -100.0, "Shopping")])
    assert not [i for i in analytics.generate_insights("2025-02", df) if i.kind == "cash_share"]
    assert [i for i in analytics.generate_insights("2025-01", df) if i.kind == "cash_share"]


def test_twelve_months_of_history_and_clean_merchants(pk_df):
    df, _ = pk_df
    assert len(analytics.month_history(df)) == 12
    merchants = {m.merchant for m in analytics.top_merchants("2025-04", 5, df)}
    assert "CHEEZIOUS" in merchants and not any("STAN" in m or "POS" in m for m in merchants)


# ---- privacy guard ----

# Generic words an address or code can contain; not identifying on their own.
GENERIC = {"STAN", "BRANCH", "MAIN", "MARKAZ", "ROAD", "BLOCK", "SECTOR", "PHASE", "TOWN", "CITY", "LIMITED", "BANK"}


def _masked_words(raw: str) -> set[str]:
    """Words of a raw description that redaction removed (names, addresses, codes)."""
    kept = set(re.findall(r"[A-Za-z0-9]+", redact(raw).upper()))
    return {w for w in re.findall(r"[A-Za-z]{3,}", raw.upper()) if w not in kept and w not in GENERIC}


@pytest.mark.parametrize("real", [p for p in REAL_STATEMENTS if p.exists()] or [None])
def test_no_real_statement_words_in_code_tests_or_docs(real):
    if real is None:
        pytest.skip("no real statement on this machine")
    with real.open(encoding="utf-8-sig") as f:
        secrets = set().union(*(_masked_words(r.get("Description", "")) for r in csv.DictReader(f)))
    files = [p for d in ("tests", "src", "scripts") for p in (ROOT / d).rglob("*")
             if p.suffix in (".py", ".csv", ".md") and "__pycache__" not in p.parts] + list(ROOT.glob("*.md"))
    test_text = " ".join(p.read_text(encoding="utf-8", errors="ignore").upper() for p in files)
    # Only a count is asserted, so a failure never prints the words themselves.
    n_leaked = sum(bool(re.search(rf"\b{w}\b", test_text)) for w in secrets)
    assert n_leaked == 0, "masked word(s) from the real statement appear in test files"
