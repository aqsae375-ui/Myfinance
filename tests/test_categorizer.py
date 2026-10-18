import sqlite3

import pandas as pd
import pytest

from src.categorizer import categorize_transactions, set_manual_category
from src.categories import CATEGORIES
from src.db import connect, ingest, ingest_csv, load_transactions
from src.llm.fake_provider import FakeProvider
from scripts.generate_data import INJECTION_TEXT

PII = ["Jane Doe", "555-201-9981", "jane.doe@example.com", "1234567890"]


def _df(rows):
    df = pd.DataFrame(rows, columns=["description", "amount"])
    df["date"] = pd.Timestamp("2026-09-01")
    df["type"] = df["amount"].map(lambda a: "credit" if a >= 0 else "debit")
    return df[["date", "description", "amount", "type"]]


@pytest.fixture
def conn():
    c = connect(":memory:")
    yield c
    c.close()


@pytest.fixture
def mixed(conn):
    ingest(conn, _df([
        ("NETFLIX.COM", -15.49),                           # rule
        ("MYSTERY MERCHANT 42", -20.00),                   # llm
        ("MYSTERY MERCHANT 42", -25.00),                   # llm, same description
        ("WHOLEFDS MKT #10234", -80.00),                   # llm
        ("Paid Jane Doe 555-201-9981 jane.doe@example.com acct 1234567890", -60.00),  # llm, PII
        (INJECTION_TEXT, -9.99),                           # llm, injection
    ]), known_names=["Jane Doe"])
    return conn


def _by_desc(conn):
    df = load_transactions(conn)
    return {r.raw_description: r for r in df.itertuples()}


def test_rules_then_llm_saved_with_source_and_confidence(mixed):
    p = FakeProvider({"wholefds": "Groceries"})
    rep = categorize_transactions(mixed, p)
    rows = _by_desc(mixed)
    assert (rows["NETFLIX.COM"].category, rows["NETFLIX.COM"].category_source) == ("Subscriptions", "rule")
    assert rows["NETFLIX.COM"].confidence >= 0.9
    w = rows["WHOLEFDS MKT #10234"]
    assert (w.category, w.category_source, w.confidence, w.needs_review) == ("Groceries", "llm", 0.9, 0)
    assert rows["MYSTERY MERCHANT 42"].category == "Other"
    # The injection row never reaches the model: it goes straight to review.
    assert rep.rows == 6 and rep.by_rule == 1 and rep.by_llm == 4 and rep.unsure == 1
    assert load_transactions(mixed)["category"].dropna().isin(CATEGORIES).all()


def test_identical_descriptions_hit_llm_once_and_cache_persists(mixed):
    p = FakeProvider()
    rep = categorize_transactions(mixed, p)
    assert p.seen_descriptions.count("MYSTERY MERCHANT 42") == 1
    assert rep.llm_unique_sent == 3 and rep.llm_cache_hits == 0

    p2 = FakeProvider()
    rep2 = categorize_transactions(mixed, p2, recategorize=True)
    assert p2.seen_descriptions == [] and p2.stats.calls == 0
    assert rep2.llm_cache_hits == 3 and rep2.llm_unique_sent == 0


def test_cache_is_per_provider(mixed):
    categorize_transactions(mixed, FakeProvider())
    other = FakeProvider()
    other.name = "fake:other-model"
    categorize_transactions(mixed, other, recategorize=True)
    assert other.stats.calls == 1  # a different model does not reuse another model's answers


def test_second_run_only_touches_uncategorized_rows(mixed):
    categorize_transactions(mixed, FakeProvider())
    ingest(mixed, _df([("BRAND NEW SHOP", -5.0)]))
    p = FakeProvider()
    rep = categorize_transactions(mixed, p)
    assert rep.rows == 1 and p.seen_descriptions == ["BRAND NEW SHOP"]


def test_manual_override_survives_reruns(mixed):
    categorize_transactions(mixed, FakeProvider())
    txn_id = int(load_transactions(mixed).query("raw_description == 'NETFLIX.COM'")["id"].iloc[0])
    set_manual_category(mixed, txn_id, "entertainment")  # normalized to canonical case

    for recat in (False, True, True):
        categorize_transactions(mixed, FakeProvider({"netflix": "Shopping"}), recategorize=recat)
    row = mixed.execute("SELECT category, category_source, confidence FROM transactions WHERE id = ?",
                        (txn_id,)).fetchone()
    assert row == ("Entertainment", "manual", 1.0)


def test_manual_override_validates(mixed):
    with pytest.raises(ValueError):
        set_manual_category(mixed, 1, "Essentials")
    with pytest.raises(KeyError):
        set_manual_category(mixed, 9999, "Health")


def test_injection_does_not_change_format_or_category(mixed):
    p = FakeProvider({"wholefds": "Groceries"})  # obey_injections=True: simulates a hijacked model
    categorize_transactions(mixed, p)
    rows = _by_desc(mixed)
    inj = rows[INJECTION_TEXT]
    assert inj.category is None and (inj.category_source, inj.confidence, inj.needs_review) == ("llm", 0.0, 1)
    # Neighbours are unaffected, and the injected text is never sent to the model at all.
    assert rows["WHOLEFDS MKT #10234"].category == "Groceries"
    assert load_transactions(mixed)["category"].dropna().isin(CATEGORIES).all()
    assert not any(INJECTION_TEXT in u or INJECTION_TEXT in s for s, u in p.prompts)


def test_no_raw_description_ever_reaches_provider(mixed):
    p = FakeProvider()
    categorize_transactions(mixed, p)
    sent = "\n".join(u for _, u in p.prompts) + "\n".join(s for s, _ in p.prompts)
    for secret in PII + ["Jane", "Doe"]:
        assert secret not in sent
    redacted = set(load_transactions(mixed)["redacted_description"])
    assert set(p.seen_descriptions) <= redacted
    raw_only = set(load_transactions(mixed)["raw_description"]) - redacted
    assert raw_only and not raw_only & set(p.seen_descriptions)


def test_no_raw_description_on_full_synthetic_statement(statement_csv, statement_txns, tmp_path):
    db = tmp_path / "t.db"
    ingest_csv(statement_csv, db)
    c = connect(db)
    p = FakeProvider()
    categorize_transactions(c, p)
    sent = "\n".join(u for _, u in p.prompts)
    for t in statement_txns:
        for secret in t.pii:
            assert secret not in sent
    df = load_transactions(c)
    assert df.loc[df["category"].isna(), "raw_description"].tolist() == [INJECTION_TEXT]  # only the injection
    c.close()


def test_provider_not_built_when_rules_cover_everything(conn, monkeypatch):
    ingest(conn, _df([("NETFLIX.COM", -15.49), ("UBER TRIP", -12.0)]))

    def boom(*a, **k):
        raise AssertionError("LLM provider should not be constructed")
    monkeypatch.setattr("src.categorizer.get_provider", boom)
    rep = categorize_transactions(conn)
    assert rep.by_rule == 2 and rep.provider is None


def test_stage1_db_is_migrated(tmp_path):
    db = tmp_path / "old.db"
    old = sqlite3.connect(db)
    old.executescript("""
        CREATE TABLE transactions (id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT NOT NULL,
            raw_description TEXT NOT NULL, redacted_description TEXT NOT NULL, amount REAL NOT NULL,
            type TEXT NOT NULL, category TEXT, source_file TEXT, row_hash TEXT NOT NULL UNIQUE,
            ingested_at TEXT NOT NULL DEFAULT (datetime('now')));
        INSERT INTO transactions (date, raw_description, redacted_description, amount, type, row_hash)
            VALUES ('2026-03-01', 'NETFLIX.COM', 'NETFLIX.COM', -15.49, 'debit', 'h1');
    """)
    old.close()
    c = connect(db)
    cols = {r[1] for r in c.execute("PRAGMA table_info(transactions)")}
    assert {"category_source", "confidence", "needs_review"} <= cols
    categorize_transactions(c, FakeProvider())
    assert c.execute("SELECT category, category_source FROM transactions").fetchone() == ("Subscriptions", "rule")
    c.close()
