import re

import pytest

from src.db import connect, ingest, ingest_csv, load_transactions
from src.parser import parse_statement


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test.db"


def test_ingest_stores_all_rows_with_empty_category(statement_csv, statement_txns, db_path):
    assert ingest_csv(statement_csv, db_path) == len(statement_txns)
    conn = connect(db_path)
    df = load_transactions(conn)
    conn.close()
    assert len(df) == len(statement_txns)
    assert df["category"].isna().all()
    assert set(df["type"]) == {"debit", "credit"}


def test_ingest_is_idempotent_but_keeps_duplicate_charges(statement_csv, db_path):
    ingest_csv(statement_csv, db_path)
    assert ingest_csv(statement_csv, db_path) == 0
    conn = connect(db_path)
    dupes = conn.execute(
        "SELECT COUNT(*) FROM transactions WHERE raw_description = 'AMZN MKTP US*2K4L81'"
    ).fetchone()[0]
    conn.close()
    assert dupes == 2


def test_redacted_column_has_no_pii_raw_column_keeps_original(statement_csv, statement_txns, db_path):
    ingest_csv(statement_csv, db_path)
    conn = connect(db_path)
    rows = conn.execute("SELECT raw_description, redacted_description FROM transactions").fetchall()
    conn.close()
    secrets = [s.lower() for t in statement_txns for s in t.pii]
    raw_text = " ".join(r[0] for r in rows).lower()
    redacted_text = " ".join(r[1] for r in rows).lower()
    assert all(s in raw_text for s in secrets)
    assert not any(s in redacted_text for s in secrets)
    assert "@" not in redacted_text
    assert not re.search(r"\d(?:[\s().+-]*\d){6,}", redacted_text)


def test_dates_stored_as_iso_and_amounts_as_real(statement_csv, db_path):
    ingest_csv(statement_csv, db_path)
    conn = connect(db_path)
    rows = conn.execute("SELECT date, typeof(amount) FROM transactions").fetchall()
    conn.close()
    assert all(re.fullmatch(r"\d{4}-\d{2}-\d{2}", d) for d, _ in rows)
    assert {t for _, t in rows} == {"real"}


def test_known_names_applied_on_ingest(statement_csv):
    conn = connect(":memory:")
    df = parse_statement(statement_csv).head(3).copy()
    df.loc[0, "description"] = "GIFT FROM GRANDMA ELEANOR VOSS"
    ingest(conn, df, known_names=["Eleanor Voss"])
    redacted = conn.execute("SELECT redacted_description FROM transactions").fetchall()
    conn.close()
    assert all("VOSS" not in r[0] for r in redacted)
