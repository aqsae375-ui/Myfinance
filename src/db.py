"""SQLite storage for transactions: schema, ingest, load.

Run:  python -m src.db data/raw/statement.csv [--db data/finance.db] [--name "Account Holder"]
"""
from __future__ import annotations

import argparse
import hashlib
import sqlite3
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.parser import parse_statement
from src.redaction import redact

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "finance.db"

TRANSACTIONS_DDL = """
CREATE TABLE IF NOT EXISTS {table} (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    date                 TEXT NOT NULL,                  -- ISO YYYY-MM-DD
    raw_description      TEXT NOT NULL,                  -- original; stays local, never sent to a model
    redacted_description TEXT NOT NULL,                  -- PII-masked; the only text a model sees
    amount               REAL NOT NULL,                  -- signed: negative = money out
    type                 TEXT NOT NULL CHECK (type IN ('debit', 'credit')),
    category             TEXT,                           -- taxonomy name (src/categories.py); NULL = not categorized
    category_source      TEXT CHECK (category_source IN ('rule', 'llm', 'manual', 'learned')),
    confidence           REAL,
    needs_review         INTEGER NOT NULL DEFAULT 0,     -- 1 = model was Unsure or below 0.6: the review queue
    source_file          TEXT,
    row_hash             TEXT NOT NULL UNIQUE,
    ingested_at          TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

SCHEMA = TRANSACTIONS_DDL.format(table="transactions") + """
CREATE INDEX IF NOT EXISTS idx_transactions_date ON transactions(date);

-- One LLM answer per (merchant key, provider/model): a merchant is sent to a model at most once.
-- `description` holds the cache key ("<in|out>|<merchant key>[|hint=<category>]", see src/categorizer.py).
CREATE TABLE IF NOT EXISTS llm_category_cache (
    description TEXT NOT NULL,
    provider    TEXT NOT NULL,
    category    TEXT NOT NULL,                               -- taxonomy name or 'Unsure'
    valid       INTEGER NOT NULL,                            -- 0 = model answer was invalid -> Unsure
    confidence  REAL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (description, provider)
);

-- Merchant rules the user saved ("always put this merchant in X"). Keyed by the merchant key of the
-- redacted description; the 20 newest are shown to the LLM as examples.
CREATE TABLE IF NOT EXISTS merchant_rules (
    merchant_key TEXT PRIMARY KEY,
    category     TEXT NOT NULL,
    updated_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Savings goals (src/advisor/goals.py). The newest row is the active goal.
CREATE TABLE IF NOT EXISTS goals (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    kind         TEXT NOT NULL CHECK (kind IN ('monthly_savings', 'target_by_date')),
    name         TEXT NOT NULL,
    amount       REAL NOT NULL CHECK (amount > 0),   -- per month, or the total target
    target_date  TEXT,                               -- ISO YYYY-MM-DD; target_by_date only
    saved_so_far REAL NOT NULL DEFAULT 0,            -- already saved toward a target_by_date goal
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Cash the user logged as spent from the cash wallet (src/cash_wallet.py). Withdrawals (category
-- Cash Withdrawal) fill the wallet; these entries say where the cash went. `note` is stored redacted.
CREATE TABLE IF NOT EXISTS cash_entries (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    date       TEXT NOT NULL,                       -- ISO YYYY-MM-DD
    amount     REAL NOT NULL CHECK (amount > 0),    -- money spent in cash (positive)
    category   TEXT NOT NULL,                       -- taxonomy name
    note       TEXT NOT NULL DEFAULT '',            -- PII-redacted
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

# Columns added after Stage 1; added in place to existing databases by init_db().
MIGRATION_COLUMNS = {
    "category_source": "TEXT CHECK (category_source IN ('rule', 'llm', 'manual'))",
    "confidence": "REAL",
    "needs_review": "INTEGER NOT NULL DEFAULT 0",
}
_COLUMNS = ("id, date, raw_description, redacted_description, amount, type, category, category_source, "
            "confidence, needs_review, source_file, row_hash, ingested_at")


def connect(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    if str(db_path) != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    init_db(conn)
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    existing = {row[1] for row in conn.execute("PRAGMA table_info(transactions)")}
    for col, decl in MIGRATION_COLUMNS.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE transactions ADD COLUMN {col} {decl}")
    if "confidence" not in {row[1] for row in conn.execute("PRAGMA table_info(llm_category_cache)")}:
        conn.execute("ALTER TABLE llm_category_cache ADD COLUMN confidence REAL")
    conn.commit()
    sql = conn.execute("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'transactions'").fetchone()[0]
    if "'learned'" not in sql:
        _rebuild_transactions(conn)


def _rebuild_transactions(conn: sqlite3.Connection) -> None:
    """Stage 2d added category_source 'learned'. SQLite cannot change a CHECK constraint in place,
    so copy the rows into a table with the new schema."""
    with conn:
        conn.execute(TRANSACTIONS_DDL.format(table="transactions_new"))
        conn.execute(f"INSERT INTO transactions_new ({_COLUMNS}) SELECT {_COLUMNS} FROM transactions")
        conn.execute("DROP TABLE transactions")
        conn.execute("ALTER TABLE transactions_new RENAME TO transactions")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_transactions_date ON transactions(date)")


def row_hashes(df: pd.DataFrame) -> list[str]:
    """Stable per-row key. Identical rows can be legitimate (a duplicate charge), so the key
    includes the occurrence number; re-ingesting the same file yields the same keys."""
    keys = (df["date"].dt.strftime("%Y-%m-%d") + "|" + df["description"] + "|"
            + df["amount"].map("{:.2f}".format))
    occurrence = keys.groupby(keys).cumcount()
    return [hashlib.sha256(f"{k}|{n}".encode()).hexdigest() for k, n in zip(keys, occurrence)]


def ingest(conn: sqlite3.Connection, df: pd.DataFrame, source_file: str | None = None,
           known_names: Iterable[str] = ()) -> int:
    """Insert parsed transactions (output of parser.parse_statement). Returns rows added;
    rows already present are skipped, so ingest is idempotent."""
    known_names = list(known_names)
    rows = [
        (d.strftime("%Y-%m-%d"), desc, redact(desc, known_names), float(amt), typ, source_file, h)
        for d, desc, amt, typ, h in zip(df["date"], df["description"], df["amount"], df["type"],
                                        row_hashes(df))
    ]
    before = conn.total_changes
    with conn:
        conn.executemany(
            "INSERT OR IGNORE INTO transactions "
            "(date, raw_description, redacted_description, amount, type, source_file, row_hash) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
    return conn.total_changes - before


def ingest_csv(csv_path: str | Path, db_path: str | Path = DEFAULT_DB_PATH,
               known_names: Iterable[str] = ()) -> int:
    df = parse_statement(csv_path)
    conn = connect(db_path)
    try:
        return ingest(conn, df, Path(csv_path).name, known_names)
    finally:
        conn.close()


def load_transactions(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM transactions ORDER BY date, id", conn, parse_dates=["date"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Parse, redact and ingest a bank-statement CSV.")
    ap.add_argument("csv", type=Path)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    ap.add_argument("--name", action="append", default=[], help="Known name to always redact (repeatable)")
    args = ap.parse_args()

    added = ingest_csv(args.csv, args.db, args.name)
    conn = connect(args.db)
    total = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    conn.close()
    print(f"Ingested {added} new rows from {args.csv} -> {args.db} ({total} total)")


if __name__ == "__main__":
    main()
