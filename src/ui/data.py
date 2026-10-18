"""Where the app's data lives, cached analytics reads, and the two actions that write to a DB
(upload pipeline, manual category override).

Caching: every analytics call is wrapped in st.cache_data keyed on (db path, db_key), where db_key
fingerprints the transactions (ids + categories), so an upload or a category override invalidates it
automatically and switching months only hits the cache.

The dashboard path imports nothing from src.llm / src.agent: the categorizer (which can build a
provider) is imported lazily inside the upload and override actions only.
"""
from __future__ import annotations

import hashlib
import io
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
import streamlit as st

from src import analytics
from src.analytics.common import prepare
from src.db import connect, ingest

ROOT = Path(__file__).resolve().parent.parent.parent


def data_dir() -> Path:
    """App-owned DBs (sample + uploads). FINANCE_APP_DATA overrides it (tests use a temp dir)."""
    return Path(os.getenv("FINANCE_APP_DATA") or ROOT / "data" / "app")


def sample_db_path() -> Path:
    return data_dir() / "sample.db"


def db_key(path: str | Path) -> str:
    """Fingerprint of the transactions' state (rows + categories) and the cash entries. Changes on ingest,
    a category override or a cash entry, but not when a goal is saved, so the analytics cache and the
    chat memory survive that."""
    conn = sqlite3.connect(path)
    try:
        h = hashlib.sha1()
        for row in conn.execute("SELECT id, category, category_source FROM transactions ORDER BY id"):
            h.update(repr(row).encode())
        try:
            for row in conn.execute("SELECT id, date, amount, category, note FROM cash_entries ORDER BY id"):
                h.update(b"cash" + repr(row).encode())
        except sqlite3.OperationalError:  # a DB from before the cash wallet: no entries yet
            pass
    finally:
        conn.close()
    return h.hexdigest()[:16]


# ---- sample data --------------------------------------------------------------------------------

def ensure_sample_db() -> Path:
    """The synthetic statement, parsed, redacted, ingested and categorized once (rules; the offline fake
    model covers the one row no rule matches, so no real LLM is ever called). Later loads are instant."""
    path = sample_db_path()
    if path.exists():
        conn = sqlite3.connect(path)
        try:
            ready = conn.execute("SELECT COUNT(*) FROM transactions WHERE category IS NOT NULL").fetchone()[0] > 0
        except sqlite3.Error:
            ready = False
        finally:
            conn.close()
        if ready:
            return path
        path.unlink()
    from scripts.generate_data import build_statement, write_outputs
    from src.categorizer import categorize_transactions
    from src.llm.fake_provider import FakeProvider
    from src.parser import parse_statement

    raw_dir = data_dir() / "sample_raw"
    csv_path, _ = write_outputs(build_statement(seed=42), raw_dir, seed=42)
    tmp = path.with_suffix(".tmp.db")
    tmp.unlink(missing_ok=True)
    conn = connect(tmp)
    try:
        ingest(conn, parse_statement(csv_path), "sample statement.csv")
        categorize_transactions(conn, provider=FakeProvider())
    finally:
        conn.close()
    tmp.replace(path)
    return path


# ---- upload pipeline ----------------------------------------------------------------------------

class UploadError(ValueError):
    """A friendly, user-facing problem with an uploaded file."""


@dataclass
class UploadResult:
    db_path: Path
    rows_added: int
    rows_total: int
    needs_llm: int
    by_rule: int = 0
    by_llm: int = 0
    llm_error: str | None = None


def upload_db_path(content: bytes) -> Path:
    return data_dir() / f"upload_{hashlib.sha256(content).hexdigest()[:12]}.db"


def parse_upload(content: bytes):
    """Bytes -> normalized DataFrame, or UploadError with a plain-English reason."""
    from src.parser import StatementParseError, parse_statement
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise UploadError("The file isn't UTF-8 text. Export the statement as CSV and try again.") from None
    if not text.strip():
        raise UploadError("The file is empty.")
    try:
        df = parse_statement(io.StringIO(text))
    except StatementParseError as e:
        raise UploadError(f"We couldn't read this statement: {e}.") from None
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        raise UploadError(f"This doesn't look like a CSV file ({str(e).splitlines()[0]}).") from None
    if df.empty:
        raise UploadError("The statement has no transactions.")
    return df


def ingest_upload(content: bytes, filename: str, provider_name: str,
                  step: Callable[[float, str], None] = lambda p, m: None,
                  on_needs_llm: Callable[[int], None] = lambda n: None) -> UploadResult:
    """parse -> redact + ingest -> categorize (rules, then `provider_name` for the rest)."""
    from src.categorizer import categorize_transactions, rows_needing_llm
    from src.llm import LLMProviderError, get_provider

    step(0.1, "Reading the CSV...")
    df = parse_upload(content)
    step(0.35, "Masking personal details and saving locally...")
    path = upload_db_path(content)
    conn = connect(path)
    try:
        added = ingest(conn, df, filename)
        total = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
        needs = rows_needing_llm(conn)
        on_needs_llm(needs)
        step(0.55, f"Categorizing: rules first{f', then {needs} rows with the AI model' if needs else ''}...")
        result = UploadResult(path, added, total, needs)
        try:
            report = categorize_transactions(conn, provider=get_provider(provider_name) if needs else None)
            result.by_rule, result.by_llm = report.by_rule, report.by_llm
        except LLMProviderError as e:
            # Rule matches are lost with the failed run, so re-run rules only: the fake provider labels
            # the rest "Other" offline, and the user is told to retry with a working model.
            from src.llm.fake_provider import FakeProvider
            report = categorize_transactions(conn, provider=FakeProvider())
            result.by_rule, result.llm_error = report.by_rule, str(e)
    finally:
        conn.close()
    step(1.0, "Done")
    return result


def set_category(db_path: str | Path, txn_id: int, category: str, remember: bool = False) -> None:
    """Manual override (category_source='manual'); never overwritten by later categorizer runs.
    `remember=True` also saves a merchant rule: the merchant's other rows follow, and the LLM sees it."""
    from src.categorizer import set_manual_category
    conn = connect(db_path)
    try:
        set_manual_category(conn, int(txn_id), category, remember=remember)
    finally:
        conn.close()


# ---- cash wallet entries ------------------------------------------------------------------------

def add_cash(db_path: str | Path, amount: float, category: str, when, note: str = "") -> int:
    """Store one cash entry (the note is redacted before it is stored). Raises CashEntryError."""
    from src.cash_wallet import add_entry
    conn = connect(db_path)
    try:
        return add_entry(conn, amount, category, when, note)
    finally:
        conn.close()


def update_cash(db_path: str | Path, entry_id: int, amount: float, category: str, when, note: str = "") -> None:
    from src.cash_wallet import update_entry
    conn = connect(db_path)
    try:
        update_entry(conn, entry_id, amount, category, when, note)
    finally:
        conn.close()


def delete_cash(db_path: str | Path, entry_id: int) -> None:
    from src.cash_wallet import delete_entry
    conn = connect(db_path)
    try:
        delete_entry(conn, entry_id)
    finally:
        conn.close()


# ---- cached analytics reads ---------------------------------------------------------------------

@st.cache_data(show_spinner=False, max_entries=8)
def txns(path: str, key: str) -> pd.DataFrame:
    return prepare(db_path=path)


@st.cache_data(show_spinner=False, max_entries=8)
def months(path: str, key: str) -> list[str]:
    df = txns(path, key)
    return analytics.all_months(df) if not df.empty else []


@st.cache_data(show_spinner=False, max_entries=8)
def history(path: str, key: str):
    return analytics.month_history(txns(path, key), previous="data")


@st.cache_data(show_spinner=False, max_entries=8)
def anomaly_months(path: str, key: str) -> list[str]:
    return sorted({f"{a.date:%Y-%m}" for a in analytics.detect_anomalies(txns(path, key))})


@st.cache_data(show_spinner=False, max_entries=64)
def month_view(path: str, key: str, month: str) -> dict:
    """Everything the dashboard shows for one month, computed once per (DB state, month)."""
    df = txns(path, key)
    return {
        "mom": analytics.month_over_month(month, df, previous="data"),
        "insights": analytics.generate_insights(month, df),
        "merchants": analytics.top_merchants(month, 5, df),
        "daily": analytics.daily_spending(month, df),
    }


@st.cache_data(show_spinner=False, max_entries=8)
def cash_entries(path: str, key: str) -> pd.DataFrame:
    from src.analytics.cash import load_cash_entries
    return load_cash_entries(path)


@st.cache_data(show_spinner=False, max_entries=8)
def wallet_txns(path: str, key: str) -> pd.DataFrame:
    """The transactions as the cash-aware dashboard counts them (analytics.apply_cash)."""
    return analytics.apply_cash(txns(path, key), cash_entries(path, key))


@st.cache_data(show_spinner=False, max_entries=8)
def wallet_history(path: str, key: str):
    return analytics.month_history(wallet_txns(path, key), previous="data")


@st.cache_data(show_spinner=False, max_entries=64)
def wallet_month_view(path: str, key: str, month: str) -> dict:
    """The dashboard's month: totals, categories, merchants and days count logged and unaccounted cash
    (analytics.apply_cash); insights stay on the bank rows. With no entries every total equals month_view's."""
    df, wdf = txns(path, key), wallet_txns(path, key)
    return {
        "mom": analytics.month_over_month(month, wdf, previous="data"),
        "insights": analytics.generate_insights(month, df),
        "merchants": analytics.top_merchants(month, 5, wdf),
        "daily": analytics.daily_spending(month, wdf),
    }


@st.cache_data(show_spinner=False, max_entries=64)
def cash_view(path: str, key: str, month: str):
    """The month's CashSummary, or None when the data has no cash withdrawals (the card is hidden)."""
    df = txns(path, key)
    return analytics.cash_summary(month, df, cash_entries(path, key)) if analytics.has_cash_withdrawals(df) else None


@st.cache_data(show_spinner=False, max_entries=8)
def date_range(path: str, key: str):
    df = txns(path, key)
    return df["date"].min().date(), df["date"].max().date()


@st.cache_data(show_spinner=False, max_entries=64)
def ledger(path: str, key: str, month: str):
    """Bank rows plus the month's cash entries (source "cash")."""
    return analytics.month_transactions(month, txns=analytics.with_entries(txns(path, key), cash_entries(path, key)))


@st.cache_data(show_spinner=False, max_entries=8)
def base_plan(path: str, key: str):
    """Opportunities + projection with template sentences and no goal (no model call). The Plan tab
    applies the goal and the accepted set with advisor.reproject, which is pure arithmetic."""
    from src.advisor import build_plan
    return build_plan(txns(path, key), goal=None, provider=None)


@st.cache_data(show_spinner=False, max_entries=8)
def subscriptions(path: str, key: str):
    """Recurring charges found in the data (the Ask tab only suggests a subscription question if any)."""
    return analytics.detect_subscriptions(txns=txns(path, key))
