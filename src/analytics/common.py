"""Shared helpers: loading transactions, month arithmetic, merchant keys, money formatting."""
from __future__ import annotations

import os
import re
from datetime import date
from pathlib import Path

import pandas as pd

from src.db import DEFAULT_DB_PATH, connect, load_transactions

# Currency shown with every amount: a setting (FINANCE_CURRENCY in .env), "Rs" by default.
CURRENCY = (os.getenv("FINANCE_CURRENCY") or "Rs").strip()
UNCATEGORIZED = "Uncategorized"
MONTH_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
_PLACEHOLDER_RE = re.compile(r"\[(?:EMAIL|PHONE|ACCOUNT|CARD|NAME|REF|ADDRESS)\]")
# Transaction-type noise in Pakistani exports: "INTL. MASTER POS Purchase X STAN [REF]" -> "X".
_POS_PREFIX_RE = re.compile(r"^\s*(?:INTL\.?\s+MASTER\s+)?POS\s+PURCHASE\s+", re.IGNORECASE)
_NOISE_RE = re.compile(r"\bSTAN\b", re.IGNORECASE)
MERCHANT_MAX_TOKENS = 6


def prepare(txns: pd.DataFrame | None = None, db_path: str | Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Return a copy of the transactions with `month` and `merchant` columns added.
    `txns=None` loads them from the SQLite DB. Safe to call on an already-prepared frame."""
    if txns is None:
        conn = connect(db_path)
        try:
            txns = load_transactions(conn)
        finally:
            conn.close()
    df = txns.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["category"] = df["category"].fillna(UNCATEGORIZED)
    df["month"] = df["date"].dt.strftime("%Y-%m")
    df["merchant"] = df["redacted_description"].map(merchant_key)
    return df.sort_values(["date", "id"]).reset_index(drop=True)


def check_month(month: str) -> str:
    if not isinstance(month, str) or not MONTH_RE.match(month):
        raise ValueError(f"month must be 'YYYY-MM', got {month!r}")
    return month


def previous_month(month: str) -> str:
    return (pd.Period(check_month(month), freq="M") - 1).strftime("%Y-%m")


def month_end(month: str) -> date:
    return pd.Period(check_month(month), freq="M").end_time.date()


def merchant_key(description: str) -> str:
    """Stable merchant label from a *redacted* description: PII placeholders and tokens containing
    digits (store numbers, reference codes) are dropped, so "AMZN MKTP US*2K4L81" -> "AMZN MKTP US".
    A "POS Purchase" prefix and STAN codes are dropped too: "POS PURCHASE CHEEZIOUS STAN [REF]" -> "CHEEZIOUS"."""
    text = _POS_PREFIX_RE.sub("", str(description).upper())
    text = _NOISE_RE.sub(" ", _PLACEHOLDER_RE.sub(" ", text)).replace("*", " ")
    tokens = [t for t in text.split() if not any(ch.isdigit() for ch in t)]
    return " ".join(tokens[:MERCHANT_MAX_TOKENS]) or "UNKNOWN"


def format_money(value: float) -> str:
    """'Rs 4,200' for amounts of 100 or more, 'Rs 44.99' below that. Always unsigned."""
    value = abs(value)
    return f"{CURRENCY} {value:,.0f}" if value >= 100 else f"{CURRENCY} {value:,.2f}"


def format_pct(value: float) -> str:
    return f"{abs(value):.0f}%"
