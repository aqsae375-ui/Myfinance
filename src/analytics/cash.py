"""Cash wallet: withdrawals fill the wallet, logged cash entries say where it went. Pure Pandas; no model.

Definitions (per month; also in `CashSummary.method`):
- withdrawn   = Cash Withdrawal debits in the month (money moved into the cash wallet)
- logged      = cash entries the user logged with a date in the month
- unaccounted = max(0, withdrawn - logged)
- outside     = max(0, logged - withdrawn): spent from cash that did not come from this account that month
                ("cash from outside the account"); it counts as spending
- accounted_pct = min(logged, withdrawn) / withdrawn * 100 (None when nothing was withdrawn)
- wallet_balance = cumulative withdrawn - cumulative logged up to the month's end (information only)
- Spent = non-cash debits + logged + unaccounted

`apply_cash` turns this into a transactions frame, so monthly.py / dashboard.py compute every figure from
it unchanged: each month's withdrawal rows are reduced by that month's logged cash (earliest first) and
relabelled UNACCOUNTED, and each entry becomes a debit row in its own category (id = -entry id,
category_source = "cash"). With no entries the totals are exactly those of the plain transactions.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analytics.common import check_month, prepare
from src.analytics.schema import CashSummary
from src.db import DEFAULT_DB_PATH, connect

CASH = "Cash Withdrawal"
UNACCOUNTED = "Unaccounted cash"
CASH_SOURCE = "cash"
OUTSIDE_FLAG = "cash from outside the account"
METHOD = ("withdrawn = Cash Withdrawal debits; logged = your cash entries; unaccounted = max(0, withdrawn - logged); "
          "logged above withdrawn counts as spending (cash from outside the account); "
          "accounted = min(logged, withdrawn) / withdrawn; wallet balance = all withdrawn - all logged so far")


def _r(x: float) -> float:
    return round(float(x), 2)


def load_cash_entries(db_path: str | Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    from src.cash_wallet import load_entries
    conn = connect(db_path)
    try:
        return load_entries(conn)
    finally:
        conn.close()


def prepare_entries(entries: pd.DataFrame | None = None, db_path: str | Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Entries with `date` as datetime and a `month` column. `None` loads them from the DB."""
    from src.cash_wallet import ENTRY_COLUMNS
    if entries is None:
        entries = load_cash_entries(db_path)
    df = (entries.copy() if entries is not None and len(entries) else pd.DataFrame(columns=ENTRY_COLUMNS))
    df["date"] = pd.to_datetime(df["date"])
    df["amount"] = df["amount"].astype(float)
    df["note"] = df["note"].fillna("").astype(str) if "note" in df else ""
    df["month"] = df["date"].dt.strftime("%Y-%m")
    return df.sort_values(["date", "id"]).reset_index(drop=True)


def _both(txns: pd.DataFrame | None, entries: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Like every analytics function, `txns=None` reads the DB, and then so do the entries. Passed
    transactions with no entries mean "no cash logged", never entries from some other DB."""
    if entries is None and txns is not None:
        entries = pd.DataFrame(columns=["id", "date", "amount", "category", "note"])
    return prepare(txns), prepare_entries(entries)


def _withdrawals(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["category"] == CASH) & (df["amount"] < 0)]


def has_cash_withdrawals(txns: pd.DataFrame | None = None) -> bool:
    return not _withdrawals(prepare(txns)).empty


def cash_summary(month: str, txns: pd.DataFrame | None = None, entries: pd.DataFrame | None = None) -> CashSummary:
    """The cash wallet for one month (see the module docstring for every formula)."""
    df, ent = _both(txns, entries)
    month = check_month(month)
    wd = _withdrawals(df)
    month_wd, month_ent = wd[wd["month"] == month], ent[ent["month"] == month]
    withdrawn, logged = _r(-month_wd["amount"].sum()), _r(month_ent["amount"].sum())
    balance = _r(-wd.loc[wd["month"] <= month, "amount"].sum() - ent.loc[ent["month"] <= month, "amount"].sum())
    return CashSummary(
        month=month, withdrawn=withdrawn, logged=logged, unaccounted=_r(max(0.0, withdrawn - logged)),
        outside=_r(max(0.0, logged - withdrawn)),
        accounted_pct=_r(min(logged, withdrawn) / withdrawn * 100) if withdrawn else None,
        wallet_balance=balance, method=METHOD,
        txn_ids=tuple(int(i) for i in month_wd["id"]), entry_ids=tuple(int(i) for i in month_ent["id"]))


def _entry_rows(ent: pd.DataFrame) -> pd.DataFrame:
    """Entries as debit transaction rows: id = -entry id, source "cash", description "CASH <note>"."""
    text = [f"CASH {n}".strip() if n else f"CASH {c}" for n, c in zip(ent["note"], ent["category"])]
    return pd.DataFrame({
        "id": -ent["id"].astype(int), "date": ent["date"], "raw_description": text, "redacted_description": text,
        "amount": -ent["amount"].astype(float), "type": "debit", "category": ent["category"],
        "category_source": CASH_SOURCE, "confidence": 1.0, "needs_review": 0, "source_file": "cash wallet",
    })


def with_entries(txns: pd.DataFrame | None = None, entries: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every bank row unchanged plus one row per cash entry (the Transactions table)."""
    df, ent = _both(txns, entries)
    if ent.empty:
        return df
    return prepare(pd.concat([df, _entry_rows(ent)], ignore_index=True))


def apply_cash(txns: pd.DataFrame | None = None, entries: pd.DataFrame | None = None) -> pd.DataFrame:
    """The transactions as the cash-aware dashboard counts them (see the module docstring)."""
    df, ent = _both(txns, entries)
    logged = ent.groupby("month")["amount"].sum().to_dict()
    df = df.copy()
    drop = []
    for month, rows in _withdrawals(df).groupby("month"):
        left = float(logged.get(month, 0.0))
        for idx in rows.sort_values(["date", "id"]).index:
            out = -float(df.at[idx, "amount"])
            take = min(out, left)
            left -= take
            remaining = _r(out - take)
            if remaining > 0:
                df.at[idx, "amount"] = -remaining
                df.at[idx, "category"] = UNACCOUNTED
            else:
                drop.append(idx)
    df = df.drop(index=drop)
    if not ent.empty:
        df = pd.concat([df, _entry_rows(ent)], ignore_index=True)
    return prepare(df)


def cash_history(txns: pd.DataFrame | None = None, entries: pd.DataFrame | None = None) -> list[CashSummary]:
    """One CashSummary per month that has transactions or entries, oldest first."""
    df, ent = _both(txns, entries)
    months = sorted(set(df["month"]) | set(ent["month"]))
    return [cash_summary(m, df, ent) for m in months]
