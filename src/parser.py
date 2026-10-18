"""Parse a bank-statement CSV into a normalized DataFrame.

Output columns (file order preserved):
  date         datetime64 (day precision)
  description  str, whitespace-collapsed
  amount       float, signed: negative = money out
  type         'debit' (amount < 0) | 'credit'
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import IO

import numpy as np
import pandas as pd

# Each value is matched by shape first, so "05/03/26" can never be read as the year 26 AD.
ISO_DATE_RE = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}$")
SLASH_DATE_RE = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{2}|\d{4})$")
OUTPUT_COLUMNS = ["date", "description", "amount", "type"]

# Aliases in priority order: when a file has several (Date + Value Date), the first one found wins and
# the others are ignored like any unknown column (e.g. Balance). Analytics use the posting date.
COLUMN_ALIASES = {
    "date": ("date", "transaction date", "posting date", "posted date", "txn date", "value date"),
    "description": ("description", "details", "memo", "narrative", "payee", "transaction details", "particulars"),
    "amount": ("amount", "transaction amount", "amount (pkr)", "amount (rs)"),
    "debit": ("debit", "withdrawal", "withdrawals", "money out", "debit (pkr)"),
    "credit": ("credit", "deposit", "deposits", "money in", "credit (pkr)"),
}

_NUMBER_RE = re.compile(r"^\d+(?:\.\d+)?$")
_STRIP_RE = re.compile(r"[\s,$€£]|^(?:PKR|RS\.?)|(?:PKR|RS\.?)$", re.IGNORECASE)


class StatementParseError(ValueError):
    """The CSV is missing required columns or has unparseable rows."""


def parse_statement(source: str | Path | IO[str]) -> pd.DataFrame:
    raw = pd.read_csv(source, dtype=str, keep_default_na=False, skipinitialspace=True)
    return normalize(raw)


def normalize(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(columns=_canonical_columns(raw.columns))
    df = df[~(df.apply(lambda col: col.str.strip()) == "").all(axis=1)]  # drop blank rows

    missing = {"date", "description"} - set(df.columns)
    has_amount = "amount" in df.columns or {"debit", "credit"} <= set(df.columns)
    if missing or not has_amount:
        need = sorted(missing | (set() if has_amount else {"amount (or debit + credit)"}))
        raise StatementParseError(f"Missing column(s): {', '.join(need)}. Found: {list(raw.columns)}")

    dates = parse_dates(df["date"])
    if "amount" in df.columns:
        amounts = df["amount"].map(parse_amount)
    else:  # debit/credit columns: fill blanks with 0 so a row with only one side still parses
        debit = df["debit"].map(lambda v: parse_amount(v) if str(v).strip() else 0.0).abs()
        credit = df["credit"].map(lambda v: parse_amount(v) if str(v).strip() else 0.0).abs()
        amounts = credit - debit

    bad = dates.isna() | amounts.isna()
    if bad.any():
        lines = [str(i + 2) for i in df.index[bad]]  # +2: header row and 1-based lines
        raise StatementParseError(f"Unparseable date or amount on line {', line '.join(lines)}")

    out = pd.DataFrame({
        "date": dates.dt.normalize(),
        "description": df["description"].str.split().str.join(" "),
        "amount": amounts.astype(float).round(2),
    })
    out["type"] = np.where(out["amount"] < 0, "debit", "credit")
    return out.reset_index(drop=True)[OUTPUT_COLUMNS]


def parse_dates(values: pd.Series) -> pd.Series:
    """ISO (YYYY-MM-DD) and slash dates (D/M/YYYY, D/M/YY, or month-first when the file is month-first);
    anything else becomes NaT. The slash order is decided once per file by slash_order()."""
    values = values.astype(str).str.strip()
    result = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    iso = values.str.match(ISO_DATE_RE)
    if iso.any():
        result[iso] = pd.to_datetime(values[iso], format="%Y-%m-%d", errors="coerce")
    parts = values.str.extract(SLASH_DATE_RE)
    slash = parts[0].notna()
    if slash.any():
        p = parts[slash].astype(int)
        day, month = (p[0], p[1]) if slash_order(p[0], p[1]) == "day_first" else (p[1], p[0])
        year = p[2].where(p[2] >= 100, 2000 + p[2])
        result[slash] = pd.to_datetime(pd.DataFrame({"year": year, "month": month, "day": day}), errors="coerce")
    return result


def slash_order(first: pd.Series, second: pd.Series) -> str:
    """'day_first' or 'month_first' for a whole file's slash dates, never guessed per row:
    a first part > 12 proves day-first, a second part > 12 proves month-first (both at once is an error).
    If no value decides it, day-first (the Pakistani/UK convention) unless only month-first puts the rows
    in date order, since statements are listed chronologically."""
    # A value with both parts > 12 is invalid either way: it becomes NaT and is reported by line number.
    day_first = ((first > 12) & (second <= 12)).any()
    month_first = ((second > 12) & (first <= 12)).any()
    if day_first and month_first:
        raise StatementParseError("Dates mix day-first and month-first order (e.g. 25/01 and 01/25); "
                                  "export the statement with one date format.")
    if day_first or month_first:
        return "day_first" if day_first else "month_first"
    as_day_first = list(zip(second, first))
    as_month_first = list(zip(first, second))
    if _monotonic(as_month_first) and not _monotonic(as_day_first):
        return "month_first"
    return "day_first"


def _monotonic(keys: list) -> bool:
    return keys == sorted(keys) or keys == sorted(keys, reverse=True)


def parse_amount(value: object) -> float:
    """'-1,650.00' -> -1650.0, '(12.50)' -> -12.5, '$9.99' -> 9.99. Returns NaN if unparseable."""
    s = str(value).strip()
    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative, s = True, s[1:-1]
    s = _STRIP_RE.sub("", s)
    if s and s[0] in "+-":
        negative ^= s[0] == "-"
        s = s[1:]
    if not _NUMBER_RE.match(s):
        return float("nan")
    return -float(s) if negative else float(s)


def _canonical_columns(columns) -> dict[str, str]:
    """{original column: canonical name}, at most one column per canonical name (by alias priority)."""
    keys = {str(col).strip().lower(): col for col in columns}
    mapping = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        found = next((keys[a] for a in aliases if a in keys), None)
        if found is not None:
            mapping[found] = canonical
    return mapping
