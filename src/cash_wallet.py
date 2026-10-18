"""Cash wallet entries: what the user logged as spent in cash. Storage (the `cash_entries` table), the
quick-line parser ("chai 150") and the category proposal, which uses the existing keyword rules only.

Privacy: a note is redacted (src/redaction.py) BEFORE it is stored, so only masked text is ever kept.
Notes are never sent to a model: the agent's cash tool returns figures only, and no LLM is called here.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import date

import pandas as pd

from src.categories import CATEGORIES, INCOME_CATEGORIES, normalize_category
from src.redaction import redact
from src.rules import find_rule

CASH_CATEGORY = "Cash Withdrawal"
# Cash is spent on things: income categories and the withdrawal itself are not valid entry categories.
ENTRY_CATEGORIES: tuple[str, ...] = tuple(c for c in CATEGORIES if c not in INCOME_CATEGORIES | {CASH_CATEGORY})
ENTRY_COLUMNS = ["id", "date", "amount", "category", "note", "created_at"]
NOTE_MAX = 80
_AMOUNT_RE = re.compile(r"(?<![\w.])(?:rs\.?\s*|pkr\s*)?(\d[\d,]*(?:\.\d{1,2})?)(?![\w.])", re.I)


class CashEntryError(ValueError):
    """A user-facing problem with a cash entry."""


def _clean(amount: float, category: str, when: date | str, note: str) -> tuple[str, float, str, str]:
    try:
        amount = round(float(amount), 2)
    except (TypeError, ValueError):
        raise CashEntryError("Enter the amount as a number.") from None
    if not amount > 0:
        raise CashEntryError("The amount must be more than 0.")
    canon = normalize_category(category)
    if canon not in ENTRY_CATEGORIES:
        raise CashEntryError("Pick a spending category.")
    try:
        iso = pd.Timestamp(when).date().isoformat()
    except (TypeError, ValueError):
        raise CashEntryError("Enter a valid date.") from None
    return iso, amount, canon, redact(" ".join(str(note or "").split())[:NOTE_MAX])


def add_entry(conn: sqlite3.Connection, amount: float, category: str, when: date | str, note: str = "") -> int:
    iso, amount, category, note = _clean(amount, category, when, note)
    with conn:
        cur = conn.execute("INSERT INTO cash_entries (date, amount, category, note) VALUES (?, ?, ?, ?)",
                           (iso, amount, category, note))
    return int(cur.lastrowid)


def update_entry(conn: sqlite3.Connection, entry_id: int, amount: float, category: str, when: date | str,
                 note: str = "") -> None:
    iso, amount, category, note = _clean(amount, category, when, note)
    with conn:
        conn.execute("UPDATE cash_entries SET date = ?, amount = ?, category = ?, note = ? WHERE id = ?",
                     (iso, amount, category, note, int(entry_id)))


def delete_entry(conn: sqlite3.Connection, entry_id: int) -> None:
    with conn:
        conn.execute("DELETE FROM cash_entries WHERE id = ?", (int(entry_id),))


def load_entries(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(f"SELECT {', '.join(ENTRY_COLUMNS)} FROM cash_entries ORDER BY date, id", conn)


# ---- quick line -----------------------------------------------------------------------------------

@dataclass(frozen=True)
class QuickLine:
    amount: float
    note: str                     # redacted
    category: str | None          # proposed by a keyword rule, or None: the user picks
    rule: str | None = None       # the rule that proposed it


def propose_category(note: str) -> tuple[str, str] | None:
    """(category, rule name) from the existing keyword rules, or None. A cash note reads like a shop
    purchase, so it is also tried as "POS PURCHASE <note>" (the rules' shop-name lists need that prefix)."""
    text = redact(note).upper()
    for candidate in (text, f"POS PURCHASE {text}"):
        match = find_rule(candidate, -1.0)
        if match and match.category in ENTRY_CATEGORIES:
            return match.category, match.rule
    return None


def parse_quick_line(line: str) -> QuickLine:
    """"chai 150" / "Rs 1,200 petrol" -> amount + note + proposed category. The last number is the amount."""
    text = " ".join(str(line or "").split())
    found = list(_AMOUNT_RE.finditer(text))
    if not found:
        raise CashEntryError('Type what you bought and the amount, for example "chai 150".')
    last = found[-1]
    amount = round(float(last.group(1).replace(",", "")), 2)
    if not amount > 0:
        raise CashEntryError("The amount must be more than 0.")
    note = redact((text[:last.start()] + " " + text[last.end():]).strip(" -,:"))[:NOTE_MAX]
    note = " ".join(note.split())
    proposal = propose_category(note) if note else None
    return QuickLine(amount, note, *(proposal or (None, None)))
