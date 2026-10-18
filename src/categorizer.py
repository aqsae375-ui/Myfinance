"""Categorization pipeline (Stage 2d). Priority, first one wins:

1. manual single-row override (category_source='manual'; never touched by a run)
2. learned merchant rule (the user's "always put this merchant in X", table merchant_rules)
3. keyword rule with confidence >= RULE_THRESHOLD (0.8)
4. LLM answer cached by merchant key
5. LLM. A low-confidence (ambiguous) rule match goes to the model as a hint only ("a rule suggested X").

- Only `redacted_description` is ever read for categorization; `raw_description` never leaves SQLite.
- The model may answer "Unsure". Unsure (or an off-taxonomy answer) leaves category NULL, and Unsure or a
  confidence below REVIEW_BELOW sets needs_review=1: the review queue. Nothing is silently guessed.
- Rows whose text looks like an instruction (prompt injection) are never sent: they go straight to review.
- The user's newest saved merchant rules (up to MAX_EXAMPLES) go into the prompt as examples.

Run:  python -m src.categorizer [--db data/finance.db] [--recategorize]
"""
from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from src.analytics.common import merchant_key
from src.categories import normalize_category
from src.db import DEFAULT_DB_PATH, connect
from src.llm import LLMProvider, get_provider
from src.llm.base import UNSURE, LLMItem, looks_like_instruction
from src.rules import find_rule

REVIEW_BELOW = 0.6   # an LLM answer below this confidence goes to the review queue
MAX_EXAMPLES = 20    # saved merchant rules shown to the LLM, newest first
UNKNOWN_MERCHANT = "UNKNOWN"


@dataclass
class CategorizationReport:
    rows: int = 0             # rows (re)categorized in this run
    by_learned: int = 0       # rows decided by a saved merchant rule
    by_rule: int = 0          # rows decided by a keyword rule (confidence >= threshold)
    by_llm: int = 0           # rows categorized from an LLM answer (fresh or cached)
    hinted: int = 0           # LLM rows that carried a low-confidence rule hint
    llm_unique_sent: int = 0  # distinct merchant keys actually sent to the provider
    llm_cache_hits: int = 0   # distinct merchant keys answered from the cache
    llm_invalid: int = 0      # distinct merchant keys whose answer was invalid -> Unsure
    unsure: int = 0           # rows left uncategorized because the model was unsure (or injected text)
    needs_review: int = 0     # rows put in the review queue in this run
    manual_kept: int = 0      # manual rows left untouched
    provider: str | None = None
    row_sources: dict[int, str] = field(default_factory=dict)


def _direction(amount: float) -> str:
    return "in" if amount >= 0 else "out"


def cache_key(description: str, amount: float, hint: str | None = None) -> str:
    """One LLM answer per merchant and money direction (a wallet name in vs out means different
    categories). Descriptions with no merchant word left fall back to the whole redacted text. The rule
    hint is part of the key (a merchant always gets the same hint), so a changed rule is asked again."""
    key = merchant_key(description)
    base = f"{_direction(amount)}|{description if key == UNKNOWN_MERCHANT else key}"
    return f"{base}|hint={hint}" if hint else base


def learned_rules(conn: sqlite3.Connection) -> dict[str, str]:
    return dict(conn.execute("SELECT merchant_key, category FROM merchant_rules"))


def learned_examples(conn: sqlite3.Connection, limit: int = MAX_EXAMPLES) -> list[tuple[str, str]]:
    """The user's saved (merchant key, category) pairs, most recent first."""
    return [tuple(r) for r in conn.execute(
        "SELECT merchant_key, category FROM merchant_rules ORDER BY updated_at DESC, rowid DESC LIMIT ?", (limit,))]


def _cache_lookup(conn: sqlite3.Connection, provider: str, keys: list[str]) -> dict[str, tuple[str, bool, float]]:
    found: dict[str, tuple[str, bool, float]] = {}
    for k in keys:
        row = conn.execute("SELECT category, valid, confidence FROM llm_category_cache "
                           "WHERE description = ? AND provider = ?", (k, provider)).fetchone()
        if row:
            found[k] = (row[0], bool(row[1]), float(row[2] or 0.0))
    return found


def categorize_transactions(conn: sqlite3.Connection, provider: LLMProvider | None = None,
                            recategorize: bool = False) -> CategorizationReport:
    """Categorize uncategorized rows (or, with recategorize=True, every non-manual row).
    `provider` is built from LLM_PROVIDER only if some row actually needs the LLM."""
    report = CategorizationReport()
    report.manual_kept = conn.execute(
        "SELECT COUNT(*) FROM transactions WHERE category_source = 'manual'").fetchone()[0]

    where = "category_source IS NOT 'manual'" if recategorize else "category IS NULL AND category_source IS NULL"
    rows = conn.execute(f"SELECT id, redacted_description, amount FROM transactions WHERE {where} ORDER BY id").fetchall()
    report.rows = len(rows)
    learned = learned_rules(conn)

    updates: list[tuple[str | None, str, float, int, int]] = []  # (category, source, confidence, review, id)
    pending: list[tuple[int, str, float, str | None]] = []        # (id, description, amount, hint)
    for txn_id, desc, amount in rows:
        if (cat := learned.get(merchant_key(desc))) is not None:
            updates.append((cat, "learned", 1.0, 0, txn_id))
            report.by_learned += 1
            continue
        if looks_like_instruction(desc):
            # Checked before keyword rules, so a rule word ("IBFT") can't keep it out of the review queue.
            updates.append((None, "llm", 0.0, 1, txn_id))  # never sent: injected text cannot steer the model
            report.unsure += 1
            continue
        hit = find_rule(desc, amount)
        if hit and hit.decides:
            updates.append((hit.category, "rule", hit.confidence, 0, txn_id))
            report.by_rule += 1
        else:
            pending.append((txn_id, desc, amount, hit.category if hit else None))

    if pending:
        provider = provider or get_provider()
        report.provider = provider.name
        first: dict[str, LLMItem] = {}  # cache key -> the item sent for it (its first row)
        for _, desc, amount, hint in pending:
            first.setdefault(cache_key(desc, amount, hint), LLMItem(desc, _direction(amount), hint))
        answers = _cache_lookup(conn, provider.name, list(first))
        report.llm_cache_hits = len(answers)

        to_send = [k for k in first if k not in answers]
        if to_send:
            results = provider.categorize_batch_detailed([first[k] for k in to_send], learned_examples(conn))
            if len(results) != len(to_send):
                raise RuntimeError(f"Provider returned {len(results)} results for {len(to_send)} descriptions")
            fresh = {k: (r.category, r.valid, r.confidence) for k, r in zip(to_send, results)}
            with conn:
                conn.executemany(
                    "INSERT OR REPLACE INTO llm_category_cache (description, provider, category, valid, confidence) "
                    "VALUES (?, ?, ?, ?, ?)", [(k, provider.name, c, int(v), conf) for k, (c, v, conf) in fresh.items()])
            answers.update(fresh)
        report.llm_unique_sent = len(to_send)
        report.llm_invalid = sum(not valid for _, valid, _ in answers.values())

        for txn_id, desc, amount, hint in pending:
            category, valid, conf = answers[cache_key(desc, amount, hint)]
            unsure = category == UNSURE or not valid
            review = int(unsure or conf < REVIEW_BELOW)
            updates.append((None if unsure else category, "llm", conf if valid else 0.0, review, txn_id))
            report.unsure += unsure
            report.hinted += hint is not None
        report.by_llm = len(pending)

    report.needs_review = sum(u[3] for u in updates)
    with conn:
        # The source guard keeps a manual override made mid-run from being overwritten.
        conn.executemany(
            "UPDATE transactions SET category = ?, category_source = ?, confidence = ?, needs_review = ? "
            "WHERE id = ? AND category_source IS NOT 'manual'", updates)
    report.row_sources = {txn_id: source for _, source, _, _, txn_id in updates}
    return report


def rows_needing_llm(conn: sqlite3.Connection) -> int:
    """Uncategorized rows that no saved merchant rule or deciding keyword rule covers, i.e. rows the
    next run will send to the LLM (before the cache is checked). Lets the UI warn it may take a while."""
    learned = learned_rules(conn)
    rows = conn.execute("SELECT redacted_description, amount FROM transactions "
                        "WHERE category IS NULL AND category_source IS NULL").fetchall()
    return sum(merchant_key(desc) not in learned and not looks_like_instruction(desc)
               and not ((hit := find_rule(desc, amount)) and hit.decides) for desc, amount in rows)


def set_manual_category(conn: sqlite3.Connection, txn_id: int, category: str, remember: bool = False) -> None:
    """User override. Sets category_source='manual'; later categorizer runs never change it.
    `remember=True` also saves a merchant rule, so the user's other rows from this merchant follow."""
    canonical = normalize_category(category)
    if canonical is None:
        raise ValueError(f"{category!r} is not a valid category")
    with conn:
        cur = conn.execute(
            "UPDATE transactions SET category = ?, category_source = 'manual', confidence = 1.0, needs_review = 0 "
            "WHERE id = ?", (canonical, txn_id))
    if cur.rowcount == 0:
        raise KeyError(f"No transaction with id {txn_id}")
    if remember:
        save_merchant_rule(conn, txn_id, canonical)


def save_merchant_rule(conn: sqlite3.Connection, txn_id: int, category: str) -> str:
    """Save "this row's merchant -> category" and apply it now to every non-manual row of that
    merchant. Returns the merchant key (from the redacted description)."""
    canonical = normalize_category(category)
    if canonical is None:
        raise ValueError(f"{category!r} is not a valid category")
    row = conn.execute("SELECT redacted_description FROM transactions WHERE id = ?", (txn_id,)).fetchone()
    if row is None:
        raise KeyError(f"No transaction with id {txn_id}")
    key = merchant_key(row[0])
    if key == UNKNOWN_MERCHANT:
        raise ValueError("This description has no merchant name to remember")
    ids = [i for i, desc in conn.execute("SELECT id, redacted_description FROM transactions "
                                         "WHERE category_source IS NOT 'manual'") if merchant_key(desc) == key]
    with conn:
        conn.execute("INSERT INTO merchant_rules (merchant_key, category, updated_at) VALUES (?, ?, datetime('now')) "
                     "ON CONFLICT(merchant_key) DO UPDATE SET category = excluded.category, "
                     "updated_at = excluded.updated_at", (key, canonical))
        conn.executemany("UPDATE transactions SET category = ?, category_source = 'learned', confidence = 1.0, "
                         "needs_review = 0 WHERE id = ?", [(canonical, i) for i in ids])
    return key


def review_queue(conn: sqlite3.Connection) -> list[tuple[int, str, float, str | None, float | None]]:
    """Rows waiting for the user: (id, redacted description, amount, category or None, confidence)."""
    return conn.execute("SELECT id, redacted_description, amount, category, confidence FROM transactions "
                        "WHERE needs_review = 1 ORDER BY date, id").fetchall()


def main() -> None:
    ap = argparse.ArgumentParser(description="Categorize transactions in the SQLite DB.")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    ap.add_argument("--recategorize", action="store_true", help="Redo every non-manual row")
    args = ap.parse_args()

    conn = connect(args.db)
    try:
        r = categorize_transactions(conn, recategorize=args.recategorize)
    finally:
        conn.close()
    print(f"Categorized {r.rows} rows: {r.by_learned} by saved merchant rules, {r.by_rule} by rules, "
          f"{r.by_llm} by LLM ({r.provider or 'not needed'}; {r.hinted} with a rule hint); "
          f"{r.llm_unique_sent} merchants sent, {r.llm_cache_hits} from cache, {r.llm_invalid} invalid; "
          f"{r.unsure} unsure, {r.needs_review} need review; {r.manual_kept} manual rows kept.")


if __name__ == "__main__":
    main()
