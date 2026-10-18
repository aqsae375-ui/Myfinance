"""Load a real bank statement into its OWN database (default data/real.db, gitignored), categorize it
(rules first, then LLM_PROVIDER for the rest) and print a privacy-safe report.

Only redacted descriptions are ever printed: raw descriptions stay in the DB's raw_description column.

Run:  $env:LLM_PROVIDER='fake'; python scripts/load_real_statement.py data/raw/statement_2025.csv
Options: --db data/real.db   --name "Account Holder" (repeatable; always masked)   --no-report
         --recategorize (redo every non-manual row, e.g. after the rules changed)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analytics import month_history  # noqa: E402
from src.analytics.common import prepare  # noqa: E402
from src.categorizer import REVIEW_BELOW, categorize_transactions, review_queue, rows_needing_llm  # noqa: E402
from src.db import connect, ingest  # noqa: E402
from src.llm import LLMProviderError  # noqa: E402
from src.parser import StatementParseError, parse_statement  # noqa: E402

DEFAULT_DB = ROOT / "data" / "real.db"


def _money(v: float | None) -> str:
    return "n/a" if v is None else f"{'-' if v < 0 else ''}Rs {abs(v):,.2f}"


def month_table(db: Path) -> str:
    rows = month_history(prepare(db_path=db))
    lines = [f"{'Month':<9}{'Spent':>16}{'Income':>16}{'Saved':>16}{'Rate':>11}{'Other in':>14}",
             "-" * 80]
    from src.analytics import monthly_summary
    df = prepare(db_path=db)
    for r in rows:
        other = monthly_summary(r.month, df).other_credits
        rate = "n/a" if r.savings_rate is None else f"{r.savings_rate:.1f}%"
        lines.append(f"{r.month:<9}{_money(r.total_spent):>16}{_money(r.total_income):>16}"
                     f"{_money(r.net_saved):>16}{rate:>11}{_money(other):>14}")
    lines.append("Income = Income + Interest + Transfers In credits; Spent = all debits; "
                 "'Other in' = other credits (refunds), not counted as income.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--name", action="append", default=[], help="known name to always redact (repeatable)")
    ap.add_argument("--no-report", action="store_true", help="skip the month-by-month table")
    ap.add_argument("--recategorize", action="store_true", help="redo every non-manual row")
    args = ap.parse_args(argv)

    try:
        df = parse_statement(args.csv)
    except (StatementParseError, FileNotFoundError) as e:
        print(f"Could not read {args.csv.name}: {e}")
        return 1

    conn = connect(args.db)
    try:
        added = ingest(conn, df, args.csv.name, args.name)
        pending = conn.execute("SELECT COUNT(*) FROM transactions WHERE category IS NULL "
                               "AND category_source IS NULL").fetchone()[0]
        needs_llm = rows_needing_llm(conn)
        print(f"Parsed {len(df)} rows from {args.csv.name}; {added} new -> {args.db}")
        print(f"Rules matched {pending - needs_llm} of {pending} new rows; {needs_llm} go to the LLM.")
        try:
            report = categorize_transactions(conn, recategorize=args.recategorize)
        except LLMProviderError as e:
            print(f"LLM unavailable ({e}); rows without a rule stay uncategorized. Retry later.")
            report = None
        if report and report.provider:
            print(f"LLM ({report.provider}): {report.by_llm} rows, {report.hinted} with a rule hint; "
                  f"{report.llm_unique_sent} merchants sent, {report.llm_cache_hits} from cache, "
                  f"{report.llm_invalid} invalid -> Unsure")

        totals = dict(conn.execute("SELECT COALESCE(category_source, 'none'), COUNT(*) FROM transactions "
                                   "GROUP BY 1").fetchall())
        total = sum(totals.values())
        print(f"\nWhole DB: {total} rows; by rule {totals.get('rule', 0)} ({totals.get('rule', 0) / total:.0%}), "
              f"by LLM {totals.get('llm', 0)}, your merchant rules {totals.get('learned', 0)}, "
              f"manual {totals.get('manual', 0)}, uncategorized {totals.get('none', 0)}")
        unmatched = conn.execute("SELECT redacted_description, category, confidence, COUNT(*) FROM transactions "
                                 "WHERE category_source IS NOT 'rule' GROUP BY 1, 2 ORDER BY 4 DESC, 1").fetchall()
        if unmatched:
            print("\nRows no confident rule decided (redacted descriptions only):")
            for desc, category, conf, n in unmatched:
                c = "" if conf is None else f" ({conf:.2f})"
                print(f"  {n:>3} x  {desc}  ->  {category or 'Unsure'}{c}")
        queue = review_queue(conn)
        print(f"\nReview queue: {len(queue)} rows (model Unsure, or confidence below {REVIEW_BELOW}):")
        for _, desc, amount, category, conf in queue:
            print(f"  {_money(amount):>14}  {desc}  ->  {category or 'Unsure'} ({conf or 0:.2f})")
        by_cat = conn.execute("SELECT category, COUNT(*), ROUND(SUM(amount), 2) FROM transactions "
                              "GROUP BY 1 ORDER BY 3").fetchall()
        print("\nBy category:")
        for cat, n, amount in by_cat:
            print(f"  {cat or 'uncategorized':<18}{n:>4} rows  {_money(amount):>16}")
    finally:
        conn.close()

    if not args.no_report:
        print("\n" + month_table(args.db))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
