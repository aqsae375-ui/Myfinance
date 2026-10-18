"""Print the full analytics report for one month, for checking the numbers by eye.

Run:  python scripts/print_month_report.py [2026-05] [--db data/finance.db]
      (no month = the latest month in the data)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analytics import (  # noqa: E402
    all_months, detect_anomalies, detect_subscriptions, generate_insights, month_over_month,
)
from src.analytics.common import format_money, month_end, prepare  # noqa: E402
from src.analytics.schema import Change  # noqa: E402
from src.db import DEFAULT_DB_PATH  # noqa: E402


def _change(ch: Change) -> str:
    if ch.abs_change is None:
        return "(no previous month)"
    pct = "n/a" if ch.pct_change is None else f"{ch.pct_change:+.1f}%"
    return f"{ch.abs_change:+,.2f} ({pct})"


def report(month: str | None, db_path: Path) -> str:
    df = prepare(db_path=db_path)
    if df["category"].eq("Uncategorized").all():
        return "No categorized transactions: run `python -m src.categorizer` first."
    months = all_months(df)
    month = month or months[-1]
    mom = month_over_month(month, df)
    cur = mom.current
    lines = [f"=== Report for {month}  (data covers {months[0]} .. {months[-1]}) ===",
             f"Compared with: {mom.previous_month or 'nothing (first month in data)'}", ""]

    lines.append(f"{'':<14}{'this month':>14}  {'change vs last':>24}")
    for label, attr in (("Spent", "total_spent"), ("Income", "total_income"), ("Net saved", "net_saved")):
        ch = getattr(mom, attr)
        lines.append(f"{label:<14}{ch.current:>14,.2f}  {_change(ch):>24}")
    lines += ["", f"{'Category':<16}{'spent':>10}{'share':>8}{'txns':>6}  {'change vs last':>24}"]
    for c in mom.categories:
        total = cur.category(c.category)
        lines.append(f"{c.category:<16}{c.amount.current:>10,.2f}{c.pct_of_spend.current:>7.1f}%"
                     f"{total.txn_count if total else 0:>6}  {_change(c.amount):>24}")
    lines.append(f"Method: {cur.method}")

    as_of = min(month_end(month), df["date"].max().date())
    lines += ["", f"Subscriptions (as of {as_of}):"]
    for s in detect_subscriptions(df, as_of=as_of):
        flag = "POSSIBLY UNUSED" if s.possibly_unused else "ok"
        lines.append(f"  {s.merchant:<24}{format_money(s.monthly_amount):>11}/mo {format_money(s.annual_cost):>9}/yr "
                     f"x{s.charge_count}  [{flag}] {s.reason}")

    lines += ["", "Anomalies this month:"]
    anomalies = detect_anomalies(df, month=month)
    lines += [f"  {a.date}  {a.merchant:<22}{format_money(a.amount):>10}  {'+'.join(a.reasons)}: {a.explanation}"
              for a in anomalies] or ["  none"]

    lines += ["", "Insights (ranked by rupee impact):"]
    insights = generate_insights(month, df)
    lines += [f"  {i}. {ins.text}   [impact {format_money(ins.impact)}, {ins.kind}]"
              for i, ins in enumerate(insights, 1)] or ["  none"]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("month", nargs="?", help="YYYY-MM (default: latest month)")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    args = ap.parse_args()
    print(report(args.month, args.db))


if __name__ == "__main__":
    main()
