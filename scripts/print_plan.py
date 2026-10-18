"""Print the full action plan: ranked ways to save, how each number was computed, and the
before/after outcome toward a goal.

Run:  python scripts/print_plan.py                                  # DB goal, else the sample goal
      python scripts/print_plan.py --target 10000 --by 2027-03-31 [--saved 2000] [--save-goal]
      python scripts/print_plan.py --monthly 1500
      python scripts/print_plan.py --no-goal --accept "cancel:FITPRO GYM MEMBERSHIP"
Options: --provider fake|ollama (default fake: offline), --db data/finance.db
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.advisor import Goal, build_plan, load_goal, outcome_sentences, save_goal  # noqa: E402
from src.advisor.opportunities import money  # noqa: E402
from src.analytics.common import prepare  # noqa: E402
from src.db import DEFAULT_DB_PATH, connect  # noqa: E402
from src.llm import get_provider  # noqa: E402

SAMPLE_GOAL = Goal.by_date(10000, "2027-03-31", name="Emergency fund")


def _money_or_na(v: float | None) -> str:
    return "n/a" if v is None else (money(v) if v >= 0 else "-" + money(v))


def _months(n: int | None, on_track: bool | None) -> str:
    if n is None:
        return "never"
    return f"{n} mo" + ("" if on_track is None else (" (on track)" if on_track else " (late)"))


def render(plan) -> str:
    p = plan.projection
    lines = [f"=== Your plan  (history {plan.months[0]} .. {plan.months[-1]}, as of {plan.as_of}) ===",
             f"Goal: {plan.goal.describe() if plan.goal else 'none set'}", ""]
    if not plan.items:
        lines.append("No savings opportunities found yet (more history may reveal some).")
    else:
        lines.append("Top ways to save (ranked by estimated monthly saving):")
    for item in plan.items:
        o = item.opportunity
        mark = "x" if item.accepted else " "
        lines += [f" [{mark}] {o.rank}. {o.action}",
                  f"        saves {money(o.monthly_saving)}/month ({money(o.annual_saving)}/year), "
                  f"{o.confidence} confidence   id: {o.id}",
                  f"        {item.explanation}",
                  f"        How I got this: {o.calculation}",
                  f"        Source: {o.source}; {len(o.txn_ids)} transaction{'s' if len(o.txn_ids) != 1 else ''}"]
        if item.rejected_text:
            lines.append(f"        (model sentence rejected: {'; '.join(item.problems)})")

    accepted = len(plan.accepted)
    lines += ["", f"Before vs after ({accepted} of {len(plan.items)} suggestions accepted):",
              f"  {'':<22}{'current path':>18}{'with plan':>18}",
              f"  {'Monthly surplus':<22}{_money_or_na(p.current_surplus):>18}{_money_or_na(p.new_surplus):>18}"]
    for h in sorted(p.cumulative_before):
        lines.append(f"  {f'Saved in {h} months':<22}{_money_or_na(p.cumulative_before[h]):>18}"
                     f"{_money_or_na(p.cumulative_after[h]):>18}")
    if p.goal is not None and p.goal.kind == "target_by_date" and p.current_surplus is not None:
        lines.append(f"  {'Months to goal':<22}{_months(p.months_to_goal_before, p.on_track_before):>18}"
                     f"{_months(p.months_to_goal_after, p.on_track_after):>18}")
    lines += ["", *outcome_sentences(p), "", f"Method: {p.method}"]
    by_source = {s: sum(i.explanation_source == s for i in plan.items) for s in ("llm", "fallback", "template")}
    lines.append(f"Sentences: provider={plan.provider or 'none'}, {by_source['llm']} model-written and verified, "
                 f"{by_source['fallback']} replaced by template, {by_source['template']} template only.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> str:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--target", type=float, help="target amount (needs --by)")
    g.add_argument("--monthly", type=float, help="target saving per month")
    g.add_argument("--no-goal", action="store_true")
    ap.add_argument("--by", help="target date YYYY-MM-DD for --target")
    ap.add_argument("--saved", type=float, default=0.0, help="already saved toward --target")
    ap.add_argument("--name", default=None, help="goal name")
    ap.add_argument("--save-goal", action="store_true", help="store the goal in the DB as the active goal")
    ap.add_argument("--accept", action="append", help="opportunity id to accept (repeatable; default all)")
    ap.add_argument("--provider", default="fake", help="fake (default, offline) | ollama")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    args = ap.parse_args(argv)
    if args.target is not None and not args.by:
        ap.error("--target needs --by YYYY-MM-DD")

    conn = connect(args.db)
    try:
        if args.target is not None:
            goal = Goal.by_date(args.target, args.by, args.saved, name=args.name or "Savings goal")
        elif args.monthly is not None:
            goal = Goal.monthly(args.monthly, name=args.name or "Monthly savings")
        elif args.no_goal:
            goal = None
        else:
            goal = load_goal(conn) or SAMPLE_GOAL
        if goal is not None and args.save_goal:
            goal = save_goal(conn, goal)
    finally:
        conn.close()

    df = prepare(db_path=args.db)
    if df.empty:
        out = "No transactions: run `python -m src.db data/raw/statement.csv` first."
    elif df["category"].eq("Uncategorized").all():
        out = "No categorized transactions: run `python -m src.categorizer` first."
    else:
        out = render(build_plan(df, goal, get_provider(args.provider), args.accept))
    print(out)
    return out


if __name__ == "__main__":
    main()
