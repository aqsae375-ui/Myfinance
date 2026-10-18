"""Rule audit (Stage 2d): every rule in src/rules.py with its pattern, category, confidence, risk rating and
whether it decides on its own (confidence >= RULE_THRESHOLD) or is only a hint for the LLM.

A rule is "ambiguous" when its matched word can mean two things (MART, MANDI, STORE, CAFE, POWER, a brand
name that is also a common word) and "clear" when it names one kind of transaction or business.

Run:  python scripts/audit_rules.py [--short]     (--short truncates long patterns to one line)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rules import RULE_THRESHOLD, RULES  # noqa: E402

PATTERN_WIDTH = 70


def audit_table(short: bool = False) -> str:
    head = f"{'#':>2}  {'rule':<22}{'category':<18}{'conf':>5}  {'risk':<10}{'decides':<8} pattern"
    lines = [head, "-" * (len(head) + PATTERN_WIDTH - 7)]
    indent = " " * 67
    for i, r in enumerate(RULES, start=1):
        direction = "" if r.direction == "any" else f" [{r.direction} only]"
        pattern = r.pattern + direction
        cut = pattern if len(pattern) <= PATTERN_WIDTH else pattern[:PATTERN_WIDTH - 4] + " ..."
        chunks = [pattern[j:j + PATTERN_WIDTH] for j in range(0, len(pattern), PATTERN_WIDTH)]
        wrapped = cut if short else ("\n" + indent).join(chunks)
        lines.append(f"{i:>2}  {r.name:<22}{r.category:<18}{r.confidence:>5.2f}  {r.risk:<10}"
                     f"{'yes' if r.decides else 'HINT':<8} {wrapped}")
    n_amb = sum(r.risk == "ambiguous" for r in RULES)
    lines.append(f"\n{len(RULES)} rules: {len(RULES) - n_amb} clear (decide at >= {RULE_THRESHOLD}), "
                 f"{n_amb} ambiguous (hint only, never decide).")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--short", action="store_true")
    print(audit_table(ap.parse_args(argv).short))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
