"""Terminal chat with the finance agent. Provider: --provider, else LLM_PROVIDER, else the offline fake.

Run:  python scripts/chat_cli.py                 # interactive; 'quit' to exit, 'reset' to clear memory
      python scripts/chat_cli.py --demo          # run a scripted 3-question conversation
      python scripts/chat_cli.py --trace-file data/agent_trace.jsonl   # also log every turn as JSONL
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agent import FinanceAgent, TurnTrace  # noqa: E402
from src.analytics.common import prepare  # noqa: E402
from src.db import DEFAULT_DB_PATH  # noqa: E402
from src.llm import PROVIDERS, get_provider  # noqa: E402

DEMO_QUESTIONS = (
    "How much did I spend on dining in May?",
    "And the month before that?",
    "Am I paying for any subscriptions I don't use?",
)


def how_i_got_this(trace: TurnTrace) -> str:
    """Compact version of the panel the UI will show."""
    if trace.refused:
        return "  [out of scope: fixed refusal, no model or tool call]"
    lines = []
    for c in trace.model_tool_calls:
        status = f"error: {c.error}" if c.error else f"{len(c.txn_ids)} source rows"
        lines.append(f"  -> {c.name}({json.dumps(c.args)})  via {c.source or '-'}, {status}")
    if trace.retried:
        lines.insert(0, f"  [retry: model first answered without a tool call: {trace.rejected_answers[0]!r}]")
    if trace.fallback_used:
        lines.append(f"  [model answer replaced; unverified numbers: {trace.unverified_numbers or 'none'}]")
    lines += [f"  [flag] {f}" for f in trace.injection_flags]
    return "\n".join(lines) or "  (no tool calls)"


def run(agent: FinanceAgent, question: str, show_trace: bool, echo: bool = False) -> None:
    reply = agent.ask(question)
    print((f"\nYou: {question}\n" if echo else "") + f"Agent: {reply.text}")
    if show_trace:
        print("How I got this:\n" + how_i_got_this(reply.trace))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    ap.add_argument("--provider", choices=PROVIDERS,
                    help="default: LLM_PROVIDER, else fake (offline); ollama calls a local model")
    ap.add_argument("--demo", action="store_true", help="run the scripted demo questions and exit")
    ap.add_argument("--no-trace", action="store_true", help="hide the 'How I got this' lines")
    ap.add_argument("--trace-file", type=Path, help="append every turn's trace as JSONL")
    args = ap.parse_args(argv)

    if not args.db.exists():
        print(f"No database at {args.db}. Run: python -m src.db data/raw/statement.csv")
        return 1
    txns = prepare(db_path=args.db)
    if txns["category"].eq("Uncategorized").all():
        print("No categorized transactions: run `python -m src.categorizer` first.")
        return 1
    agent = FinanceAgent(get_provider(args.provider), txns=txns, trace_path=args.trace_file)
    print(f"Finance agent ({agent.provider.name}), {len(txns)} transactions. Ask about your spending.")

    if args.demo:
        for q in DEMO_QUESTIONS:
            run(agent, q, not args.no_trace, echo=True)
        return 0
    while True:
        try:
            q = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not sys.stdin.isatty():
            print(q)  # piped input is not echoed by the terminal
        if q.lower() in {"quit", "exit", "q"}:
            return 0
        if q.lower() == "reset":
            agent.reset()
            print("(memory cleared)")
        elif q:
            run(agent, q, not args.no_trace)


if __name__ == "__main__":
    raise SystemExit(main())
