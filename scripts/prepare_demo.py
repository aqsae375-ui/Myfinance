"""Demo safety: build a fresh demo DB, pre-categorize it, record 5 example answers, then prove the app
runs fully offline. Prints a pass/fail checklist; exit code 1 if anything fails.

The demo DB is the app's own sample DB (data/app/sample.db, what "Load sample data" opens), so the
sidebar needs no change. Every row is categorized now (rules, then --provider for the rows no rule
decides; answers are cached in the DB), so the live demo never calls a model to categorize.

Run:  python scripts/prepare_demo.py                     # rules + offline fake model (default)
      python scripts/prepare_demo.py --provider ollama   # categorize + record traces with the local model
Then: streamlit run app.py   (keep LLM_PROVIDER=fake, or unset, for a fully offline Ask tab)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.generate_data import build_statement, write_outputs  # noqa: E402
from src.agent.chat import FinanceAgent  # noqa: E402
from src.agent.verify import unverified_numbers  # noqa: E402
from src.categorizer import categorize_transactions, review_queue, rows_needing_llm  # noqa: E402
from src.db import connect, ingest, load_transactions  # noqa: E402
from src.llm import get_provider  # noqa: E402
from src.parser import parse_statement  # noqa: E402
from src.ui.data import data_dir, sample_db_path  # noqa: E402

DEMO_QUESTIONS = (
    "How much did I spend on dining out in May 2026?",
    "Compare my spending in May 2026 and June 2026.",
    "Which subscriptions might be unused?",
    "Were there any unusual charges in July 2026?",
    "Should I invest my savings in bitcoin?",
)
TRACES_RESULTS = ROOT / "results" / "demo_traces.jsonl"

# Runs in a fresh process with the network blocked (any non-loopback address, and the local Ollama port;
# loopback stays open because asyncio's event loop uses a localhost socket pair on Windows), then the app
# is driven through all three views.
OFFLINE_RUN = textwrap.dedent("""
    import socket
    _connect = socket.socket.connect
    def _guarded(self, address):
        host, port = (address[0], address[1]) if isinstance(address, tuple) else (str(address), 0)
        if host not in ("127.0.0.1", "::1", "localhost") or port == 11434:
            raise OSError(f"network blocked by prepare_demo offline check: {{host}}:{{port}}")
        return _connect(self, address)
    socket.socket.connect = _guarded
    try:  # the guard itself must work, or this check would prove nothing
        socket.create_connection(("127.0.0.1", 11434), timeout=2)
        raise SystemExit("guard failed: the Ollama port was reachable")
    except OSError:
        pass
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("app.py", default_timeout=300)
    at.session_state["db_path"] = {db!r}
    at.run()
    assert not at.exception and not at.error, [e.value for e in [*at.exception, *at.error]]
    assert sum('class="fa-kpi"' in m.value for m in at.markdown) == 4, "dashboard KPIs missing"
    at.segmented_control(key="nav").set_value("Ask").run()
    at.button(key="suggest-0").click().run()
    assert not at.exception and not at.error, [e.value for e in [*at.exception, *at.error]]
    turn = at.session_state["chat"][-1]
    assert turn.get("answer") and turn["trace"].verified, turn
    at.segmented_control(key="nav").set_value("Your Plan").run()
    assert not at.exception and not at.error, [e.value for e in [*at.exception, *at.error]]
    print("OFFLINE_OK")
""")


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def build_demo_db(provider_name: str) -> Path:
    """Fresh sample.db: synthetic statement -> parse -> redact + ingest -> categorize (cached)."""
    path = sample_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    csv_path, _ = write_outputs(build_statement(seed=42), data_dir() / "sample_raw", seed=42)
    tmp = path.with_suffix(".tmp.db")
    tmp.unlink(missing_ok=True)
    conn = connect(tmp)
    try:
        ingest(conn, parse_statement(csv_path), "sample statement.csv")
        categorize_transactions(conn, provider=get_provider(provider_name))
    finally:
        conn.close()
    path.unlink(missing_ok=True)
    tmp.replace(path)
    return path


def record_traces(db: Path, provider_name: str) -> list[dict]:
    traces_path = data_dir() / "demo_traces.jsonl"
    traces_path.unlink(missing_ok=True)
    conn = connect(db)
    txns = load_transactions(conn)
    conn.close()
    out = []
    for q in DEMO_QUESTIONS:
        agent = FinanceAgent(get_provider(provider_name), txns=txns, trace_path=traces_path)
        reply = agent.ask(q)
        t = reply.trace
        shown_bad = unverified_numbers(reply.text, [(c.args, c.result) for c in t.tool_calls])
        out.append({"question": q, "refused": t.refused, "verified": t.verified and not t.fallback_used,
                    "unverified_shown": shown_bad, "answer": reply.text})
    TRACES_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    TRACES_RESULTS.write_text(traces_path.read_text(encoding="utf-8"), encoding="utf-8")
    return out


def offline_check(db: Path) -> tuple[bool, str]:
    env = {**os.environ, "LLM_PROVIDER": "fake", "FINANCE_APP_DATA": str(data_dir())}
    proc = subprocess.run([sys.executable, "-c", OFFLINE_RUN.format(db=str(db))], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=600)
    ok = proc.returncode == 0 and "OFFLINE_OK" in proc.stdout
    return ok, "" if ok else (proc.stdout + proc.stderr).strip().splitlines()[-1][:300]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", default="fake", help="model for the rows no rule decides, and for the traces")
    args = ap.parse_args()

    checks: list[tuple[str, bool, str]] = []
    db = build_demo_db(args.provider)
    conn = connect(db)
    total = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    uncategorized = conn.execute("SELECT COUNT(*) FROM transactions WHERE category IS NULL").fetchone()[0]
    queue = review_queue(conn)
    pending = rows_needing_llm(conn)
    conn.close()
    checks.append(("Fresh demo DB built", total > 0, f"{_rel(db)}, {total} rows"))
    checks.append(("No row waits for a live LLM call", pending == 0, f"{pending} rows still need the model"))
    checks.append(("Every uncategorized row is in the review queue", uncategorized <= len(queue),
                   f"{uncategorized} uncategorized, {len(queue)} in review (the planted injection row)"))

    traces = record_traces(db, args.provider)
    answered = [t for t in traces if not t["refused"]]
    checks.append((f"{len(traces)} example questions recorded with traces", len(traces) == len(DEMO_QUESTIONS),
                   f"{_rel(TRACES_RESULTS)} and {_rel(data_dir() / 'demo_traces.jsonl')}"))
    checks.append(("Every example answer passed the number check", all(t["verified"] for t in answered),
                   ", ".join(f"Q{i + 1}: {'ok' if t['verified'] else 'REPLACED'}" for i, t in enumerate(traces)
                             if not t["refused"])))
    checks.append(("0 unverified numbers shown to the user", not any(t["unverified_shown"] for t in traces),
                   str(sum(len(t["unverified_shown"]) for t in traces))))
    checks.append(("Out-of-scope example refused", traces[-1]["refused"], traces[-1]["question"]))

    ok, detail = offline_check(db)
    checks.append(("App runs offline (network blocked): Dashboard, Ask, Your Plan", ok, detail or "internet and the Ollama port refused; all three views rendered"))
    app_provider = get_provider().name  # what the sidebar defaults to (LLM_PROVIDER / .env)
    checks.append(("App's default model is the offline one (LLM_PROVIDER unset or fake)", app_provider == "fake",
                   f"LLM_PROVIDER resolves to {app_provider!r}" + ("" if app_provider == "fake" else
                                                                    ": Ask would call a live model")))

    print(f"\nDemo checklist (provider for categorization + traces: {args.provider})")
    for name, passed, detail in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}  ({detail})")
    failed = [c for c in checks if not c[1]]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} passed." +
          ("" if failed else " Start the app with: streamlit run app.py, then 'Load sample data'."))
    (data_dir() / "demo_checklist.json").write_text(
        json.dumps([{"check": n, "passed": p, "detail": d} for n, p, d in checks], indent=2), encoding="utf-8")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
