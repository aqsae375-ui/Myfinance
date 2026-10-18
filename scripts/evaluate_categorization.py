"""Evaluate categorization against labelled data and save results/categorization.json.

Two datasets:
  synthetic  data/raw/statement.csv + ground_truth.csv (Stage 1; easy, we generated it)
  hard       data/eval/hard_descriptions.csv (60 messy real-world-style descriptions)

Each dataset runs through the real pipeline (parse -> redact -> ingest -> rules -> LLM) in a fresh
in-memory DB with an empty cache, so the LLM call/token counts reflect a cold run.

Run:  LLM_PROVIDER=fake python scripts/evaluate_categorization.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.categories import CATEGORIES, GROUND_TRUTH_MAP  # noqa: E402
from src.categorizer import categorize_transactions  # noqa: E402
from src.llm.base import UNSURE  # noqa: E402
from src.db import connect, ingest, load_transactions  # noqa: E402
from src.llm import LLMProvider, get_provider  # noqa: E402
from src.parser import parse_statement  # noqa: E402

RAW = ROOT / "data" / "raw"
DEFAULT_HARD = ROOT / "data" / "eval" / "hard_descriptions.csv"
DEFAULT_OUT = ROOT / "results" / "categorization.json"


def load_synthetic(statement: Path, truth: Path) -> tuple[pd.DataFrame, list[str]]:
    df = parse_statement(statement)
    gt = pd.read_csv(truth).sort_values("row")
    if len(gt) != len(df):
        raise SystemExit(f"{truth} has {len(gt)} labels but {statement} has {len(df)} rows")
    unknown = set(gt["category"]) - set(GROUND_TRUTH_MAP)
    if unknown:
        raise SystemExit(f"Unmapped ground-truth labels: {sorted(unknown)}")
    return df, [GROUND_TRUTH_MAP[c] for c in gt["category"]]


def load_hard(path: Path) -> tuple[pd.DataFrame, list[str]]:
    raw = pd.read_csv(path)
    bad = set(raw["category"]) - set(CATEGORIES)
    if bad:
        raise SystemExit(f"{path}: labels not in taxonomy: {sorted(bad)}")
    df = pd.DataFrame({
        "date": pd.Timestamp("2026-09-01"),
        "description": raw["description"].astype(str),
        "amount": raw["amount"].astype(float),
    })
    df["type"] = df["amount"].map(lambda a: "credit" if a >= 0 else "debit")
    return df, list(raw["category"])


def _acc(correct: int, n: int) -> float | None:
    return round(correct / n, 4) if n else None


def evaluate(df: pd.DataFrame, expected: list[str], provider: LLMProvider) -> dict:
    conn = connect(":memory:")
    ingest(conn, df, source_file="eval")
    report = categorize_transactions(conn, provider)
    out = load_transactions(conn)
    conn.close()
    if len(out) != len(expected):
        raise SystemExit(f"Ingested {len(out)} rows but have {len(expected)} labels")

    out["expected"] = expected
    # A row the model was unsure about has no category: shown as "Unsure" and counted as not correct.
    out["category"] = out["category"].fillna(UNSURE)
    out["correct"] = out["category"] == out["expected"]

    per_category = {}
    for cat in CATEGORIES:
        truth, pred = out["expected"] == cat, out["category"] == cat
        tp = int((truth & pred).sum())
        if truth.any() or pred.any():
            per_category[cat] = {"support": int(truth.sum()), "correct": tp,
                                 "accuracy": _acc(tp, int(truth.sum())),  # recall on this category
                                 "precision": _acc(tp, int(pred.sum()))}

    by_source = {}
    for src_name in ("rule", "learned", "llm"):
        part = out[out["category_source"] == src_name]
        by_source[src_name] = {"rows": len(part), "accuracy": _acc(int(part["correct"].sum()), len(part)),
                               "wrong": int((~part["correct"]).sum())}

    labels = [c for c in (*CATEGORIES, UNSURE) if c in set(out["expected"]) | set(out["category"])]
    cm = pd.crosstab(pd.Categorical(out["expected"], labels), pd.Categorical(out["category"], labels),
                     rownames=["expected"], colnames=["predicted"], dropna=False)

    errors = out[~out["correct"]]
    return {
        "rows": len(out),
        "accuracy": _acc(int(out["correct"].sum()), len(out)),
        "by_source": by_source,
        "rule_wrong": by_source["rule"]["wrong"],  # should be zero: a rule decides only when confident
        "unsure": int((out["category"] == UNSURE).sum()),
        "needs_review": int(out["needs_review"].sum()),
        "hinted": report.hinted,                    # LLM rows that carried a low-confidence rule hint
        "per_category": per_category,
        "confusion_matrix": {"labels": labels, "matrix": cm.values.tolist()},
        "llm": {"unique_descriptions_sent": report.llm_unique_sent, "invalid_answers": report.llm_invalid,
                "calls": provider.stats.calls, "input_tokens": provider.stats.input_tokens,
                "output_tokens": provider.stats.output_tokens,
                "tokens_estimated": provider.stats.estimated_tokens},
        "errors": [{"description": r.redacted_description, "expected": r.expected,
                    "predicted": r.category, "source": r.category_source} for r in errors.itertuples()],
        "_cm_frame": cm,
    }


def cost_block(provider_name: str, input_tokens: int, output_tokens: int, estimated: bool) -> dict:
    """Token usage only: every provider runs locally or offline, so the actual cost is $0."""
    return {
        "input_tokens": input_tokens, "output_tokens": output_tokens, "tokens_estimated": estimated,
        "actual_usd": 0.0,
        "note": ("Token counts estimated at ~4 chars/token. " if estimated else "")
                + f"Local/offline provider ({provider_name}): actual cost $0.",
    }


def print_dataset(name: str, res: dict) -> None:
    print(f"\n=== {name}: {res['rows']} rows, accuracy {res['accuracy']:.1%}, "
          f"runtime {res['runtime_seconds']:.1f}s ===")
    for s, v in res["by_source"].items():
        acc = "n/a" if v["accuracy"] is None else f"{v['accuracy']:.1%}"
        print(f"  {s:<7} rows: {v['rows']:>4}   accuracy: {acc}   wrong: {v['wrong']}")
    print(f"  Unsure: {res['unsure']}   needs review: {res['needs_review']}   "
          f"LLM rows with a rule hint: {res['hinted']}")
    llm = res["llm"]
    print(f"  LLM: {llm['unique_descriptions_sent']} unique descriptions in {llm['calls']} calls, "
          f"{llm['invalid_answers']} invalid -> Unsure")
    print("  Per category (accuracy = correct / support):")
    for cat, v in res["per_category"].items():
        acc = "  n/a" if v["accuracy"] is None else f"{v['accuracy']:>5.0%}"
        print(f"    {cat:<17} {v['correct']:>3}/{v['support']:<3} {acc}")
    print("  Confusion matrix (rows = expected, cols = predicted):")
    cm = res["_cm_frame"].copy()
    cm.columns = [c[:6] for c in cm.columns]
    print("    " + cm.to_string().replace("\n", "\n    "))
    if res["errors"]:
        print(f"  Misclassified ({len(res['errors'])}):")
        for e in res["errors"][:20]:
            print(f"    [{e['source']}] {e['description'][:60]!r}: expected {e['expected']}, got {e['predicted']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--statement", type=Path, default=RAW / "statement.csv")
    ap.add_argument("--truth", type=Path, default=RAW / "ground_truth.csv")
    ap.add_argument("--hard", type=Path, default=DEFAULT_HARD)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--provider", help="Override LLM_PROVIDER (ollama|fake)")
    args = ap.parse_args()

    datasets = {"synthetic": load_synthetic(args.statement, args.truth), "hard": load_hard(args.hard)}
    results, providers = {}, []
    for name, (df, expected) in datasets.items():
        provider = get_provider(args.provider)  # fresh provider per dataset -> separate token stats
        if hasattr(provider, "ensure_ready"):
            provider.ensure_ready()
        providers.append(provider)
        t0 = time.perf_counter()
        results[name] = evaluate(df, expected, provider)
        results[name]["runtime_seconds"] = round(time.perf_counter() - t0, 2)
        print_dataset(name, results[name])

    name = providers[0].name
    total_in = sum(p.stats.input_tokens for p in providers)
    total_out = sum(p.stats.output_tokens for p in providers)
    cost = cost_block(name, total_in, total_out, any(p.stats.estimated_tokens for p in providers))
    print(f"\nProvider: {name}. LLM tokens: {total_in} in / {total_out} out. "
          f"Cost: ${cost['actual_usd']:.4f}{' (estimated tokens)' if cost['tokens_estimated'] else ''}.")

    for res in results.values():
        res.pop("_cm_frame")
    payload = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "provider": name, "datasets": results, "cost": cost}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
