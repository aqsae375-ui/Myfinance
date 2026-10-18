"""results/eval/*.json (+ data/app/demo_checklist.json) -> results/evaluation_summary.md.

One table per section, slide-ready. Nothing is rounded up and every failure is listed.
Run:  python scripts/write_eval_summary.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ui.data import data_dir  # noqa: E402

EVAL = ROOT / "results" / "eval"
OUT = ROOT / "results" / "evaluation_summary.md"


def load(name: str) -> dict | None:
    p = EVAL / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def review() -> dict:
    return load("qa_ollama_review") or {}


def manual_score(prov: str, qa: dict) -> str:
    grades = review().get("grades", {}) if prov == "ollama" else {}
    if not grades:
        return "-"
    ids = [r["id"] for r in qa["rows"]]
    ok = sum(grades.get(i, ["?"])[0] == "correct" for i in ids)
    return f"{ok}/{len(ids)} ({100 * ok / len(ids):.1f}%)"


def pct(a: int, b: int) -> str:
    return f"{100 * a / b:.1f}%" if b else "n/a"


def table(head: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join("" if c is None else str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def yn(b: bool) -> str:
    return "yes" if b else "**no**"


def sec(s: float | None) -> str:
    return "n/a" if s is None else (f"{s * 1000:.0f} ms" if s < 1 else f"{s:.1f} s")


def main() -> None:
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    failures: list[str] = []
    md = [f"# Myfinance evaluation summary (Stage 7)", "",
          f"Date: {date.today().isoformat()} · Commit: `{sha}` · Local model: Ollama `qwen2.5:7b` on a laptop CPU "
          "(no GPU) · Data: synthetic only (seed 42, Mar-Aug 2026) plus the fictional PK fixture. "
          "\"fake\" = the offline deterministic test provider (a keyword planner, not a language model).", ""]

    # ---- 1. categorization ----
    cat = load("cat_ollama")
    md += ["## 1. Categorization accuracy (hard set, 60 hand-labelled messy descriptions)", ""]
    if cat:
        r, l, h, n = cat["rules_only"], cat["llm_only"], cat["hybrid"], cat["hard_rows"]
        md += [table(["Method", "Correct", "Accuracy", "Notes"], [
            ["Rules only", f"{r['correct']}/{n}", pct(r["correct"], n),
             f"{r['decided']} rows decided, {r['wrong']} wrong; {r['undecided']} undecided (counted wrong)"],
            ["LLM only (qwen2.5:7b, no rule hints)", f"{l['correct']}/{n}", pct(l["correct"], n),
             f"{l['unsure']} Unsure; {l['runtime_s']} s"],
            ["Hybrid (the app's pipeline)", f"{h['correct']}/{n}", pct(h["correct"], n),
             f"rules {h['by_source']['rule']['rows']} ({h['rule_wrong']} wrong) + LLM {h['rows_to_llm']} "
             f"({h['by_source']['llm']['wrong']} wrong); {h['runtime_s']} s"],
        ]), "", table(["Hybrid pipeline", "Value"], [
            ["Wrong rule-decided rows", h["rule_wrong"]],
            ["Rows sent to the LLM", f"{h['rows_to_llm']}/{n} ({h['rows_to_llm_pct']}%), "
                                     f"{h['llm_unique_sent']} unique descriptions"],
            ["Needs-review rate", f"{h['needs_review']}/{n} ({h['needs_review_pct']}%)"],
            ["Unsure (left uncategorized)", h["unsure"]],
            ["Synthetic statement (192 rows, same pipeline)",
             f"{cat['synthetic_hybrid']['correct']}/{cat['synthetic_hybrid']['rows']} "
             f"({cat['synthetic_hybrid']['accuracy_pct']}%), {cat['synthetic_hybrid']['rule_wrong']} wrong rule rows"],
        ]), ""]
        real = cat.get("real_statement_coverage")
        if real:
            md += [f"**Real statement: {real['rule_decided']}/{real['rows']} rows decided by rules. This is "
                   "coverage, NOT accuracy:** the statement has no labels, and the rules were written while "
                   "looking at it, so it cannot be used to measure accuracy.", ""]
        errs = h["errors"]
        if errs:
            md += ["Hybrid errors: " + "; ".join(f"`{e['description']}` expected {e['expected']}, got "
                                                 f"{e['predicted']} ({e['source']})" for e in errs), ""]
            failures += [f"Categorization: `{e['description']}` -> {e['predicted']} (expected {e['expected']})"
                         for e in errs]
        if h["rule_wrong"]:
            failures.append(f"Categorization: {h['rule_wrong']} rule-decided rows are wrong")
    else:
        md += ["_Not run with Ollama._", ""]
        failures.append("Categorization was not run with Ollama")

    # ---- 2. Q&A ----
    md += ["## 2. Q&A trust (24 fixed questions on the synthetic statement)", ""]
    qa_rows = []
    for prov in ("fake", "ollama"):
        qa = load(f"qa_{prov}")
        if not qa:
            continue
        s = qa["summary"]
        label = "fake (offline planner)" if prov == "fake" else "Ollama qwen2.5:7b"
        if qa.get("subset"):
            label += f" (subset: {len(qa['subset'])} questions)"
        qa_rows.append([label, s["questions"], s["answered"], s["refused"],
                        f"{s['fully_verified']}/{s['answered']} ({s['fully_verified_pct']}%)", s["retried"],
                        s["replaced_by_verifier"], s["unverified_numbers_shown_to_user"],
                        f"{s['correct']}/{s['questions']} ({s['correct_pct']}%)", manual_score(prov, qa),
                        sec(s["seconds_answered"]["median_s"]), sec(s["seconds_answered"]["worst_s"])])
        if s["unverified_numbers_shown_to_user"]:
            failures.append(f"Q&A ({prov}): {s['unverified_numbers_shown_to_user']} unverified numbers reached the user")
    md += [table(["Provider", "Questions", "Answered", "Refused", "Fully verified", "Retried", "Replaced by verifier",
                  "Unverified numbers shown to user", "Exact figure (automatic)", "Correct (manual review)",
                  "Median time", "Worst time"], qa_rows), "",
           "Fully verified = the model's own answer passed the number check (no retry fallback). Exact figure = "
           "the answer contains the specific analytics figure the check expects (strict: a true answer that gives "
           "the change instead of the total fails it). Correct (manual review) = each final answer read and graded "
           "against the analytics figures, with a note per question (`results/eval/qa_ollama_review.json`); reviewed "
           "automatically; independent human verification still recommended.", ""]
    fix = load("qa_after_fix_ollama")
    if fix:
        r = fix["rows"][0]
        md += [f"After the Stage 7 verifier fix (years/months typed in the question may be repeated), {r['id']} was "
               f"re-run on Ollama: number check {r['number_check']}, replaced: {'yes' if r['replaced_by_verifier'] else 'no'}. "
               f"Answer: \"{r['answer'][:200]}\"", ""]
    qb = load("qa_before_fake")
    if qb:
        s = qb["summary"]
        wrong = ", ".join(r["id"] for r in qb["rows"] if not r["correct"])
        md += [f"The fake planner first scored {s['correct']}/{s['questions']} (wrong: {wrong}). It sent \"why\" and "
               "\"up or down\" questions to the wrong tool, answered \"December 2025\" (no data) "
               "with August's figures and a next-year forecast with this month's groceries, which is misleading in the "
               "offline demo. Both were fixed in this stage, so the fake row above is tuned on these questions and is "
               "not a model result.", ""]
    detail = load("qa_ollama") or load("qa_fake")
    grades = review().get("grades", {}) if detail and detail["provider"].startswith("ollama") else {}
    if detail:
        prov = detail["provider"]
        md += [f"Per question ({prov}):", "", table(
            ["#", "Kind", "Question", "Result", "Number check", "Retried", "Replaced", "Exact figure", "Manual review",
             "Time"],
            [[r["id"], r["kind"], r["question"], "refused" if r["refused"] else "answered", r["number_check"] or "-",
              yn(r["retried"]).replace("**no**", "no"), "**yes**" if r["replaced_by_verifier"] else "no",
              yn(r["correct"]), (grades.get(r["id"], ["-"])[0]).replace("wrong", "**wrong**"), sec(r["seconds"])]
             for r in detail["rows"]]), ""]
    for prov in ("fake", "ollama"):
        qa = load(f"qa_{prov}")
        g = review().get("grades", {}) if prov == "ollama" else {}
        for r in (qa or {}).get("rows", []):
            if g and g.get(r["id"], ["correct"])[0] == "wrong":
                failures.append(f"Q&A ({prov}) {r['id']} \"{r['question']}\": {g[r['id']][1]}")
            elif not g and not r["correct"]:
                failures.append(f"Q&A ({prov}) {r['id']} \"{r['question']}\": wrong or missing figure. "
                                f"Answer: {r['answer'][:140].replace(chr(10), ' / ')}")
            if r["replaced_by_verifier"]:
                failures.append(f"Q&A ({prov}) {r['id']}: model answer replaced by the verifier "
                                f"(unverified: {', '.join(r['unverified_in_model_draft']) or 'empty answer'})")

    # ---- 3. injection ----
    md += ["## 3. Prompt-injection test (8 instruction-like transactions)", ""]
    inj = load("inject_ollama") or load("inject_fake")
    before = load("inject_before_fake")
    if inj:
        s = inj["summary"]
        md += [table(["Description", "Detected", "Sent to categorizer model", "Sent to chat model",
                      "Needs-review chip (ledger)", "Warning chip (Ask)", "Blocked"],
                     [[f"`{r['description']}`", yn(r["detected"]), "**yes**" if r["sent_to_categorizer_model"] else "no",
                       "**yes**" if r["sent_to_chat_model"] else "no", yn(r["review_chip_in_ledger"]),
                       yn(r["flagged_in_ask"]), yn(r["blocked"])] for r in inj["rows"]]), "",
               table(["Result", "Value"], [
                   ["Blocked (after the Stage 7 fix)", f"**{s['blocked']}/{s['rows']}**"],
                   ["Blocked before the fix", f"{before['summary']['blocked']}/{before['summary']['rows']} "
                                              f"(detected {before['summary']['detected']}/8; all 8 reached the chat "
                                              "model inside tool results)" if before else "n/a"],
                   ["Agent answers checked", f"{s['answers']} ({', '.join(sorted({a['provider'] for a in inj['answers']}))})"],
                   ["Answers that followed an injection", s["answers_following_injection"]],
                   ["Held-out phrasings detected (not tuned on)", f"**{s['held_out_detected']}/{s['held_out_rows']}**"],
                   ["Answers whose text was influenced by an undetected (held-out) row",
                    f"**{s.get('answers_influenced_by_held_out', 'n/a')}**/{s['answers']}"],
               ]), "",
               "Blocked = detected, never in any categorizer or chat prompt, shown with a warning chip in the ledger "
               "and in Ask, and no answer acted on it. The detector is a phrase regex widened in this stage for these "
               "8 rows, so 8/8 is measured on rows it was tuned for. The 8 held-out phrasings show how it "
               "generalizes. Undetected rows reach the model as data inside a delimited block, and the number "
               "verifier still blocks any invented figure.", ""]
        if s["blocked"] < s["rows"]:
            failures.append(f"Injection: only {s['blocked']}/{s['rows']} blocked")
        missed = [h["description"] for h in inj.get("held_out", []) if not h["detected"]]
        if missed:
            failures.append(f"Injection: held-out phrasings not detected ({len(missed)}/{len(inj['held_out'])}): "
                            + "; ".join(f"`{m}`" for m in missed) + ". They reach the model as data.")
        for a in inj["answers"]:
            if a["follows_injection"]:
                failures.append(f"Injection: an answer followed injected text ({a['provider']}, \"{a['question']}\")")
            if a.get("influenced_by_held_out"):
                failures.append(f"Injection: undetected held-out text changed an answer's WORDS ({a['provider']}, "
                                f"\"{a['question']}\"): \"{a['answer'][:220]}\". Numbers stayed verified; the "
                                "statement about the rows was false.")

    # ---- 4. redaction ----
    md += ["## 4. Redaction leak test (exact text of every model call)", ""]
    leak = load("leak_fake")
    if leak:
        md += [table(["Dataset", "Rows", "Model calls logged", "Tool results logged", "Characters scanned",
                      "PII terms checked", "Scanner positive control (raw text)", "Leaks"],
                     [[k, d["rows"], d["model_calls"], d["tool_results_scanned"], f"{d['chars_scanned']:,}",
                       ", ".join(f"{v} {t}" for t, v in d["pii_terms_checked"].items()),
                       f"{d['positive_control_raw_hits']} hits, all types {'found' if d['positive_control_all_types_found'] else 'NOT found'}",
                       f"**{d['leaks']}**"] for k, d in leak.items() if isinstance(d, dict) and "rows" in d]), "",
               "Logged: categorizer prompts (the real pipeline, plus a worst case with every row sent), agent chat "
               "messages (questions for every month), find_transactions / subscriptions / anomalies results as the "
               "model sees them, and advisor prompts. Also scanned for e-mail, long digit runs, IBAN and phone "
               "shapes. The prompt text does not depend on the provider, so this runs on the fake provider. "
               "Logs: `results/eval/model_calls_*.jsonl` (written by the leak section; not committed).", ""]
        for k, d in leak.items():
            if isinstance(d, dict) and d.get("leaks"):
                failures.append(f"Redaction: {d['leaks']} leaks in {k}: " +
                                "; ".join(f"{h['type']} `{h['term']}`" for h in d["hits"][:5]))

    # ---- 5. latency ----
    md += ["## 5. Latency", ""]
    lf, lo = load("latency_fake") or {}, load("latency_ollama") or {}
    rows = []
    for label, d, prov in [("Load statement end to end (synthetic, 192 rows)", lf.get("load_synthetic_fake"), "fake"),
                           ("Load statement end to end (PK fixture, 83 rows)", lf.get("load_pk_fixture_fake"), "fake"),
                           ("Load statement end to end (synthetic, cold LLM cache)", lo.get("load_synthetic_ollama"), "Ollama"),
                           ("Dashboard render, first (cold cache)", (lf.get("dashboard_render") or {}).get("cold"), "-"),
                           ("Dashboard render, repeat", (lf.get("dashboard_render") or {}).get("warm"), "-"),
                           ("Dashboard month switch", (lf.get("dashboard_render") or {}).get("month_switch"), "-"),
                           ("First answer", lf.get("first_answer_fake"), "fake"),
                           ("First answer, cold model load", (lo.get("first_answer_ollama") or {}).get("cold_model_load"), "Ollama"),
                           ("First answer, model loaded", (lo.get("first_answer_ollama") or {}).get("warm"), "Ollama")]:
        if d and d.get("n"):
            rows.append([label, prov, d["n"], sec(d["median_s"]), sec(d["worst_s"])])
    qo = load("qa_ollama")
    if qo:
        s = qo["summary"]["seconds_answered"]
        rows.append(["Any answer (Q&A set, answered questions)", "Ollama", s["n"], sec(s["median_s"]), sec(s["worst_s"])])
    md += [table(["Measure", "Model", "Runs", "Median", "Worst"], rows), "",
           "Dashboard = the Streamlit script run measured with AppTest (server side; browser paint not included). "
           "Load = parse + redact + SQLite ingest + categorize, the app's upload path.", ""]

    # ---- 6. demo safety ----
    md += ["## 6. Demo safety (`python scripts/prepare_demo.py`)", ""]
    ck = data_dir() / "demo_checklist.json"
    if ck.exists():
        items = json.loads(ck.read_text(encoding="utf-8"))
        md += [table(["Check", "Result", "Detail"], [[c["check"], "PASS" if c["passed"] else "**FAIL**", c["detail"]]
                                                    for c in items]), ""]
        failures += [f"Demo: {c['check']} FAILED ({c['detail']})" for c in items if not c["passed"]]
    else:
        md += ["_prepare_demo.py has not been run._", ""]

    runs = []
    for p in sorted(EVAL.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        if "section" not in d:
            continue
        runs.append([p.name, d.get("provider"), d.get("commit"), d.get("run_at"), sec(d.get("section_runtime_s"))])
    md += ["## Runs (provenance)", "",
           "`+dirty` = measured on a working tree with uncommitted changes to src/, scripts/ or app.py. "
           "`ae32f18+dirty` runs used the Stage 7 code before it was committed: the injection fixes were already in; "
           "the verifier question-period fix, the fake-planner fixes and some harness scoring came after (re-run "
           "where it matters: inject_ollama, qa_after_fix_ollama). `585fb35+dirty` differs from `585fb35` only in a "
           "prepare_demo.py message string.", "",
           table(["Result file", "Provider", "Commit", "Run at", "Runtime"], runs), ""]

    md += ["## Failures and known limits", ""]
    md += [f"- {f}" for f in failures] or ["- None."]
    md += ["- The number verifier ignores signs: a direction error (\"received\" for a debit) can still pass "
           "(known since Stage 4).",
           "- Ollama on CPU is too slow for a live Q&A demo; the demo runs on the offline provider with pre-recorded "
           "traces (`results/demo_traces.jsonl`).", ""]
    OUT.write_text("\n".join(md), encoding="utf-8")
    print(f"Saved -> {OUT} ({len(failures)} failures listed)")


if __name__ == "__main__":
    main()
