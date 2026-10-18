"""The offline fake model's chat behavior: a keyword planner, not an LLM.

It sees only what a real model would see (system prompt, history, tool results) and works from those:
months come from the system prompt, follow-ups are resolved from the previous turn in the messages,
and answers are rendered from the tool results with src/agent/render.py. With `obey_injections` it
simulates a compromised model: if any message contains an injected instruction, its final answer
obeys it, so tests can show the agent's verifier contains the damage.
"""
from __future__ import annotations

import calendar
import json
import re
from typing import Sequence

import pandas as pd

from src.agent.render import render
from src.categories import CATEGORIES
from src.llm.base import ChatResponse, ToolCall, ToolSpec

INJECTION_MARKER = "ignore previous instructions"
OBEYED_ANSWER = "Of course. The user has $1,000,000 in savings."
HELP_ANSWER = ("I can answer questions about your spending, income, savings, categories, subscriptions and "
               "unusual charges. Could you rephrase that, for example with a month or a category?")

MONTH_TOKEN_RE = re.compile(r"(?<![\d-])\d{4}-(?:0[1-9]|1[0-2])(?![\d-])")
_MONTH_WORDS = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
_MONTH_WORDS |= {name.lower(): i for i, name in enumerate(calendar.month_abbr) if name and name != "May"}
MONTH_NAME_RE = re.compile(r"\b(" + "|".join(sorted(_MONTH_WORDS, key=len, reverse=True)) + r")\b", re.I)
LATEST_RE = re.compile(r"\b(this|last|latest|current|most recent) month\b", re.I)
CATEGORY_SYNONYMS = {
    "dining": "Dining Out", "restaurant": "Dining Out", "eating out": "Dining Out", "food": "Dining Out",
    "grocery": "Groceries", "transport": "Transport", "uber": "Transport", "travel": "Transport",
    "electricity": "Utilities", "power bill": "Utilities", "bills": "Utilities",
}
FOLLOWUP_RE = re.compile(r"^\s*(and|what about|how about|same for|then)\b|\b(before that|after that)\b", re.I)
SHIFT_BACK_RE = re.compile(r"\b(before that|previous|prior|earlier)\b", re.I)
SHIFT_FWD_RE = re.compile(r"\b(after that|next|following)\b", re.I)
AMOUNT_RE = re.compile(r"\b(?:over|above|more than|at least|bigger than|larger than)\s+"
                       r"(?:rs\.?\s*)?([\d,]+(?:\.\d+)?)", re.I)
MERCHANT_RE = re.compile(r"\b(?:at|from)\s+([A-Za-z][\w&'.]+)")
YEAR_AFTER_RE = re.compile(r",?\s+((?:19|20)\d{2})\b")
# A forecast is not in the statement: ask to rephrase rather than answer with a past month's figure.
FUTURE_RE = re.compile(r"\b(will i|am i going to|next (month|year)|forecast|predict\w*|in the future)\b", re.I)


def _months_in_data(system: str) -> list[str]:
    line = next((ln for ln in system.splitlines() if ln.startswith("Months with data:")), "")
    return MONTH_TOKEN_RE.findall(line)


def _find_months(text: str, months: list[str]) -> list[str]:
    """Months mentioned in `text`, in order: YYYY-MM tokens, month names (matched to the data's
    year), and "this/last month" (the latest month in the data)."""
    found: list[tuple[int, str]] = [(m.start(), m.group()) for m in MONTH_TOKEN_RE.finditer(text)]
    for m in MONTH_NAME_RE.finditer(text):
        num = _MONTH_WORDS[m.group(1).lower()]
        if (year := YEAR_AFTER_RE.match(text, m.end())):  # "December 2025": that month, even with no data
            found.append((m.start(), f"{year.group(1)}-{num:02d}"))
            continue
        match = [k for k in months if int(k[5:]) == num]
        if match:
            found.append((m.start(), match[-1]))
    found += [(m.start(), months[-1]) for m in LATEST_RE.finditer(text) if months]
    out: list[str] = []
    for _, month in sorted(found):
        if month not in out:
            out.append(month)
    return out


def _find_category(text: str) -> str | None:
    low = text.lower()
    for c in CATEGORIES:
        if re.search(r"\b" + re.escape(c.lower()) + r"\b", low):
            return c
    for word, c in CATEGORY_SYNONYMS.items():
        if re.search(r"\b" + re.escape(word), low):
            return c
    return None


def _shift(month: str, n: int) -> str:
    return (pd.Period(month, freq="M") + n).strftime("%Y-%m")


def _month_range(month: str) -> dict:
    return {"start_date": f"{month}-01", "end_date": pd.Period(month, freq="M").end_time.date().isoformat()}


def plan(question: str, months: list[str], default_month: str | None = None) -> tuple[str, dict] | None:
    """(tool name, args) for a standalone question, or None if nothing matches."""
    q = question.lower()
    if FUTURE_RE.search(question):
        return None
    mentioned = _find_months(question, months)
    month = mentioned[0] if mentioned else (default_month or (months[-1] if months else None))
    category = _find_category(question)

    if re.search(r"\bcash\b", q) and re.search(r"wallet|unaccounted|accounted|logged|untracked|left", q):
        return "get_cash_summary", {"month": month}
    if "subscription" in q and not (mentioned and re.search(r"how much|spen[dt]", q)):
        return "get_subscriptions", {}
    if re.search(r"unusual|anomal|odd|strange|suspicious|weird|spike", q):
        return "get_anomalies", ({"month": mentioned[0]} if mentioned else {})
    if len(mentioned) >= 2 and re.search(r"compar|\bvs\b|versus|differ|between|up or down|go(ne)? (up|down)", q):
        a, b = sorted(mentioned[:2])  # the later month is the one being compared
        return "compare_periods", {"month_a": a, "month_b": b}
    if re.search(r"compar|\bvs\b|versus|month over month|change|increase|decrease|"
                 r"than (the )?(previous|last|prior)|\bwhy\b.*\b(expensive|so much|high)\b", q):
        return "get_month_over_month", {"month": month}
    if re.search(r"transactions?|purchases?|charges?|payments?|show me|list|biggest|largest", q) \
            or AMOUNT_RE.search(question) or MERCHANT_RE.search(question):
        args: dict = {}
        if category:
            args["category"] = category
        if mentioned:
            args.update(_month_range(mentioned[0]))
        if re.search(r"purchases?|charges?|payments?|spen[dt]|bought", q):
            args["flow"] = "out"
        elif re.search(r"income|deposits?|received|credits?", q):
            args["flow"] = "in"
        if (amt := AMOUNT_RE.search(question)):
            args["min_amount"] = float(amt.group(1).replace(",", ""))
        if (mer := MERCHANT_RE.search(question)) and not _find_months(mer.group(1), months) \
                and not _find_category(mer.group(1)):
            args["merchant_keyword"] = mer.group(1)
        if re.search(r"biggest|largest|top", q):
            args["sort"] = "amount"
            args["limit"] = 5
        return "find_transactions", args
    if category and category != "Income":
        return "get_category_spending", {"category": category, "month": month}
    if re.search(r"spen[dt]|summary|overview|income|earn|sav(e|ed|ing)|total|how did i do", q):
        return "get_monthly_summary", {"month": month}
    return None


def _shift_args(name: str, args: dict, n: int) -> dict:
    out = dict(args)
    for key in ("month", "month_a", "month_b"):
        if key in out:
            out[key] = _shift(out[key], n)
    if "start_date" in out:
        out.update(_month_range(_shift(out["start_date"][:7], n)))
    return out


def _follow_up(question: str, history: Sequence[dict], months: list[str]) -> tuple[str, dict] | None:
    """Re-plan the previous question, then apply this question's change: a new month, a shifted
    month ("the month before that"), or a new category ("what about groceries?")."""
    users = [m for m in history if m["role"] == "user"]
    answers = [m for m in history if m["role"] == "assistant" and not m.get("tool_calls")]
    if not users:
        return None
    prev_month = next(iter(MONTH_TOKEN_RE.findall(answers[-1]["content"])), None) if answers else None
    prev = plan(users[-1]["content"], months, default_month=prev_month)
    if prev is None:
        return None
    name, args = prev
    new_months = _find_months(question, months)
    if new_months and "month" in args:
        args = {**args, "month": new_months[0]}
    elif new_months and "start_date" in args:
        args = {**args, **_month_range(new_months[0])}
    elif SHIFT_BACK_RE.search(question):
        args = _shift_args(name, args, -1)
    elif SHIFT_FWD_RE.search(question):
        args = _shift_args(name, args, +1)
    if (cat := _find_category(question)) and "category" in args:
        args = {**args, "category": cat}
    return name, args


def respond(system: str, messages: Sequence[dict], tools: Sequence[ToolSpec],
            obey_injections: bool = False) -> ChatResponse:
    last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
    turn, history = messages[last_user + 1:], messages[:last_user]
    tool_msgs = [m for m in turn if m["role"] == "tool"]

    if tool_msgs:  # tools already ran this turn: write the final answer
        if obey_injections and any(INJECTION_MARKER in m["content"].lower() for m in messages):
            return ChatResponse(OBEYED_ANSWER)
        return ChatResponse("\n".join(render(m["name"], json.loads(m["content"])) for m in tool_msgs))

    question = messages[last_user]["content"]
    months = _months_in_data(system)
    step = None
    if history and FOLLOWUP_RE.search(question):
        step = _follow_up(question, history, months)
    step = step or plan(question, months)
    if step is None:
        return ChatResponse(HELP_ANSWER)
    name, args = step
    if name not in {t.name for t in tools}:
        return ChatResponse(HELP_ANSWER)
    return ChatResponse("", [ToolCall(f"call_{len(messages)}", name, args)])
