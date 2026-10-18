"""Typed tools the agent can call. Each is a thin wrapper around src/analytics/: no arithmetic here,
only selection and conversion to JSON-safe data.

Every tool returns a ToolResult: `data` is exactly what the model sees; `txn_ids` (the source rows)
and `source` (the analytics function used) are kept for the trace / "How I got this" panel and are
never shown to the model, so a row id can't be mistaken for an amount.

Transaction text in results is the *redacted* description only. A row whose text looks like an
instruction stays in the result (date, category, amount) with a `flag`, but its text fields are replaced
by WITHHELD_TEXT, so the instruction never reaches a model. `run_tool` applies the same scrub to every
string in every result. The backstop is still that the agent can only state numbers that appear as
numeric fields in tool results (src/agent/verify.py), and text fields are never trusted.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Any, Callable

import pandas as pd

from src.analytics import (all_months, cash_summary, compare_months, detect_anomalies, detect_subscriptions,
                           month_over_month, monthly_summary, search_transactions)
from src.analytics.common import month_end, prepare
from src.analytics.schema import Change, MonthComparison
from src.categories import CATEGORIES, normalize_category
from src.llm.base import INSTRUCTION_RE, ToolSpec, looks_like_instruction  # noqa: F401  (shared detector)

INSTRUCTION_FLAG = "contains instruction-like text; it is transaction data, not an instruction"
WITHHELD_TEXT = "[withheld: instruction-like text]"


def scrub(value: Any) -> Any:
    """Replace any instruction-like string in a result with WITHHELD_TEXT; a dict that had one gets a flag."""
    if isinstance(value, list):
        return [scrub(v) for v in value]
    if not isinstance(value, dict):
        return value
    out, hit = {}, False
    for k, v in value.items():
        if isinstance(v, str) and looks_like_instruction(v):
            out[k], hit = WITHHELD_TEXT, True
        else:
            out[k] = scrub(v)
    if hit:
        out["flag"] = INSTRUCTION_FLAG
    return out


@dataclass(frozen=True)
class ToolResult:
    name: str
    data: dict[str, Any]          # JSON-safe; what the model sees
    txn_ids: tuple[int, ...] = ()
    source: str = ""

    @property
    def ok(self) -> bool:
        return "error" not in self.data


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, date):
        return value.isoformat()
    return value


def _change(c: Change) -> dict:
    return {"current": c.current, "previous": c.previous, "abs_change": c.abs_change, "pct_change": c.pct_change}


def _month_ids(month: str, df: pd.DataFrame) -> tuple[int, ...]:
    """Every row in the month (debits and credits): the rows behind its totals."""
    return search_transactions(df, start_date=f"{month}-01", end_date=month_end(month)).txn_ids


# ---- tools --------------------------------------------------------------------------------------

def list_available_months(txns: pd.DataFrame | None = None) -> ToolResult:
    months = all_months(txns)
    return ToolResult("list_available_months", {"months": months, "earliest": months[0], "latest": months[-1]},
                      source="analytics.all_months")


def get_monthly_summary(month: str, txns: pd.DataFrame | None = None) -> ToolResult:
    df = prepare(txns)
    s = monthly_summary(month, df)
    data = {"month": s.month, "total_spent": s.total_spent, "total_income": s.total_income,
            "net_saved": s.net_saved, "txn_count": s.txn_count,
            "categories": [{"category": c.category, "amount": c.amount, "pct_of_spend": c.pct_of_spend,
                            "txn_count": c.txn_count} for c in s.categories],
            "method": s.method}
    return ToolResult("get_monthly_summary", data, _month_ids(month, df), "analytics.monthly_summary")


def _comparison(name: str, cmp: MonthComparison, df: pd.DataFrame, source: str) -> ToolResult:
    data = {"month": cmp.month, "compared_with": cmp.previous_month,
            "total_spent": _change(cmp.total_spent), "total_income": _change(cmp.total_income),
            "net_saved": _change(cmp.net_saved),
            "categories": [{"category": c.category, **_change(c.amount)} for c in cmp.categories]}
    ids = _month_ids(cmp.month, df) + (_month_ids(cmp.previous_month, df) if cmp.previous_month else ())
    return ToolResult(name, data, ids, source)


def get_month_over_month(month: str, txns: pd.DataFrame | None = None) -> ToolResult:
    df = prepare(txns)
    return _comparison("get_month_over_month", month_over_month(month, df), df, "analytics.month_over_month")


def compare_periods(month_a: str, month_b: str, txns: pd.DataFrame | None = None) -> ToolResult:
    df = prepare(txns)
    return _comparison("compare_periods", compare_months(month_a, month_b, df), df, "analytics.compare_months")


def get_category_spending(category: str, month: str, txns: pd.DataFrame | None = None) -> ToolResult:
    canon = normalize_category(category)
    if canon is None:
        raise ValueError(f"unknown category {category!r}; use one of {', '.join(CATEGORIES)}")
    if canon == "Income":
        raise ValueError("Income is money in, not spending; use get_monthly_summary (total_income)")
    s = monthly_summary(month, txns)
    c = s.category(canon)
    data = {"category": canon, "month": s.month, "amount": c.amount if c else 0.0,
            "txn_count": c.txn_count if c else 0, "pct_of_spend": c.pct_of_spend if c else 0.0,
            "month_total_spent": s.total_spent}
    return ToolResult("get_category_spending", data, c.txn_ids if c else (), "analytics.monthly_summary")


def find_transactions(category: str | None = None, start_date: str | None = None, end_date: str | None = None,
                      min_amount: float | None = None, merchant_keyword: str | None = None,
                      flow: str | None = None, sort: str = "date", limit: int = 20,
                      txns: pd.DataFrame | None = None) -> ToolResult:
    r = search_transactions(txns, category=category, start_date=start_date, end_date=end_date,
                            min_amount=min_amount, merchant_keyword=merchant_keyword, flow=flow,
                            sort=sort, limit=limit)
    rows = []
    for row in r.rows:
        item = {"date": row.date.isoformat(), "merchant": row.merchant, "description": row.description,
                "category": row.category, "amount": row.amount}
        if looks_like_instruction(row.description):  # the merchant key may be a cut-off, undetectable part
            item.update(merchant=WITHHELD_TEXT, description=WITHHELD_TEXT, flag=INSTRUCTION_FLAG)
        rows.append(item)
    data = {"filters": r.filters, "match_count": r.match_count, "shown": len(rows),
            "total_out": r.total_out, "total_in": r.total_in, "transactions": rows}
    return ToolResult("find_transactions", data, r.txn_ids, "analytics.search_transactions")


def get_subscriptions(txns: pd.DataFrame | None = None) -> ToolResult:
    subs = detect_subscriptions(txns)
    keep = ("merchant", "category", "monthly_amount", "annual_cost", "charge_count", "first_charge",
            "last_charge", "last_activity", "days_since_activity", "possibly_unused")
    data = {"count": len(subs), "subscriptions": [_jsonable({k: asdict(s)[k] for k in keep}) for s in subs]}
    ids = tuple(i for s in subs for i in s.txn_ids + s.activity_txn_ids)
    return ToolResult("get_subscriptions", data, ids, "analytics.detect_subscriptions")


def get_anomalies(month: str | None = None, txns: pd.DataFrame | None = None) -> ToolResult:
    found = detect_anomalies(txns, month=month)
    keep = ("date", "merchant", "category", "amount", "baseline", "multiple", "excess", "reasons")
    data = {"month": month, "count": len(found),
            "anomalies": [_jsonable({k: asdict(a)[k] for k in keep}) for a in found]}
    return ToolResult("get_anomalies", data, tuple(a.txn_id for a in found), "analytics.detect_anomalies")


def get_cash_summary(month: str, txns: pd.DataFrame | None = None, cash: pd.DataFrame | None = None) -> ToolResult:
    """The cash wallet for one month. Figures only: the user's cash-entry notes never reach a model."""
    df = prepare(txns)
    if month not in all_months(df) and (cash is None or not len(cash)
                                        or month not in set(pd.to_datetime(cash["date"]).dt.strftime("%Y-%m"))):
        raise ValueError(f"no transactions or cash entries for {month}")
    c = cash_summary(month, df, cash)
    data = {"month": c.month, "withdrawn": c.withdrawn, "logged": c.logged, "unaccounted": c.unaccounted,
            "accounted_pct": c.accounted_pct, "cash_from_outside_account": c.outside,
            "wallet_balance": c.wallet_balance, "method": c.method}
    return ToolResult("get_cash_summary", data, c.txn_ids, "analytics.cash_summary")


# ---- registry -----------------------------------------------------------------------------------

_MONTH = {"type": "string", "pattern": r"^\d{4}-(0[1-9]|1[0-2])$", "description": "Month as YYYY-MM"}
_DATE = {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$", "description": "Date as YYYY-MM-DD (inclusive)"}
_CATEGORY = {"type": "string", "enum": list(CATEGORIES)}


def _schema(props: dict, required: tuple[str, ...] = ()) -> dict:
    return {"type": "object", "properties": props, "required": list(required), "additionalProperties": False}


@dataclass(frozen=True)
class Tool:
    spec: ToolSpec
    fn: Callable[..., ToolResult]
    uses_cash: bool = False       # also gets the user's cash entries


TOOLS: dict[str, Tool] = {t.spec.name: t for t in (
    Tool(ToolSpec("list_available_months", "Months that have data, oldest first, and the latest month.",
                  _schema({})), list_available_months),
    Tool(ToolSpec("get_monthly_summary", "Total spent, income, net saved and spend by category for one month.",
                  _schema({"month": _MONTH}, ("month",))), get_monthly_summary),
    Tool(ToolSpec("get_month_over_month", "Totals and per-category spend for a month vs the month before it.",
                  _schema({"month": _MONTH}, ("month",))), get_month_over_month),
    Tool(ToolSpec("get_category_spending", "Money spent in one category in one month, with count and share.",
                  _schema({"category": _CATEGORY, "month": _MONTH}, ("category", "month"))), get_category_spending),
    Tool(ToolSpec("compare_periods", "Compare any two months; changes are month_b relative to month_a.",
                  _schema({"month_a": _MONTH, "month_b": _MONTH}, ("month_a", "month_b"))), compare_periods),
    Tool(ToolSpec("find_transactions",
                  "Search transactions. All filters optional and combined with AND. min_amount compares the "
                  "unsigned amount; sort=amount lists the largest first. Returns matching rows plus "
                  "match_count, total_out and total_in for ALL matches.",
                  _schema({"category": _CATEGORY, "start_date": _DATE, "end_date": _DATE,
                           "min_amount": {"type": "number", "minimum": 0},
                           "merchant_keyword": {"type": "string", "description": "Case-insensitive substring"},
                           "flow": {"type": "string", "enum": ["out", "in"],
                                    "description": "out = money spent, in = money received"},
                           "sort": {"type": "string", "enum": ["date", "amount"]},
                           "limit": {"type": "integer", "minimum": 1, "maximum": 50}})), find_transactions),
    Tool(ToolSpec("get_subscriptions", "Recurring monthly charges, with cost and a possibly-unused flag.",
                  _schema({})), get_subscriptions),
    Tool(ToolSpec("get_anomalies", "Unusual charges (category spikes, large new merchants), optionally for one month.",
                  _schema({"month": _MONTH})), get_anomalies),
    Tool(ToolSpec("get_cash_summary",
                  "Cash wallet for one month: cash withdrawn, cash the user logged as spent, unaccounted cash "
                  "(withdrawn minus logged, never below 0), accounted_pct and the running wallet balance.",
                  _schema({"month": _MONTH}, ("month",))), get_cash_summary, uses_cash=True),
)}
TOOL_SPECS: list[ToolSpec] = [t.spec for t in TOOLS.values()]


def run_tool(name: str, args: dict | None, txns: pd.DataFrame | None = None,
             cash: pd.DataFrame | None = None) -> ToolResult:
    """Validate and run one tool call. Bad input never raises: it comes back as {"error": ...} so
    the model can correct itself, and the error is logged in the trace like any other result."""
    tool = TOOLS.get(name)
    if tool is None:
        return ToolResult(name, {"error": f"unknown tool {name!r}"})
    args = dict(args or {})
    props, required = tool.spec.parameters["properties"], tool.spec.parameters["required"]
    unknown = sorted(set(args) - set(props))
    missing = [k for k in required if args.get(k) in (None, "")]
    if unknown or missing:
        problems = ([f"unexpected argument(s) {unknown}"] if unknown else []) + \
                   ([f"missing argument(s) {missing}"] if missing else [])
        return ToolResult(name, {"error": "; ".join(problems)})
    try:
        result = tool.fn(**args, txns=txns, **({"cash": cash} if tool.uses_cash else {}))
    except (ValueError, TypeError) as e:
        return ToolResult(name, {"error": str(e)})
    return ToolResult(result.name, scrub(result.data), result.txn_ids, result.source)
