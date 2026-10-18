"""Plain-English sentences built directly from tool results, with no arithmetic: every number printed
is a field of the result, copied exactly. Used as the fake model's answers and as the agent's
fallback when a model's answer fails number verification.
"""
from __future__ import annotations

from typing import Any, Callable

WITHHELD = "[description withheld: it contains instruction-like text]"
TOP_CATEGORIES = 3


def money(v: float) -> str:
    return f"Rs {abs(v):,.2f}"


def pct(v: float) -> str:
    return f"{abs(round(v, 2))}%"


def _direction(change: dict, unit: Callable[[float], str] = money) -> str:
    """'up Rs 1,228.15 (39.3%)' from a Change dict; 'unchanged' when abs_change is 0."""
    delta = change["abs_change"]
    if not delta:
        return "unchanged"
    text = f"{'up' if delta > 0 else 'down'} {unit(delta)}"
    return text + (f" ({pct(change['pct_change'])})" if change["pct_change"] is not None else "")


def _saved(net: float) -> str:
    return f"you saved {money(net)}" if net >= 0 else f"you spent {money(net)} more than you received"


def _months(d: dict) -> str:
    return f"Your statement covers {d['earliest']} to {d['latest']}."


def _summary(d: dict) -> str:
    top = ", ".join(f"{c['category']} {money(c['amount'])} ({pct(c['pct_of_spend'])})"
                    for c in d["categories"][:TOP_CATEGORIES])
    text = (f"In {d['month']} you spent {money(d['total_spent'])} and received {money(d['total_income'])}, "
            f"so {_saved(d['net_saved'])}.")
    return text + (f" Biggest categories: {top}." if top else "")


def _comparison(d: dict) -> str:
    spent, base = d["total_spent"], d["compared_with"]
    if base is None:
        return f"In {d['month']} you spent {money(spent['current'])}; there is no earlier month to compare it with."
    text = (f"In {d['month']} you spent {money(spent['current'])}, {_direction(spent)} from "
            f"{money(spent['previous'])} in {base}. Net saved was {money(d['net_saved']['current'])}"
            f"{' (a deficit)' if d['net_saved']['current'] < 0 else ''}, {_direction(d['net_saved'])}.")
    changed = [c for c in d["categories"] if c["abs_change"]]
    if changed:
        big = max(changed, key=lambda c: abs(c["abs_change"]))
        text += (f" Biggest change: {big['category']}, {_direction(big)} to {money(big['current'])} "
                 f"from {money(big['previous'])}.")
    return text


def _category(d: dict) -> str:
    if not d["txn_count"]:
        return f"You spent nothing on {d['category']} in {d['month']}."
    return (f"You spent {money(d['amount'])} on {d['category']} in {d['month']} across {d['txn_count']} "
            f"transaction{'s' if d['txn_count'] != 1 else ''} ({pct(d['pct_of_spend'])} of that month's spending).")


def _transactions(d: dict) -> str:
    if not d["match_count"]:
        return "No transactions matched those filters."
    text = (f"Found {d['match_count']} matching transaction{'s' if d['match_count'] != 1 else ''} "
            f"(money out {money(d['total_out'])}, money in {money(d['total_in'])}).")
    if d["shown"] < d["match_count"]:
        text += f" Showing {d['shown']}:"
    for t in d["transactions"]:
        label = WITHHELD if t.get("flag") else t["merchant"]
        flow = "out" if t["amount"] < 0 else "in"
        text += f"\n- {t['date']}  {label}  ({t['category']})  {money(t['amount'])} {flow}"
    return text


def _subscriptions(d: dict) -> str:
    if not d["count"]:
        return "I didn't find any active subscriptions."
    text = f"You have {d['count']} active subscription{'s' if d['count'] != 1 else ''}:"
    for s in d["subscriptions"]:
        text += f"\n- {s['merchant']}: {money(s['monthly_amount'])}/month ({money(s['annual_cost'])}/year)"
        if s["possibly_unused"]:
            text += (f", possibly unused: no related activity for {s['days_since_activity']} days "
                     f"(last on {s['last_activity']}) but still charged")
    return text


def _anomalies(d: dict) -> str:
    where = f" in {d['month']}" if d["month"] else ""
    if not d["count"]:
        return f"I found no unusual charges{where}."
    text = f"I found {d['count']} unusual charge{'s' if d['count'] != 1 else ''}{where}:"
    for a in d["anomalies"]:
        text += (f"\n- {a['date']}  {a['merchant']}  ({a['category']})  {money(a['amount'])}, "
                 f"{a['multiple']}x the usual {money(a['baseline'])}, {money(a['excess'])} above it")
    return text


def _cash(d: dict) -> str:
    text = (f"In {d['month']} you withdrew {money(d['withdrawn'])} in cash and logged {money(d['logged'])} of "
            f"cash spending, so {money(d['unaccounted'])} is unaccounted")
    text += f" ({pct(d['accounted_pct'])} accounted for)." if d["accounted_pct"] is not None else "."
    if d["cash_from_outside_account"]:
        text += f" {money(d['cash_from_outside_account'])} was cash from outside the account."
    balance = d["wallet_balance"]
    return text + (f" By your records the cash wallet balance is {money(balance)}." if balance >= 0 else
                   f" You logged {money(balance)} more cash than you withdrew so far.")


RENDERERS: dict[str, Callable[[dict], str]] = {
    "list_available_months": _months, "get_monthly_summary": _summary, "get_month_over_month": _comparison,
    "compare_periods": _comparison, "get_category_spending": _category, "find_transactions": _transactions,
    "get_subscriptions": _subscriptions, "get_anomalies": _anomalies, "get_cash_summary": _cash,
}


def render(tool_name: str, data: dict[str, Any]) -> str:
    if "error" in data:
        return f"I couldn't get that: {data['error']}."
    fn = RENDERERS.get(tool_name)
    return fn(data) if fn else ""
