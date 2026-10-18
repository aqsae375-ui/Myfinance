"""Ranked ways to save, built only from Stage 3 analytics. Every saving is a monthly figure computed
here, with the formula and its inputs attached (`calculation`, `numbers`); no LLM is involved.

Sources (all compared against the user's own history, so no saving rate is invented):
1. unused_subscription: a subscription flagged possibly_unused -> cancel it; saving = its full
   monthly charge (median charge, from detect_subscriptions).
2. category_trim: a discretionary category whose average month is above its typical (median) month,
   i.e. a few heavy months push the average up -> keep it to the typical month;
   saving = average monthly spend - median monthly spend (months with no spend count as 0).
   Needs MIN_TRIM_MONTHS months. Anomalies in the category are listed as evidence.
3. one_off_review: an anomaly in a non-discretionary category (e.g. a power bill spike) -> review it
   and plan to avoid a repeat; saving = its excess over the baseline / months of history,
   i.e. what that one-off cost per month on average.
Ranked by monthly saving, largest first; savings below MIN_MONTHLY_SAVING are dropped, except an
unused subscription (cancelling one is always worth suggesting).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.analytics import detect_anomalies, detect_subscriptions
from src.analytics.common import prepare

DISCRETIONARY = ("Dining Out", "Shopping", "Entertainment", "Groceries", "Transport")
# Moving money, fixed contracts and statutory deductions: not savings.
NO_REVIEW = frozenset({"Income", "Interest", "Transfers", "Transfers In", "Rent", "Tax"})
MIN_TRIM_MONTHS = 3
MIN_MONTHLY_SAVING = 10.0
MAX_OPPORTUNITIES = 5
_KIND_ORDER = {"unused_subscription": 0, "category_trim": 1, "one_off_review": 2}


def money(v: float) -> str:
    """Always 2 decimals, so the same string can be checked exactly in any model-written sentence."""
    return f"Rs {abs(v):,.2f}"


def _r(x: float) -> float:
    return round(float(x), 2)


@dataclass(frozen=True)
class Opportunity:
    id: str                       # stable key, e.g. "trim:Dining Out", "cancel:FITPRO GYM MEMBERSHIP"
    rank: int                     # 1 = largest monthly saving
    kind: str                     # unused_subscription | category_trim | one_off_review
    category: str
    target: str                   # merchant or category the action is about
    action: str                   # suggested action (deterministic)
    monthly_saving: float         # Rs per month, positive
    annual_saving: float          # monthly_saving * 12
    confidence: str               # high | medium | low
    calculation: str              # the formula with its inputs, for "How I got this"
    numbers: dict[str, float] = field(default_factory=dict)  # every figure in action/calculation
    dates: tuple[str, ...] = ()   # every ISO date/month in action/calculation
    source: str = ""              # which analytics function(s) produced the inputs
    txn_ids: tuple[int, ...] = ()


def _monthly_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """Debit totals: one row per month in the data, one column per category, 0 where nothing was spent."""
    months = sorted(df["month"].unique())
    debits = df[df["amount"] < 0]
    table = (-debits.pivot_table(index="month", columns="category", values="amount", aggfunc="sum"))
    return table.reindex(months).fillna(0.0)


def _subscription_opportunities(df: pd.DataFrame) -> list[dict]:
    out = []
    for s in detect_subscriptions(df):
        if not s.possibly_unused:
            continue
        saving = s.monthly_amount
        out.append(dict(
            id=f"cancel:{s.merchant}", kind="unused_subscription", category=s.category, target=s.merchant,
            action=f"Cancel {s.merchant} ({money(saving)} a month)",
            monthly_saving=saving, confidence="high",
            calculation=(f"Charged a median {money(saving)} a month ({s.charge_count} charges, "
                         f"{s.first_charge.isoformat()} to {s.last_charge.isoformat()}). Last related activity "
                         f"{s.last_activity.isoformat()}, {s.days_since_activity} days ago, so cancelling saves "
                         f"the full {money(saving)} a month."),
            numbers={"charge_count": s.charge_count, "days_since_activity": s.days_since_activity},
            dates=(s.first_charge.isoformat(), s.last_charge.isoformat(), s.last_activity.isoformat()),
            source="detect_subscriptions (possibly_unused)", txn_ids=s.txn_ids + s.activity_txn_ids,
        ))
    return out


def _trim_opportunities(df: pd.DataFrame, table: pd.DataFrame, anomalies) -> list[dict]:
    n = len(table)
    if n < MIN_TRIM_MONTHS:
        return []
    out = []
    for cat in DISCRETIONARY:
        if cat not in table.columns:
            continue
        col = table[cat].round(2)
        mean, median = _r(col.mean()), _r(col.median())
        saving = _r(mean - median)
        if saving < MIN_MONTHLY_SAVING:
            continue
        heavy = [m for m in col.index if col[m] > median]
        breakdown = ", ".join(f"{m} {money(v)}" for m, v in col.items())
        spikes = [a for a in anomalies if a.category == cat]
        evidence = "".join(f" Includes a one-off {money(a.amount)} at {a.merchant} on {a.date.isoformat()}."
                           for a in spikes)
        if median == 0:
            action = f"Plan big {cat} purchases ahead (a typical month has none)"
        else:
            action = f"Keep {cat} to your typical month of {money(median)}"
        numbers = {"average": mean, "median": median, "months": n}
        numbers |= {f"month_{m}": float(v) for m, v in col.items()}
        numbers |= {f"spike_{a.txn_id}": a.amount for a in spikes}
        rows = df[(df["amount"] < 0) & (df["category"] == cat)]
        out.append(dict(
            id=f"trim:{cat}", kind="category_trim", category=cat, target=cat, action=action,
            monthly_saving=saving, confidence="medium",
            calculation=(f"Average {cat} spend {money(mean)} a month over {n} months minus your typical "
                         f"(median) month {money(median)} = {money(saving)} a month. The months above typical "
                         f"({', '.join(heavy)}) push the average up. Monthly totals: {breakdown}.{evidence}"),
            numbers=numbers, dates=tuple(col.index) + tuple(a.date.isoformat() for a in spikes),
            source="monthly category debits (monthly_summary); detect_anomalies for evidence",
            txn_ids=tuple(int(i) for i in rows["id"]),
        ))
    return out


def _review_opportunities(n_months: int, anomalies) -> list[dict]:
    out = []
    for a in anomalies:
        if a.category in DISCRETIONARY or a.category in NO_REVIEW:
            continue  # discretionary spikes are evidence on their category trim
        saving = _r(a.excess / n_months)
        out.append(dict(
            id=f"review:{a.txn_id}", kind="one_off_review", category=a.category, target=a.merchant,
            action=f"Review the unusual {money(a.amount)} {a.merchant} charge and plan to avoid a repeat",
            monthly_saving=saving, confidence="low",
            calculation=(f"{money(a.amount)} at {a.merchant} on {a.date.isoformat()} was {money(a.excess)} above "
                         f"the usual {money(a.baseline)} ({a.multiple}x). Spread over {n_months} months of "
                         f"history, that one-off cost {money(saving)} a month."),
            numbers={"amount": a.amount, "excess": a.excess, "baseline": a.baseline,
                     "multiple": a.multiple, "months": n_months},
            dates=(a.date.isoformat(),), source="detect_anomalies", txn_ids=(a.txn_id,),
        ))
    return out


def find_opportunities(txns: pd.DataFrame | None = None,
                       max_items: int = MAX_OPPORTUNITIES) -> list[Opportunity]:
    """Top `max_items` ways to save, largest monthly saving first. `txns=None` loads the DB.
    Deterministic: the same transactions always give the same list."""
    df = prepare(txns)
    if df.empty:
        return []
    table = _monthly_by_category(df)
    anomalies = detect_anomalies(df)
    raw = (_subscription_opportunities(df) + _trim_opportunities(df, table, anomalies)
           + _review_opportunities(len(table), anomalies))
    raw = [o for o in raw if o["monthly_saving"] >= MIN_MONTHLY_SAVING or o["kind"] == "unused_subscription"]
    raw.sort(key=lambda o: (-o["monthly_saving"], _KIND_ORDER[o["kind"]], o["id"]))
    result = []
    for rank, o in enumerate(raw[:max_items], 1):
        monthly = _r(o.pop("monthly_saving"))
        numbers = {"monthly_saving": monthly, "annual_saving": _r(monthly * 12), **o.pop("numbers")}
        result.append(Opportunity(rank=rank, monthly_saving=monthly, annual_saving=_r(monthly * 12),
                                  numbers=numbers, **o))
    return result
