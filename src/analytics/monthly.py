"""Monthly totals and month-over-month change. Pure Pandas; no model is involved.

Definitions (also returned in `MonthlySummary.method`):
- total_spent  = sum of all debits (money out) in the month, including transfers and fees
- total_income = credits that are income: Income (salary-type), Interest (profit paid) and Transfers In
                 (money received from people/wallets). Not-yet-categorized credits count too, so a fresh
                 upload isn't shown with zero income. Other credits (refunds, reversals) are
                 `other_credits`: reported, but neither income nor negative spending.
- net_saved    = total_income - total_spent
- category totals are debits only; pct_of_spend = category / total_spent * 100
"""
from __future__ import annotations

import pandas as pd

from src.analytics.common import UNCATEGORIZED, check_month, prepare, previous_month
from src.categories import INCOME_CATEGORIES
from src.analytics.schema import CategoryChange, CategoryTotal, Change, MonthComparison, MonthlySummary

METHOD = ("total_spent = sum of debits; total_income = sum of credits in Income, Interest and Transfers In; "
          "net_saved = income - spent; category share = category debits / total_spent")


def _r(x: float) -> float:
    return round(float(x), 2)


def all_months(txns: pd.DataFrame | None = None) -> list[str]:
    """Months present in the data, oldest first, as 'YYYY-MM'."""
    return sorted(prepare(txns)["month"].unique().tolist())


def _summary(df: pd.DataFrame, month: str) -> MonthlySummary:
    rows = df[df["month"] == month]
    if rows.empty:
        raise ValueError(f"no transactions for {month}")
    debits = rows[rows["amount"] < 0]
    spent = _r(-debits["amount"].sum())
    credits = rows[rows["amount"] > 0]
    is_income = credits["category"].isin(INCOME_CATEGORIES | {UNCATEGORIZED})
    income = _r(credits.loc[is_income, "amount"].sum())
    other_credits = _r(credits.loc[~is_income, "amount"].sum())

    cats = []
    for cat, g in debits.groupby("category"):
        amount = _r(-g["amount"].sum())
        cats.append(CategoryTotal(
            category=cat, amount=amount, pct_of_spend=_r(amount / spent * 100) if spent else 0.0,
            txn_count=len(g), txn_ids=tuple(int(i) for i in g["id"]),
        ))
    cats.sort(key=lambda c: (-c.amount, c.category))
    return MonthlySummary(month=month, total_spent=spent, total_income=income,
                          net_saved=_r(income - spent), txn_count=len(rows),
                          categories=tuple(cats), method=METHOD, other_credits=other_credits)


def monthly_summary(month: str, txns: pd.DataFrame | None = None) -> MonthlySummary:
    """Totals for one month. `txns=None` loads from the DB. Raises ValueError for a month with no data."""
    return _summary(prepare(txns), check_month(month))


def change(current: float, previous: float | None) -> Change:
    """abs and % change. % is relative to |previous| so a smaller deficit reads as an improvement."""
    if previous is None:
        return Change(current=current, previous=None, abs_change=None, pct_change=None)
    delta = _r(current - previous)
    pct = _r(delta / abs(previous) * 100) if previous else None
    return Change(current=current, previous=previous, abs_change=delta, pct_change=pct)


def _compare(df: pd.DataFrame, month: str, base: str | None) -> MonthComparison:
    """`month` vs `base`. If `base` is None or has no data, every Change has previous = None."""
    cur = _summary(df, month)
    prev = _summary(df, base) if base is not None and (df["month"] == base).any() else None

    names = [c.category for c in cur.categories]
    if prev:
        names += [c.category for c in prev.categories if c.category not in names]
    cats = []
    for name in names:
        c, p = cur.category(name), prev.category(name) if prev else None
        prev_amount = (p.amount if p else 0.0) if prev else None
        prev_pct = (p.pct_of_spend if p else 0.0) if prev else None
        cats.append(CategoryChange(
            category=name,
            amount=change(c.amount if c else 0.0, prev_amount),
            pct_of_spend=change(c.pct_of_spend if c else 0.0, prev_pct),
        ))

    def pick(attr: str) -> Change:
        return change(getattr(cur, attr), getattr(prev, attr) if prev else None)

    return MonthComparison(
        month=month, previous_month=base if prev else None,
        total_spent=pick("total_spent"), total_income=pick("total_income"), net_saved=pick("net_saved"),
        categories=tuple(cats), current=cur, previous=prev,
    )


def previous_data_month(months: list[str], month: str) -> str | None:
    """The latest month before `month` that has data (skips gaps), or None for the first month."""
    earlier = [m for m in months if m < month]
    return earlier[-1] if earlier else None


def month_over_month(month: str, txns: pd.DataFrame | None = None, previous: str = "calendar") -> MonthComparison:
    """`month` vs the month before it. previous="calendar" (default): the calendar month before; if that
    month has no data (e.g. the first month, or a gap), `previous_month` is None and every Change has
    previous/abs_change/pct_change = None. previous="data": the latest earlier month that has data
    (the dashboard uses this, so a gap does not hide the comparison)."""
    df = prepare(txns)
    month = check_month(month)
    prev = previous_data_month(sorted(df["month"].unique().tolist()), month) if previous == "data"         else previous_month(month)
    return _compare(df, month, prev)


def compare_months(month_a: str, month_b: str, txns: pd.DataFrame | None = None) -> MonthComparison:
    """Any two months: changes are `month_b` relative to `month_a` (so `month` = month_b and
    `previous_month` = month_a in the result). Raises ValueError if either month has no data."""
    df = prepare(txns)
    check_month(month_a)
    _summary(df, month_a)  # fail loudly on a missing base month instead of returning previous=None
    return _compare(df, check_month(month_b), month_a)
