"""Figures the dashboard needs beyond monthly.py: month history with savings rate, top merchants,
daily spending and the categorized ledger. Pure Pandas; no model is involved.

Definitions:
- savings_rate = net_saved / total_income * 100 (None when income is 0, never a division by zero)
- month-history changes are vs the calendar month before, as in month_over_month; the savings-rate
  change is in percentage points
- top merchants and daily spending count debits (money out) only
"""
from __future__ import annotations

import pandas as pd

from src.analytics.common import check_month, prepare, previous_month
from src.analytics.monthly import _summary, change
from src.analytics.schema import CashShare, Change, DailySpending, DaySpend, LedgerRow, MerchantTotal, MonthRow


def _r(x: float) -> float:
    return round(float(x), 2)


def savings_rate(total_income: float, net_saved: float) -> float | None:
    return _r(net_saved / total_income * 100) if total_income > 0 else None


def _rate_change(current: float | None, previous: float | None, has_previous: bool) -> Change:
    if current is None or previous is None or not has_previous:
        return Change(current=current if current is not None else 0.0, previous=None, abs_change=None,
                      pct_change=None)
    return Change(current=current, previous=previous, abs_change=_r(current - previous), pct_change=None)


def month_history(txns: pd.DataFrame | None = None, previous: str = "calendar") -> list[MonthRow]:
    """One row per month with data, oldest first. Changes are vs the calendar month before (default) or,
    with previous="data", vs the latest earlier month that has data (see monthly.month_over_month)."""
    df = prepare(txns)
    months = sorted(df["month"].unique().tolist())
    sums = {m: _summary(df, m) for m in months}
    rows = []
    for i, m in enumerate(months):
        cur = sums[m]
        prev = sums.get(months[i - 1] if i else None) if previous == "data" else sums.get(previous_month(m))
        rate = savings_rate(cur.total_income, cur.net_saved)
        prev_rate = savings_rate(prev.total_income, prev.net_saved) if prev else None
        rows.append(MonthRow(
            month=m, total_spent=cur.total_spent, total_income=cur.total_income, net_saved=cur.net_saved,
            savings_rate=rate,
            spent=change(cur.total_spent, prev.total_spent if prev else None),
            income=change(cur.total_income, prev.total_income if prev else None),
            net=change(cur.net_saved, prev.net_saved if prev else None),
            rate=_rate_change(rate, prev_rate, prev is not None),
        ))
    return rows


def top_merchants(month: str, n: int = 5, txns: pd.DataFrame | None = None) -> list[MerchantTotal]:
    """The `n` merchants with the most money out in `month`, largest first."""
    df = prepare(txns)
    debits = df[(df["month"] == check_month(month)) & (df["amount"] < 0)]
    out = [MerchantTotal(merchant=m, category=g["category"].mode().iloc[0], amount=_r(-g["amount"].sum()),
                         txn_count=len(g), txn_ids=tuple(int(i) for i in g["id"]))
           for m, g in debits.groupby("merchant")]
    out.sort(key=lambda t: (-t.amount, t.merchant))
    return out[:n]


def daily_spending(month: str, txns: pd.DataFrame | None = None) -> DailySpending:
    """Money out per calendar day of `month` (zero-spend days included) and the peak day."""
    df = prepare(txns)
    debits = df[(df["month"] == check_month(month)) & (df["amount"] < 0)]
    by_day = debits.groupby(debits["date"].dt.date)["amount"].agg(["sum", "count"])
    period = pd.Period(month, freq="M")
    days = []
    for d in pd.date_range(period.start_time, period.end_time.normalize(), freq="D").date:
        total, count = (by_day.loc[d, "sum"], by_day.loc[d, "count"]) if d in by_day.index else (0.0, 0)
        days.append(DaySpend(date=d, amount=_r(-total) if total else 0.0, txn_count=int(count)))
    spent = [d for d in days if d.amount > 0]
    peak = max(spent, key=lambda d: (d.amount, -d.date.toordinal())) if spent else None
    return DailySpending(month=month, days=tuple(days), peak=peak)


def month_transactions(month: str, category: str | None = None,
                       txns: pd.DataFrame | None = None) -> list[LedgerRow]:
    """Every transaction in `month` (optionally one category), newest first."""
    df = prepare(txns)
    rows = df[df["month"] == check_month(month)]
    if category is not None:
        rows = rows[rows["category"] == category]
    rows = rows.sort_values(["date", "id"], ascending=[False, False])
    source = rows["category_source"] if "category_source" in rows else pd.Series(None, index=rows.index)
    conf = rows["confidence"] if "confidence" in rows else pd.Series(None, index=rows.index)
    review = rows["needs_review"] if "needs_review" in rows else pd.Series(0, index=rows.index)
    return [LedgerRow(txn_id=int(r.id), date=r.date.date(), merchant=r.merchant, description=r.redacted_description,
                      category=r.category, amount=_r(r.amount),
                      source=s if isinstance(s, str) else None,
                      confidence=None if c is None or pd.isna(c) else float(c),
                      needs_review=bool(v) if not pd.isna(v) else False)
            for r, s, c, v in zip(rows.itertuples(), source, conf, review)]


CASH = "Cash Withdrawal"


def cash_share(month: str, txns: pd.DataFrame | None = None) -> CashShare:
    """Cash withdrawals as a share of spending, year to date (January of `month`'s year up to `month`).
    Cash spending can't be tracked after withdrawal, so this is the part of spending the app can't see."""
    df = prepare(txns)
    check_month(month)
    ytd = df[(df["month"] >= f"{month[:4]}-01") & (df["month"] <= month) & (df["amount"] < 0)]
    cash_rows = ytd[ytd["category"] == CASH]
    spent = _r(-ytd["amount"].sum())
    cash = _r(-cash_rows["amount"].sum())
    return CashShare(month=month, start_month=ytd["month"].min() if not ytd.empty else month, cash=cash,
                     spent=spent, pct_of_spend=_r(cash / spent * 100) if spent else None,
                     month_cash=_r(-cash_rows.loc[cash_rows["month"] == month, "amount"].sum()),
                     txn_ids=tuple(int(i) for i in cash_rows["id"]))
