"""Rule-based insight sentences. Every number is computed upstream and formatted in here;
no LLM is involved. A later stage may let a model rephrase `text`, never change `numbers`.

Rules (per month):
- category change: |% change| > CHANGE_THRESHOLD_PCT and |Rs change| >= MIN_CHANGE vs last month
  (an increase is a warning, a decrease is a positive callout); skipped when anomaly excess explains
  >= ANOMALY_SHARE of it, so a spike is reported once and its non-repeat isn't praised next month
- possibly-unused subscription (impact = its annual cost, the saving from cancelling)
- anomaly dated in the month (impact = its excess over the baseline)
- cash share: in a month with an ATM/cash withdrawal, the year-to-date cash total and its share of all
  spending, when the share is at least MIN_CASH_SHARE_PCT (impact = cash withdrawn this month)
Ranked by impact, largest first, capped at MAX_INSIGHTS.
"""
from __future__ import annotations

import pandas as pd

from src.analytics.anomalies import detect_anomalies
from src.analytics.common import format_money, format_pct, month_end, prepare
from src.analytics.dashboard import cash_share
from src.analytics.monthly import month_over_month
from src.analytics.schema import Insight
from src.analytics.subscriptions import detect_subscriptions

CHANGE_THRESHOLD_PCT = 15.0
MIN_CHANGE = 25.0
MAX_INSIGHTS = 5
ANOMALY_SHARE = 0.5   # skip a category change when anomalies explain at least half of it
MIN_CASH_SHARE_PCT = 10.0
_KIND_ORDER = {"anomaly": 0, "category_increase": 1, "unused_subscription": 2, "cash_share": 3,
               "category_decrease": 4}


def _category_insights(month: str, df: pd.DataFrame) -> list[Insight]:
    mom = month_over_month(month, df)
    if mom.previous is None:
        return []
    # Anomaly excess per (month, category). If one-off spikes explain most of a change, the anomaly
    # insight already covers a rise, and a fall only means the one-off didn't repeat.
    excess: dict[tuple[str, str], float] = {}
    for a in detect_anomalies(df):
        key = (f"{a.date:%Y-%m}", a.category)
        excess[key] = excess.get(key, 0.0) + a.excess
    out = []
    for c in mom.categories:
        ch = c.amount
        if ch.abs_change is None or abs(ch.abs_change) < MIN_CHANGE:
            continue
        if ch.pct_change is not None and abs(ch.pct_change) <= CHANGE_THRESHOLD_PCT:
            continue
        spike_month = month if ch.abs_change > 0 else mom.previous_month
        if excess.get((spike_month, c.category), 0.0) >= ANOMALY_SHARE * abs(ch.abs_change):
            continue
        cur_total = mom.current.category(c.category)
        numbers = {"current": ch.current, "previous": ch.previous, "change": ch.abs_change}
        if ch.pct_change is None:  # nothing spent last month
            text = (f"{c.category} is new this month: {format_money(ch.current)} spent, "
                    f"nothing last month.")
        elif ch.abs_change > 0:
            numbers["pct_change"] = ch.pct_change
            text = (f"{c.category} is up {format_pct(ch.pct_change)} vs last month, "
                    f"an extra {format_money(ch.abs_change)}.")
        else:
            numbers["pct_change"] = ch.pct_change
            text = (f"Nice work: {c.category} is down {format_pct(ch.pct_change)} vs last month, "
                    f"{format_money(ch.abs_change)} less.")
        out.append(Insight(
            month=month, kind="category_increase" if ch.abs_change > 0 else "category_decrease",
            text=text, impact=abs(ch.abs_change), numbers=numbers,
            source=f"{c.category} debits in {month} vs {mom.previous_month} (monthly.month_over_month)",
            txn_ids=cur_total.txn_ids if cur_total else (),
        ))
    return out


def _subscription_insights(month: str, df: pd.DataFrame) -> list[Insight]:
    as_of = min(pd.Timestamp(month_end(month)), df["date"].max())
    out = []
    for s in detect_subscriptions(df, as_of=as_of):
        if not s.possibly_unused:
            continue
        out.append(Insight(
            month=month, kind="unused_subscription",
            text=(f"{s.merchant} may be unused: no related activity in {s.days_since_activity} days, "
                  f"but it still costs {format_money(s.monthly_amount)} a month "
                  f"({format_money(s.annual_cost)} a year)."),
            impact=s.annual_cost,
            numbers={"days_since_activity": s.days_since_activity, "monthly_amount": s.monthly_amount,
                     "annual_cost": s.annual_cost},
            source=f"recurring charges + related activity up to {as_of:%Y-%m-%d} (subscriptions.detect_subscriptions)",
            txn_ids=s.txn_ids + s.activity_txn_ids,
        ))
    return out


def _anomaly_insights(month: str, df: pd.DataFrame) -> list[Insight]:
    out = []
    for a in detect_anomalies(df, month=month):
        if "category_spike" in a.reasons:
            text = (f"Unusual charge: {a.merchant} for {format_money(a.amount)} on {a.date:%d %b}, "
                    f"{a.multiple:.1f}x your average {a.category} transaction ({format_money(a.baseline)}).")
        else:
            text = (f"Unusual charge: {format_money(a.amount)} at a new merchant, {a.merchant}, "
                    f"on {a.date:%d %b}.")
        out.append(Insight(
            month=month, kind="anomaly", text=text, impact=a.excess,
            numbers={"amount": a.amount, "baseline": a.baseline, "multiple": a.multiple, "excess": a.excess},
            source=a.explanation, txn_ids=(a.txn_id,),
        ))
    return out


def _cash_insights(month: str, df: pd.DataFrame) -> list[Insight]:
    c = cash_share(month, df)
    if not c.month_cash or c.pct_of_spend is None or c.pct_of_spend < MIN_CASH_SHARE_PCT:
        return []
    start, end = pd.Period(c.start_month, freq="M"), pd.Period(month, freq="M")
    period = (f"in {end.strftime('%b %Y')}" if start == end
              else f"from {start.strftime('%b')} to {end.strftime('%b %Y')}")
    return [Insight(
        month=month, kind="cash_share",
        text=(f"{format_money(c.cash)} ({format_pct(c.pct_of_spend)} of your spending) left your account as cash "
              f"{period}. Cash spending is not tracked after withdrawal."),
        impact=c.month_cash,
        numbers={"cash": c.cash, "spent": c.spent, "pct_of_spend": c.pct_of_spend, "month_cash": c.month_cash},
        source=f"Cash Withdrawal debits / all debits, {c.start_month} to {month} (dashboard.cash_share)",
        txn_ids=c.txn_ids,
    )]


def generate_insights(month: str, txns: pd.DataFrame | None = None, limit: int = MAX_INSIGHTS) -> list[Insight]:
    """Up to `limit` insights for `month`, largest rupee impact first."""
    df = prepare(txns)
    candidates = (_category_insights(month, df) + _subscription_insights(month, df) + _anomaly_insights(month, df)
                  + _cash_insights(month, df))
    candidates.sort(key=lambda i: (-i.impact, _KIND_ORDER[i.kind], i.text))
    return candidates[:limit]
