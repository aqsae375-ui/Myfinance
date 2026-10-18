"""Recurring-charge detection and the "possibly unused" flag.

A merchant is a subscription when it has >= MIN_CHARGES debits of a similar amount (within
AMOUNT_TOLERANCE of the median) about a month apart, and it is still being charged.

"Possibly unused": a bank statement cannot see whether you watch Netflix, so we only flag a
subscription when we *can* see usage: related non-charge activity at the same brand (e.g. the
gym's cafe) that happened before but has stopped for UNUSED_AFTER_DAYS+ while charges continue.
Flagging every subscription with no visible usage would flag every streaming service.
"""
from __future__ import annotations

import re
from datetime import date

import pandas as pd

from src.analytics.common import prepare
from src.analytics.schema import Subscription

MIN_CHARGES = 3
AMOUNT_TOLERANCE = 0.10          # charge within 10% of the median charge
CADENCE_DAYS = (25, 35)          # median gap between charges must be roughly monthly
MAX_GAP_DAYS = 45                # no single gap longer than this (a skipped month breaks the pattern)
MIN_RECURRING_SHARE = 0.75       # >= 75% of the merchant's debits must be the recurring charge
STILL_ACTIVE_DAYS = 45           # last charge must be this recent, or the subscription has ended
UNUSED_AFTER_DAYS = 60
# Recurring but not subscriptions: rent, money moving, monthly profit and its withholding tax, bank
# charges, and regular ATM withdrawals (never flagged as an "unused subscription").
NOT_SUBSCRIPTIONS = frozenset({"Rent", "Transfers", "Transfers In", "Income", "Interest", "Tax", "Bank Fees",
                               "Cash Withdrawal"})
BRAND_STOPWORDS = frozenset({"THE", "INTL", "PAYMENT", "ONLINE", "POS", "DEBIT", "PURCHASE"})


def _brand(merchant: str) -> str | None:
    """First meaningful word of the merchant key: "FITPRO GYM MEMBERSHIP" -> "FITPRO"."""
    for word in re.findall(r"[A-Z]+", merchant):
        if len(word) >= 3 and word not in BRAND_STOPWORDS:
            return word
    return None


def _to_date(value) -> date:
    return pd.Timestamp(value).date()


def detect_subscriptions(txns: pd.DataFrame | None = None, as_of: str | date | None = None) -> list[Subscription]:
    """Subscriptions as seen on `as_of` (default: the latest transaction date); only data up to
    `as_of` is used. Sorted by annual cost, largest first."""
    df = prepare(txns)
    as_of_ts = pd.Timestamp(as_of) if as_of is not None else df["date"].max()
    df = df[df["date"] <= as_of_ts]
    debits = df[df["amount"] < 0]
    words = df["merchant"].map(lambda m: set(re.findall(r"[A-Z]+", m)))

    subs: list[Subscription] = []
    for merchant, g in debits.groupby("merchant"):
        category = g["category"].mode().iloc[0]
        if len(g) < MIN_CHARGES or category in NOT_SUBSCRIPTIONS:
            continue
        spend = -g["amount"]
        median = spend.median()
        charges = g[(spend - median).abs() <= AMOUNT_TOLERANCE * median]
        if len(charges) < MIN_CHARGES or len(charges) / len(g) < MIN_RECURRING_SHARE:
            continue
        gaps = charges["date"].diff().dt.days.dropna()
        cadence = float(gaps.median())
        if not (CADENCE_DAYS[0] <= cadence <= CADENCE_DAYS[1]) or gaps.max() > MAX_GAP_DAYS:
            continue
        last_charge = charges["date"].max()
        if (as_of_ts - last_charge).days > STILL_ACTIVE_DAYS:
            continue

        brand = _brand(merchant)
        is_activity = words.map(lambda w: brand in w) if brand else pd.Series(False, index=df.index)
        activity = df[is_activity & ~df["id"].isin(charges["id"])]
        last_activity = activity["date"].max() if not activity.empty else None
        days_since = (as_of_ts - last_activity).days if last_activity is not None else None
        unused = days_since is not None and days_since >= UNUSED_AFTER_DAYS
        if last_activity is None:
            reason = "No usage signal in the statement, so we can't tell whether it is used."
        elif unused:
            reason = (f"Last related activity was {days_since} days ago ({_to_date(last_activity):%d %b %Y}), "
                      f"but it is still charged monthly.")
        else:
            reason = f"Related activity {days_since} days ago, so it looks in use."

        monthly = round(float(-charges["amount"].median()), 2)
        subs.append(Subscription(
            merchant=merchant, category=category, monthly_amount=monthly, annual_cost=round(monthly * 12, 2),
            charge_count=len(charges), first_charge=_to_date(charges["date"].min()),
            last_charge=_to_date(last_charge), cadence_days=cadence,
            last_activity=_to_date(last_activity) if last_activity is not None else None,
            days_since_activity=days_since, possibly_unused=unused, reason=reason,
            txn_ids=tuple(int(i) for i in charges["id"]),
            activity_txn_ids=tuple(int(i) for i in activity["id"]),
        ))
    subs.sort(key=lambda s: (-s.annual_cost, s.merchant))
    return subs
