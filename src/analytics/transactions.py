"""Filtered transaction search: the one place that selects and totals rows by user-given filters."""
from __future__ import annotations

from datetime import date

import pandas as pd

from src.analytics.common import prepare
from src.analytics.schema import TransactionRow, TransactionSearch
from src.categories import normalize_category

MAX_LIMIT = 50
SORTS = ("date", "amount")
FLOWS = ("out", "in")


def search_transactions(txns: pd.DataFrame | None = None, *, category: str | None = None,
                        start_date: str | date | None = None, end_date: str | date | None = None,
                        min_amount: float | None = None, merchant_keyword: str | None = None,
                        flow: str | None = None, sort: str = "date", limit: int = 20) -> TransactionSearch:
    """Rows matching every given filter. `min_amount` compares the unsigned amount, so it applies
    to money in and out alike; `merchant_keyword` is a case-insensitive substring of the *redacted*
    description; `flow` is "out" (debits) or "in" (credits); dates are inclusive.
    `sort="amount"` = largest first. Totals cover all matches, not just the returned rows."""
    if flow is not None and flow not in FLOWS:
        raise ValueError(f"flow must be one of {FLOWS}, got {flow!r}")
    if sort not in SORTS:
        raise ValueError(f"sort must be one of {SORTS}, got {sort!r}")
    limit = max(1, min(int(limit), MAX_LIMIT))
    df = prepare(txns)
    mask = pd.Series(True, index=df.index)
    filters: dict[str, object] = {}
    if category is not None:
        canon = normalize_category(category)
        if canon is None:
            raise ValueError(f"unknown category {category!r}")
        mask &= df["category"] == canon
        filters["category"] = canon
    if start_date is not None:
        mask &= df["date"] >= pd.Timestamp(start_date)
        filters["start_date"] = pd.Timestamp(start_date).date().isoformat()
    if end_date is not None:
        mask &= df["date"] <= pd.Timestamp(end_date)
        filters["end_date"] = pd.Timestamp(end_date).date().isoformat()
    if min_amount is not None:
        mask &= df["amount"].abs() >= float(min_amount)
        filters["min_amount"] = float(min_amount)
    if merchant_keyword:
        mask &= df["redacted_description"].str.contains(merchant_keyword, case=False, regex=False)
        filters["merchant_keyword"] = merchant_keyword

    if flow is not None:
        mask &= (df["amount"] < 0) if flow == "out" else (df["amount"] > 0)
        filters["flow"] = flow

    hits = df[mask]
    ordered = (hits.assign(_size=hits["amount"].abs()).sort_values(["_size", "id"], ascending=[False, True])
               if sort == "amount" else hits)
    rows = tuple(TransactionRow(txn_id=int(r.id), date=r.date.date(), description=r.redacted_description,
                                merchant=r.merchant, category=r.category, amount=round(float(r.amount), 2))
                 for r in ordered.head(limit).itertuples())
    return TransactionSearch(
        filters=filters, match_count=len(hits),
        total_out=round(float(-hits.loc[hits["amount"] < 0, "amount"].sum()), 2),
        total_in=round(float(hits.loc[hits["amount"] > 0, "amount"].sum()), 2),
        rows=rows, txn_ids=tuple(int(i) for i in hits["id"]),
    )
