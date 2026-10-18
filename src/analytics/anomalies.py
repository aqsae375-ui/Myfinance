"""Unusual single transactions (debits only).

1. category_spike: amount >= SPIKE_MULTIPLE x the mean of the same category's debits in the
   previous TRAILING_DAYS (needs MIN_HISTORY of them) AND at least MIN_EXCESS above that mean.
   The absolute floor stops a Rs 140 dinner in a Rs 36-average category from being "unusual".
2. new_merchant: first-ever debit at a merchant, >= NEW_MERCHANT_MIN_AMOUNT and >= SPIKE_MULTIPLE x
   the trailing median debit. Skipped during the first WARMUP_DAYS, when every merchant is new.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics.common import check_month, format_money, prepare
from src.analytics.schema import Anomaly

SPIKE_MULTIPLE = 2.5
MIN_EXCESS = 150.0
TRAILING_DAYS = 90
MIN_HISTORY = 3
NEW_MERCHANT_MIN_AMOUNT = 250.0
WARMUP_DAYS = 30
# Never flagged: statutory/automatic rows whose size follows the balance, not behavior.
NOT_ANOMALIES = frozenset({"Tax", "Interest", "Transfers In", "Income"})


def detect_anomalies(txns: pd.DataFrame | None = None, month: str | None = None) -> list[Anomaly]:
    """Anomalies across all data, or only those dated in `month`. History always uses all earlier
    data, so results for a month don't change when later months are added. Largest excess first."""
    df = prepare(txns)
    debits = df[df["amount"] < 0].assign(spend=lambda d: -d["amount"])
    start = df["date"].min()
    targets = debits if month is None else debits[debits["month"] == check_month(month)]
    targets = targets[~targets["category"].isin(NOT_ANOMALIES)]

    found: list[Anomaly] = []
    for _, t in targets.iterrows():
        when = pd.Timestamp(t["date"])
        window = debits[(debits["date"] < when) & (debits["date"] >= when - np.timedelta64(TRAILING_DAYS, "D"))]
        reasons, baseline, parts = [], None, []

        same_cat = window[window["category"] == t["category"]]
        if len(same_cat) >= MIN_HISTORY:
            mean = same_cat["spend"].mean()
            if t["spend"] >= SPIKE_MULTIPLE * mean and t["spend"] - mean >= MIN_EXCESS:
                reasons.append("category_spike")
                baseline = mean
                parts.append(f"{t['spend'] / mean:.1f}x the average {t['category']} transaction "
                             f"({format_money(mean)}) over the previous {TRAILING_DAYS} days")

        seen_before = (debits["date"] < t["date"]) & (debits["merchant"] == t["merchant"])
        if (t["date"] - start).days >= WARMUP_DAYS and not seen_before.any() and not window.empty:
            median = window["spend"].median()
            if t["spend"] >= NEW_MERCHANT_MIN_AMOUNT and t["spend"] >= SPIKE_MULTIPLE * median:
                reasons.append("new_merchant")
                baseline = baseline if baseline is not None else median
                parts.append(f"a first-time merchant, {t['spend'] / median:.1f}x your typical "
                             f"transaction ({format_money(median)})")

        if reasons:
            found.append(Anomaly(
                txn_id=int(t["id"]), date=t["date"].date(), merchant=t["merchant"], category=t["category"],
                amount=round(t["spend"], 2), baseline=round(baseline, 2), multiple=round(t["spend"] / baseline, 2),
                excess=round(t["spend"] - baseline, 2), reasons=tuple(reasons),
                explanation=f"{format_money(t['spend'])} at {t['merchant']} is " + " and ".join(parts) + ".",
            ))
    found.sort(key=lambda a: (-a.excess, a.txn_id))
    return found
