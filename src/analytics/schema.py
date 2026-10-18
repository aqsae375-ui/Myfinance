"""Typed return shapes for the analytics engine. The dashboard (Stage 6) and the agent/advisor
consume these directly, so every figure carries enough context to explain where it came from.

Money is always a positive float rounded to 2 decimals unless the field says otherwise
(e.g. `net_saved` and `Change.abs_change` are signed). Percentages are 0-100, not 0-1.
Months are "YYYY-MM" strings.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass(frozen=True)
class CategoryTotal:
    category: str
    amount: float                 # money out in this category (positive)
    pct_of_spend: float           # share of the month's total_spent, 0-100
    txn_count: int
    txn_ids: tuple[int, ...]      # source rows, for the "How I got this" panel


@dataclass(frozen=True)
class MonthlySummary:
    month: str
    total_spent: float            # sum of all debits (money out), positive
    total_income: float           # sum of all credits (money in)
    net_saved: float              # total_income - total_spent, signed
    txn_count: int
    categories: tuple[CategoryTotal, ...]  # spend by category, largest first
    method: str                   # plain-English formula, shown as the source of these figures
    other_credits: float = 0.0    # money in that is not income (refunds, reversals), positive

    def category(self, name: str) -> CategoryTotal | None:
        return next((c for c in self.categories if c.category == name), None)


@dataclass(frozen=True)
class Change:
    current: float
    previous: float | None        # None when there is no previous month
    abs_change: float | None      # current - previous, signed
    pct_change: float | None      # abs_change / |previous| * 100; None if previous is missing or 0


@dataclass(frozen=True)
class CategoryChange:
    category: str
    amount: Change
    pct_of_spend: Change          # abs_change here is in percentage points


@dataclass(frozen=True)
class MonthComparison:
    month: str
    previous_month: str | None    # None for the first month in the data
    total_spent: Change
    total_income: Change
    net_saved: Change
    categories: tuple[CategoryChange, ...]  # union of both months, largest current spend first
    current: MonthlySummary
    previous: MonthlySummary | None

    def category(self, name: str) -> CategoryChange | None:
        return next((c for c in self.categories if c.category == name), None)


@dataclass(frozen=True)
class Subscription:
    merchant: str                 # normalized merchant key (from the redacted description)
    category: str
    monthly_amount: float         # median charge
    annual_cost: float            # monthly_amount * 12
    charge_count: int
    first_charge: date
    last_charge: date
    cadence_days: float           # median days between charges
    last_activity: date | None    # latest related non-charge transaction, if any ever
    days_since_activity: int | None
    possibly_unused: bool
    reason: str                   # why it is / isn't flagged, in plain English
    txn_ids: tuple[int, ...]      # the recurring charges
    activity_txn_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class Anomaly:
    txn_id: int
    date: date
    merchant: str
    category: str
    amount: float                 # money out (positive)
    baseline: float | None        # trailing category mean (spike) or trailing median debit (new merchant)
    multiple: float | None        # amount / baseline
    excess: float                 # amount - baseline: the rupee impact
    reasons: tuple[str, ...]      # "category_spike" and/or "new_merchant"
    explanation: str


@dataclass(frozen=True)
class Insight:
    month: str
    kind: str                     # category_increase | category_decrease | unused_subscription | anomaly
    text: str                     # final sentence, numbers already embedded
    impact: float                 # absolute rupee impact used for ranking (positive)
    numbers: dict[str, float] = field(default_factory=dict)  # every figure used in `text`
    source: str = ""              # how the numbers were computed
    txn_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class TransactionRow:
    txn_id: int
    date: date
    description: str              # redacted description (never the raw one)
    merchant: str
    category: str
    amount: float                 # signed, as stored: negative = money out


@dataclass(frozen=True)
class TransactionSearch:
    filters: dict[str, object]    # the filters actually applied, for the "How I got this" panel
    match_count: int              # all matches, even beyond `limit`
    total_out: float              # money out across all matches (positive)
    total_in: float               # money in across all matches
    rows: tuple[TransactionRow, ...]  # at most `limit` rows, in the requested order
    txn_ids: tuple[int, ...]      # every matching row, not just the returned ones


# ---- Dashboard views (Stage 6) ------------------------------------------------------------------

@dataclass(frozen=True)
class MonthRow:
    """One row of the month-history table. Changes are vs the calendar month before (None if absent)."""
    month: str
    total_spent: float
    total_income: float
    net_saved: float              # signed
    savings_rate: float | None    # net_saved / total_income * 100; None when there is no income
    spent: Change
    income: Change
    net: Change
    rate: Change                  # abs_change in percentage points; pct_change is always None


@dataclass(frozen=True)
class MerchantTotal:
    merchant: str
    category: str                 # most common category of the merchant's debits this month
    amount: float                 # money out (positive)
    txn_count: int
    txn_ids: tuple[int, ...]


@dataclass(frozen=True)
class DaySpend:
    date: date
    amount: float                 # money out that day (positive; 0 on days with no debits)
    txn_count: int


@dataclass(frozen=True)
class DailySpending:
    month: str
    days: tuple[DaySpend, ...]    # every calendar day of the month, in order
    peak: DaySpend | None         # highest-spend day; None if nothing was spent


@dataclass(frozen=True)
class LedgerRow:
    """A transaction as the dashboard table shows it, with how it was categorized."""
    txn_id: int
    date: date
    merchant: str
    description: str              # redacted description
    category: str
    amount: float                 # signed: negative = money out
    source: str | None            # rule | llm | manual | learned | None (not categorized)
    confidence: float | None
    needs_review: bool = False    # the model was unsure or not confident: waiting for the user


@dataclass(frozen=True)
class CashShare:
    """Cash withdrawn as a share of all spending, from January of `month`'s year through `month`."""
    month: str                    # last month included
    start_month: str              # first month with data in that year, up to `month`
    cash: float                   # Cash Withdrawal debits (positive)
    spent: float                  # all debits (positive)
    pct_of_spend: float | None    # cash / spent * 100; None when nothing was spent
    month_cash: float             # cash withdrawn in `month` itself
    txn_ids: tuple[int, ...]


@dataclass(frozen=True)
class CashSummary:
    """The cash wallet for one month: withdrawals in, logged cash entries out (src/analytics/cash.py)."""
    month: str
    withdrawn: float              # Cash Withdrawal debits this month (positive)
    logged: float                 # cash entries dated this month (positive)
    unaccounted: float            # max(0, withdrawn - logged): counted as spending in "Unaccounted cash"
    outside: float                # max(0, logged - withdrawn): "cash from outside the account", also spending
    accounted_pct: float | None   # min(logged, withdrawn) / withdrawn * 100; None when nothing was withdrawn
    wallet_balance: float         # all withdrawn - all logged up to this month (signed; information only)
    method: str
    txn_ids: tuple[int, ...]      # the withdrawals
    entry_ids: tuple[int, ...]    # the cash entries
