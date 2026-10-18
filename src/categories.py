"""The one category taxonomy used everywhere (rules, LLM prompts, validation, DB, UI)."""
from __future__ import annotations

CATEGORIES: tuple[str, ...] = (
    "Income", "Interest", "Transfers In", "Rent", "Groceries", "Dining Out", "Transport", "Utilities",
    "Subscriptions", "Entertainment", "Shopping", "Health", "Charity/Donations", "Cash Withdrawal",
    "Bank Fees", "Tax", "Transfers", "Other",
)
FALLBACK = "Other"

# Money in that counts as income (salary-type credits, profit/interest, money received from people).
# Other credits (refunds, reversals) are not income; see src/analytics/monthly.py.
INCOME_CATEGORIES = frozenset({"Income", "Interest", "Transfers In"})

# Extra spellings a model may answer with.
_ALIASES = {"charity": "Charity/Donations", "donation": "Charity/Donations", "donations": "Charity/Donations",
            "charity / donations": "Charity/Donations", "bank fee": "Bank Fees", "fees": "Bank Fees",
            "taxes": "Tax", "transfer in": "Transfers In", "cash withdrawals": "Cash Withdrawal"}

_BY_KEY = {c.casefold(): c for c in CATEGORIES} | _ALIASES


def normalize_category(value: object) -> str | None:
    """Return the canonical taxonomy name for `value` (case/whitespace-insensitive), else None."""
    if not isinstance(value, str):
        return None
    return _BY_KEY.get(" ".join(value.split()).casefold())


def is_valid_category(value: object) -> bool:
    return normalize_category(value) is not None


# Ground-truth labels (synthetic generator + real-format fixture) -> taxonomy.
GROUND_TRUTH_MAP = {
    "income": "Income", "rent": "Rent", "groceries": "Groceries", "dining": "Dining Out",
    "transport": "Transport", "utilities": "Utilities", "subscriptions": "Subscriptions",
    "entertainment": "Entertainment", "shopping": "Shopping", "health": "Health",
    "transfer": "Transfers", "transfer_in": "Transfers In", "fees": "Bank Fees", "other": "Other",
    "interest": "Interest", "tax": "Tax", "cash": "Cash Withdrawal", "charity": "Charity/Donations",
}
