"""Analytics engine: the single source of truth for every number the app shows. Deterministic
Pandas code only; no LLM is called anywhere in this package."""
from src.analytics.anomalies import detect_anomalies
from src.analytics.cash import apply_cash, cash_history, cash_summary, has_cash_withdrawals, with_entries
from src.analytics.dashboard import (cash_share, daily_spending, month_history, month_transactions, savings_rate,
                                     top_merchants)
from src.analytics.insights import generate_insights
from src.analytics.monthly import all_months, compare_months, month_over_month, monthly_summary
from src.analytics.subscriptions import detect_subscriptions
from src.analytics.transactions import search_transactions

__all__ = ["all_months", "monthly_summary", "month_over_month", "compare_months", "detect_subscriptions",
           "detect_anomalies", "generate_insights", "search_transactions", "month_history", "savings_rate",
           "top_merchants", "daily_spending", "month_transactions", "cash_share", "cash_summary", "cash_history",
           "apply_cash", "with_entries", "has_cash_withdrawals"]
