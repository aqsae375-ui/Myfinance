from collections import defaultdict

import pytest

from scripts.generate_data import MONTHS
from src.analytics.common import format_money, merchant_key, previous_month
from src.analytics.monthly import all_months, change, month_over_month, monthly_summary
from src.categories import GROUND_TRUTH_MAP
from src.llm.base import looks_like_instruction


def _expected_category(t):
    """Generator label; the planted injection row is never categorized (it goes to the review queue)."""
    return "Uncategorized" if looks_like_instruction(t.description) else GROUND_TRUTH_MAP[t.category]


MONTH_KEYS = [f"{y}-{m:02d}" for y, m in MONTHS]


def _expected(statement_txns):
    """Totals straight from the generator, independent of parser/DB/categorizer."""
    exp = defaultdict(lambda: {"spent": 0.0, "income": 0.0, "cats": defaultdict(float)})
    for t in statement_txns:
        e = exp[t.date.strftime("%Y-%m")]
        if t.amount < 0:
            e["spent"] += -t.amount
            e["cats"][_expected_category(t)] += -t.amount
        else:
            e["income"] += t.amount
    return exp


def test_all_months_sorted(synthetic_df):
    assert all_months(synthetic_df) == MONTH_KEYS


@pytest.mark.parametrize("month", MONTH_KEYS)
def test_totals_match_generator(synthetic_df, statement_txns, month):
    exp = _expected(statement_txns)[month]
    s = monthly_summary(month, synthetic_df)
    assert s.total_spent == pytest.approx(exp["spent"], abs=0.01)
    assert s.total_income == pytest.approx(exp["income"], abs=0.01)
    assert s.net_saved == pytest.approx(exp["income"] - exp["spent"], abs=0.01)
    assert {c.category: c.amount for c in s.categories} == pytest.approx(dict(exp["cats"]), abs=0.01)
    assert sum(c.pct_of_spend for c in s.categories) == pytest.approx(100, abs=0.1)
    assert sum(c.txn_count for c in s.categories) == sum(1 for t in statement_txns
                                                         if t.date.strftime("%Y-%m") == month and t.amount < 0)


def test_heavy_dining_month_is_the_dining_peak(synthetic_df):
    dining = {m: monthly_summary(m, synthetic_df).category("Dining Out").amount for m in MONTH_KEYS}
    assert max(dining, key=dining.get) == "2026-05"


def test_categories_sorted_and_carry_source_ids(synthetic_df):
    s = monthly_summary("2026-05", synthetic_df)
    amounts = [c.amount for c in s.categories]
    assert amounts == sorted(amounts, reverse=True)
    ids = set(synthetic_df["id"])
    assert all(set(c.txn_ids) <= ids and len(c.txn_ids) == c.txn_count for c in s.categories)


@pytest.fixture
def two_months(make_txns):
    return make_txns([
        ("2026-01-05", "SALARY", 1000.0, "Income"),
        ("2026-01-10", "SUSHI", -100.0, "Dining Out"),
        ("2026-01-11", "GROCER", -200.0, "Groceries"),
        ("2026-01-12", "SHELL", -40.0, "Transport"),
        ("2026-02-05", "SALARY", 1000.0, "Income"),
        ("2026-02-10", "SUSHI", -135.0, "Dining Out"),
        ("2026-02-11", "GROCER", -150.0, "Groceries"),
        ("2026-02-12", "AMZN", -80.0, "Shopping"),
    ])


def test_pct_change_math(two_months):
    mom = month_over_month("2026-02", two_months)
    assert mom.previous_month == "2026-01"
    assert (mom.total_spent.current, mom.total_spent.previous) == (365.0, 340.0)
    assert mom.total_spent.abs_change == 25.0 and mom.total_spent.pct_change == pytest.approx(7.35, abs=0.01)
    assert mom.net_saved.abs_change == -25.0 and mom.net_saved.pct_change == pytest.approx(-3.79, abs=0.01)

    dining = mom.category("Dining Out").amount
    assert (dining.abs_change, dining.pct_change) == (35.0, 35.0)
    groceries = mom.category("Groceries").amount
    assert (groceries.abs_change, groceries.pct_change) == (-50.0, -25.0)
    shopping = mom.category("Shopping").amount   # new: nothing to compare against
    assert (shopping.previous, shopping.abs_change, shopping.pct_change) == (0.0, 80.0, None)
    transport = mom.category("Transport").amount  # disappeared
    assert (transport.current, transport.pct_change) == (0.0, -100.0)
    # share-of-spend change is in percentage points
    share = mom.category("Dining Out").pct_of_spend
    assert share.abs_change == pytest.approx(round(135 / 365 * 100, 2) - round(100 / 340 * 100, 2), abs=0.01)


def test_pct_change_uses_abs_previous_for_negative_base():
    ch = change(-50.0, -100.0)  # a deficit shrinking from 100 to 50 is +50%
    assert (ch.abs_change, ch.pct_change) == (50.0, 50.0)
    assert change(10.0, 0.0).pct_change is None


def test_first_month_has_no_previous(synthetic_df):
    mom = month_over_month("2026-03", synthetic_df)
    assert mom.previous_month is None and mom.previous is None
    for ch in [mom.total_spent, mom.total_income, mom.net_saved] + [c.amount for c in mom.categories]:
        assert ch.previous is None and ch.abs_change is None and ch.pct_change is None
    assert mom.total_spent.current == monthly_summary("2026-03", synthetic_df).total_spent


def test_gap_before_month_means_no_comparison(make_txns):
    df = make_txns([("2026-01-05", "A", -10.0, "Other"), ("2026-03-05", "A", -20.0, "Other")])
    assert month_over_month("2026-03", df).previous_month is None
    assert previous_month("2026-01") == "2025-12"


def test_bad_month_inputs(synthetic_df):
    with pytest.raises(ValueError):
        monthly_summary("2026-5", synthetic_df)
    with pytest.raises(ValueError):
        monthly_summary("2025-01", synthetic_df)


def test_uncategorized_rows_are_counted(make_txns):
    df = make_txns([("2026-01-05", "A", -10.0, None), ("2026-01-06", "B", -30.0, "Other")])
    s = monthly_summary("2026-01", df)
    assert s.category("Uncategorized").amount == 10.0 and s.total_spent == 40.0


def test_merchant_key_and_money_format():
    assert merchant_key("AMZN MKTP US*2K4L81") == "AMZN MKTP US"
    assert merchant_key("ZELLE TO [NAME] [PHONE]") == "ZELLE TO"
    assert merchant_key("GREENLEAF GROCERY #0412") == "GREENLEAF GROCERY"
    assert format_money(4200.4) == "Rs 4,200" and format_money(-44.99) == "Rs 44.99"
