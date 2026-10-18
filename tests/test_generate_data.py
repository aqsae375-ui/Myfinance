import csv
from collections import Counter

from scripts.generate_data import (
    HEAVY_DINING_MONTH, INJECTION_TEXT, LARGE_BILL_MONTH, MONTHS, build_statement,
)


def _month(t):
    return (t.date.year, t.date.month)


def test_deterministic_for_same_seed():
    assert build_statement(7) == build_statement(7)


def test_spans_six_months(statement_txns):
    assert sorted({_month(t) for t in statement_txns}) == MONTHS


def test_heavy_dining_month_is_the_peak(statement_txns):
    dining = Counter()
    for t in statement_txns:
        if t.category == "dining":
            dining[_month(t)] += -t.amount
    peak, runner_up = dining.most_common(2)
    assert peak[0] == HEAVY_DINING_MONTH and peak[1] > 2 * runner_up[1]


def test_large_bill_month(statement_txns):
    bills = [t for t in statement_txns if "large_bill" in t.tags]
    assert len(bills) == 1 and _month(bills[0]) == LARGE_BILL_MONTH and bills[0].amount < -400


def test_required_features_present(statement_txns):
    tags = Counter(tag for t in statement_txns for tag in t.tags)
    assert tags["unused_subscription"] == len(MONTHS)
    assert tags["subscription"] == 3 * len(MONTHS)
    assert tags["anomaly"] >= 4
    assert tags["duplicate_charge"] == 2
    assert any(t.description == INJECTION_TEXT for t in statement_txns)
    assert sum(1 for t in statement_txns if t.pii) >= 3 * len(MONTHS)


def test_csv_mixes_two_date_formats(statement_csv):
    with open(statement_csv, newline="", encoding="utf-8") as f:
        dates = [row["Date"] for row in csv.DictReader(f)]
    assert any("-" in d for d in dates) and any("/" in d for d in dates)
