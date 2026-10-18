from datetime import date

import pytest

from src.analytics.anomalies import detect_anomalies


def _by_merchant(anomalies):
    return {a.merchant: a for a in anomalies}


def test_catches_the_planted_anomalies(synthetic_df):
    found = _by_merchant(detect_anomalies(synthetic_df))
    apex = found["APEX ELECTRONICS"]  # large purchase at a first-time merchant
    assert apex.date == date(2026, 6, 20) and apex.amount == 1349.0 and "new_merchant" in apex.reasons
    power = found["CITY POWER & LIGHT"]  # the large July bill
    assert power.date == date(2026, 7, 8) and power.amount == 486.2 and "category_spike" in power.reasons
    assert power.multiple >= 2.5 and power.excess == pytest.approx(power.amount - power.baseline, abs=0.01)
    wire = found["INTL WIRE TO IBAN"]
    assert wire.date == date(2026, 8, 22) and wire.amount == 750.0


def test_no_false_alarms_on_normal_spending(synthetic_df):
    found = detect_anomalies(synthetic_df)
    assert {a.merchant for a in found} == {"APEX ELECTRONICS", "CITY POWER & LIGHT", "INTL WIRE TO IBAN"}
    # the heavy-dining month is a monthly trend (an insight), not a pile of single-transaction anomalies
    assert detect_anomalies(synthetic_df, month="2026-05") == []
    assert detect_anomalies(synthetic_df, month="2026-03") == []  # warm-up: everything is new


def test_month_filter_and_order(synthetic_df):
    assert [a.merchant for a in detect_anomalies(synthetic_df, month="2026-07")] == ["CITY POWER & LIGHT"]
    excess = [a.excess for a in detect_anomalies(synthetic_df)]
    assert excess == sorted(excess, reverse=True)


def test_spike_threshold_math(make_txns):
    history = [("2026-01-05", "GROCER", -30.0, "Groceries"), ("2026-01-12", "GROCER", -40.0, "Groceries"),
               ("2026-01-19", "GROCER", -50.0, "Groceries")]  # trailing mean 40
    below = make_txns(history + [("2026-01-26", "GROCER", -99.0, "Groceries")])      # 2.475x
    assert detect_anomalies(below) == []
    small = make_txns(history + [("2026-01-26", "GROCER", -120.0, "Groceries")])     # 3x but only +80
    assert detect_anomalies(small) == []
    above = make_txns(history + [("2026-01-26", "GROCER", -250.0, "Groceries")])
    [a] = detect_anomalies(above)
    assert (a.baseline, a.multiple, a.excess, a.reasons) == (40.0, 6.25, 210.0, ("category_spike",))
