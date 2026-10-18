from datetime import date

from src.analytics.subscriptions import UNUSED_AFTER_DAYS, detect_subscriptions


def _by_merchant(subs):
    return {s.merchant: s for s in subs}


def test_detects_the_monthly_subscriptions(synthetic_df):
    subs = _by_merchant(detect_subscriptions(synthetic_df))
    assert {"FITPRO GYM MEMBERSHIP", "NETFLIX.COM", "SPOTIFY USA"} <= set(subs)
    gym = subs["FITPRO GYM MEMBERSHIP"]
    assert gym.monthly_amount == 44.99 and gym.annual_cost == 539.88 and gym.charge_count == 6
    assert 28 <= gym.cadence_days <= 31
    # rent is recurring but not a subscription; the power bill varies too much to count
    assert not any(m.startswith("RENT") or m.startswith("CITY POWER") for m in subs)


def test_flags_the_planted_unused_subscription_only(synthetic_df):
    subs = detect_subscriptions(synthetic_df)
    unused = [s for s in subs if s.possibly_unused]
    assert [s.merchant for s in unused] == ["FITPRO GYM MEMBERSHIP"]
    gym = unused[0]
    last_date = synthetic_df["date"].max().date()
    assert gym.last_activity == date(2026, 4, 16)
    assert gym.days_since_activity == (last_date - date(2026, 4, 16)).days >= UNUSED_AFTER_DAYS
    assert len(gym.activity_txn_ids) == 2
    # streaming has no visible usage in a bank statement: reported, not flagged
    netflix = _by_merchant(subs)["NETFLIX.COM"]
    assert netflix.last_activity is None and not netflix.possibly_unused


def test_not_flagged_while_usage_is_recent(synthetic_df):
    subs = _by_merchant(detect_subscriptions(synthetic_df, as_of="2026-05-31"))
    gym = subs["FITPRO GYM MEMBERSHIP"]
    assert gym.days_since_activity == 45 and not gym.possibly_unused
    assert gym.charge_count == 3  # only data up to as_of is used


def test_irregular_and_cancelled_charges_are_not_subscriptions(make_txns):
    df = make_txns([
        # irregular amounts at one merchant, monthly
        ("2026-01-03", "LUIGIS", -12.0, "Dining Out"), ("2026-02-03", "LUIGIS", -55.0, "Dining Out"),
        ("2026-03-03", "LUIGIS", -30.0, "Dining Out"), ("2026-04-03", "LUIGIS", -20.0, "Dining Out"),
        # steady but weekly
        ("2026-01-01", "WEEKLY BOX", -9.99, "Groceries"), ("2026-01-08", "WEEKLY BOX", -9.99, "Groceries"),
        ("2026-01-15", "WEEKLY BOX", -9.99, "Groceries"), ("2026-01-22", "WEEKLY BOX", -9.99, "Groceries"),
        # monthly, but last charged in January: cancelled
        ("2025-11-10", "OLDTV", -8.0, "Subscriptions"), ("2025-12-10", "OLDTV", -8.0, "Subscriptions"),
        ("2026-01-10", "OLDTV", -8.0, "Subscriptions"),
        # a real one
        ("2026-02-05", "STREAMCO", -10.0, "Subscriptions"), ("2026-03-05", "STREAMCO", -10.0, "Subscriptions"),
        ("2026-04-05", "STREAMCO", -10.5, "Subscriptions"),
    ])
    assert [s.merchant for s in detect_subscriptions(df)] == ["STREAMCO"]
