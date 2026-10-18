import statistics

from src.advisor.opportunities import MIN_MONTHLY_SAVING, find_opportunities
from src.analytics import all_months, monthly_summary


def _known_txns(make_txns):
    """4 months. Dining: 100, 100, 500, 100 (avg 200, median 100 -> 100/mo). Shopping: 0, 0, 0, 120
    (avg 30, median 0 -> 30/mo). Gym charged 40 monthly, cafe visit only in Jan -> unused (40/mo).
    Groceries and rent are flat -> nothing to trim."""
    rows = []
    for m, dining in zip(("01", "02", "03", "04"), (100.0, 100.0, 500.0, 100.0)):
        rows += [(f"2026-{m}-01", "ACME PAYROLL", 3000.0, "Income"),
                 (f"2026-{m}-02", "RENT PAYMENT", -1000.0, "Rent"),
                 (f"2026-{m}-05", "GYMCO MEMBERSHIP", -40.0, "Subscriptions"),
                 (f"2026-{m}-10", "FRESHMART", -300.0, "Groceries"),
                 (f"2026-{m}-{12 + int(m)}", "BISTRO", -dining, "Dining Out")]
    rows += [("2026-01-06", "GYMCO CAFE", -5.0, "Other"), ("2026-04-20", "SHOPCO", -120.0, "Shopping"),
             ("2026-04-28", "BUS PASS", -3.0, "Transport")]
    return make_txns(rows)


def test_ranking_on_known_data(make_txns):
    opps = find_opportunities(_known_txns(make_txns))
    assert [(o.rank, o.id, o.monthly_saving) for o in opps] == [
        (1, "trim:Dining Out", 100.0), (2, "cancel:GYMCO MEMBERSHIP", 40.0), (3, "trim:Shopping", 30.0)]
    dining, gym, shop = opps
    assert dining.numbers["average"] == 200.0 and dining.numbers["median"] == 100.0
    assert dining.annual_saving == 1200.0 and dining.confidence == "medium"
    assert "Rs 200.00 a month over 4 months minus your typical (median) month Rs 100.00 = Rs 100.00" in dining.calculation
    assert dining.action == "Keep Dining Out to your typical month of Rs 100.00"
    assert gym.kind == "unused_subscription" and gym.confidence == "high" and gym.annual_saving == 480.0
    assert "Cancel GYMCO MEMBERSHIP" in gym.action and "2026-01-06" in gym.calculation
    assert shop.action.startswith("Plan big Shopping purchases ahead")
    # flat categories, essentials and tiny savings are not suggested
    assert not any(o.category in ("Groceries", "Rent", "Transport") for o in opps)


def test_max_items_and_threshold(make_txns):
    df = _known_txns(make_txns)
    assert [o.id for o in find_opportunities(df, max_items=2)] == ["trim:Dining Out", "cancel:GYMCO MEMBERSHIP"]
    assert all(o.monthly_saving >= MIN_MONTHLY_SAVING for o in find_opportunities(df)
               if o.kind != "unused_subscription")


def test_every_figure_in_the_calculation_is_recorded(make_txns):
    from src.agent.verify import unverified_numbers
    for o in find_opportunities(_known_txns(make_txns)):
        assert unverified_numbers(o.calculation + " " + o.action, [o.numbers, list(o.dates)]) == []


def test_synthetic_plan_order_and_numbers(synthetic_df):
    opps = find_opportunities(synthetic_df)
    assert [o.id.split(":")[0] + ":" + o.target for o in opps] == [
        "trim:Shopping", "trim:Dining Out", "review:CITY POWER & LIGHT", "cancel:FITPRO GYM MEMBERSHIP",
        "trim:Groceries"]
    assert [o.monthly_saving for o in opps] == [254.83, 196.67, 71.65, 44.99, 18.84]
    # the dining trim matches an independent computation from Stage 3 monthly summaries
    months = all_months(synthetic_df)
    dining = [monthly_summary(m, synthetic_df).category("Dining Out").amount for m in months]
    expected = round(round(statistics.mean(dining), 2) - round(statistics.median(dining), 2), 2)
    assert opps[1].monthly_saving == expected
    # the one-off power bill: excess spread over the months of history
    review = opps[2]
    assert review.monthly_saving == round(review.numbers["excess"] / len(months), 2)
    assert "APEX ELECTRONICS" in opps[0].calculation   # anomaly shown as evidence on its category
    # the August wire is a transfer (moving money), never a "saving"
    assert not any(o.category == "Transfers" for o in opps)


def test_one_month_has_no_opportunities(make_txns):
    df = make_txns([("2026-01-01", "ACME PAYROLL", 3000.0, "Income"), ("2026-01-05", "BISTRO", -80.0, "Dining Out"),
                    ("2026-01-09", "BISTRO", -900.0, "Dining Out")])
    assert find_opportunities(df) == []
