import re
from pathlib import Path

import pytest

from src.analytics import insights as insights_module
from src.analytics.common import format_money, format_pct
from src.analytics.insights import MAX_INSIGHTS, generate_insights
from src.analytics.monthly import all_months, month_over_month


def _kinds(ins):
    return [i.kind for i in ins]


def test_exact_sentence_for_a_category_increase(make_txns):
    df = make_txns([
        ("2026-01-10", "SUSHI", -100.0, "Dining Out"), ("2026-01-11", "GROCER", -200.0, "Groceries"),
        ("2026-02-10", "SUSHI", -135.0, "Dining Out"), ("2026-02-11", "GROCER", -210.0, "Groceries"),
    ])
    [ins] = generate_insights("2026-02", df)  # groceries +5% is below the 15% threshold
    assert ins.text == "Dining Out is up 35% vs last month, an extra Rs 35.00."
    assert ins.kind == "category_increase" and ins.impact == 35.0
    assert ins.numbers == {"current": 135.0, "previous": 100.0, "change": 35.0, "pct_change": 35.0}


def test_heavy_dining_month_sentence_has_the_computed_numbers(synthetic_df):
    ch = month_over_month("2026-05", synthetic_df).category("Dining Out").amount
    dining = [i for i in generate_insights("2026-05", synthetic_df) if "Dining Out" in i.text]
    assert dining and dining[0].kind == "category_increase"
    assert format_pct(ch.pct_change) in dining[0].text and format_money(ch.abs_change) in dining[0].text
    assert dining[0].numbers["change"] == ch.abs_change


def test_positive_callout_when_dining_drops_after_may(synthetic_df):
    ch = month_over_month("2026-06", synthetic_df).category("Dining Out").amount
    [dining] = [i for i in generate_insights("2026-06", synthetic_df) if "Dining Out" in i.text]
    assert dining.kind == "category_decrease" and dining.text.startswith("Nice work: Dining Out is down")
    assert format_pct(ch.pct_change) in dining.text and format_money(ch.abs_change) in dining.text


def test_unused_subscription_insight(synthetic_df):
    [gym] = [i for i in generate_insights("2026-08", synthetic_df) if i.kind == "unused_subscription"]
    assert "FITPRO GYM MEMBERSHIP" in gym.text
    assert "Rs 44.99 a month" in gym.text and "Rs 540 a year" in gym.text
    assert f"{gym.numbers['days_since_activity']} days" in gym.text and gym.impact == 539.88
    # not yet flagged in May: the last gym visit was 45 days earlier
    assert "unused_subscription" not in _kinds(generate_insights("2026-05", synthetic_df, limit=50))


def test_anomaly_insights(synthetic_df):
    july = generate_insights("2026-07", synthetic_df)
    [power] = [i for i in july if i.kind == "anomaly"]
    assert "CITY POWER & LIGHT for Rs 486 on 08 Jul" in power.text
    assert f"{power.numbers['multiple']:.1f}x" in power.text and format_money(power.numbers["baseline"]) in power.text
    [apex] = [i for i in generate_insights("2026-06", synthetic_df) if i.kind == "anomaly"]
    assert apex.text == "Unusual charge: Rs 1,349 at a new merchant, APEX ELECTRONICS, on 20 Jun."


def test_changes_caused_only_by_anomalies_are_not_repeated(synthetic_df):
    july = [i.text for i in generate_insights("2026-07", synthetic_df, limit=50)]
    assert not any(t.startswith("Utilities is up") for t in july)       # the power-bill anomaly says it
    assert not any(t.startswith("Nice work: Shopping") for t in july)   # June's TV simply didn't repeat
    august = [i.text for i in generate_insights("2026-08", synthetic_df, limit=50)]
    assert not any(t.startswith("Transfers is up") for t in august)     # the wire anomaly says it
    assert not any(t.startswith("Nice work: Utilities") for t in august)


@pytest.mark.parametrize("month", ["2026-04", "2026-05", "2026-06", "2026-07", "2026-08"])
def test_capped_and_ranked_by_impact(synthetic_df, month):
    ins = generate_insights(month, synthetic_df)
    assert 1 <= len(ins) <= MAX_INSIGHTS
    assert [i.impact for i in ins] == sorted((i.impact for i in ins), reverse=True)
    everything = generate_insights(month, synthetic_df, limit=100)
    assert ins == everything[:MAX_INSIGHTS]


def test_every_number_in_every_sentence_is_a_computed_number(synthetic_df):
    for month in all_months(synthetic_df):
        for ins in generate_insights(month, synthetic_df):
            allowed = set()
            for v in ins.numbers.values():
                allowed |= {format_money(v).split()[1], format_pct(v).rstrip("%"), f"{abs(v):.1f}", str(v)}
            for num in re.findall(r"\d[\d,]*(?:\.\d+)?", re.sub(r"\d{2} [A-Z][a-z]{2}", "", ins.text)):
                assert num in allowed, (num, ins.text)


def test_first_month_has_no_change_insights(synthetic_df):
    ins = generate_insights("2026-03", synthetic_df)
    assert not {"category_increase", "category_decrease"} & set(_kinds(ins))


def test_no_llm_anywhere_in_analytics():
    pkg = Path(insights_module.__file__).parent
    for path in pkg.glob("*.py"):
        src = path.read_text(encoding="utf-8").lower()
        assert "src.llm" not in src and "ollama" not in src, path.name
