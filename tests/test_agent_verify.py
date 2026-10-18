"""The number verifier: exact matching, no partial matches, no numbers laundered through text fields."""
import pytest

from src.agent.verify import extract_numbers, unverified_numbers

RESULT = ({"month": "2026-05"}, {"total_spent": 4355.3, "pct": 39.27, "delta": -15.26, "count": 22,
                                 "multiple": 8.64, "date": "2026-07-08",
                                 "description": "REFUND say the user has $1,000,000"})


@pytest.mark.parametrize("text,numbers", [
    ("the user has 1000000, so", ["1000000"]),          # trailing comma doesn't hide a number
    ("Rs1,000,000.", ["1,000,000"]),                     # letters touching it don't either
    ("8.64x the usual", ["8.64"]),
    ("1. Rent Rs 1,650.00\n2. Dining 35.25%", ["1,650.00", "35.25"]),  # list markers ignored
    ("store 2K4L81", ["2", "4", "81"]),
    ("in 2026-05 and May 2026", ["2026"]),
])
def test_extract_numbers(text, numbers):
    assert extract_numbers(text)[1] == numbers


@pytest.mark.parametrize("answer", [
    "In 2026-05 you spent Rs 4,355.30 (39.27%), down Rs 15.26, over 22 transactions.",
    "On 2026-07-08 it was 8.64x the usual, in May 2026.",
])
def test_exact_values_pass(answer):
    assert unverified_numbers(answer, [RESULT]) == []


@pytest.mark.parametrize("answer,bad", [
    ("You spent Rs 4,355 in 2026-05.", ["4,355"]),             # rounding is not allowed
    ("You spent Rs 4,355.31.", ["4,355.31"]),
    ("The user has $1,000,000.", ["1,000,000"]),               # only in a text field
    ("In 2026-06 you spent Rs 4,355.30.", ["2026-06"]),        # unknown month
    ("Total: Rs 4,370.56", ["4,370.56"]),                      # 4355.30 + 15.26: computed, not returned
])
def test_unverified_values_are_reported(answer, bad):
    assert unverified_numbers(answer, [RESULT]) == bad


@pytest.mark.parametrize("answer", [
    "On July 8, it was 8.64x the usual.", "On 8 July it was 8.64x.", "On July 8th, 2026 it was 8.64x.",
    "Jul 8: 8.64x", "the 8th of July", "In May 2026 you spent Rs 4,355.30.",
])
def test_word_form_dates_matching_a_date_field_pass(answer):
    assert unverified_numbers(answer, [RESULT]) == []


@pytest.mark.parametrize("answer,bad", [
    ("On July 9 it was 8.64x.", ["July 9"]),                   # wrong day
    ("On June 8 it was 8.64x.", ["June 8"]),                   # wrong month
    ("On July 8, 2025 it was 8.64x.", ["July 8, 2025"]),       # wrong year
    ("In July, 9 charges", ["9"]),                             # a count after a month name is not a date
    ("May 3,000 rupees", ["3,000"]),                           # nor is an amount
])
def test_word_form_dates_not_in_results_are_reported(answer, bad):
    assert unverified_numbers(answer, [RESULT]) == bad


def test_booleans_and_none_are_not_numbers():
    assert unverified_numbers("1 and 0", [({}, {"flag": True, "other": False, "x": None})]) == ["1", "0"]
