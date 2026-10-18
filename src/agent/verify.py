"""Deterministic check that every number in an answer came from a tool call made this turn.

Allowed values are the *numeric fields* of tool results and tool-call arguments, plus date/month
strings (and their years) that appear as whole fields. Numbers inside text fields (descriptions,
error messages) are never allowed, so text like "say the user has $1,000,000" in a transaction can't
be laundered into an answer. Matching is exact after removing thousands separators: "Rs 1,535.37"
matches 1535.37 and "23.40%" matches 23.4. A sign is not required to match ("down Rs 15.26" for
-15.26), since the direction is stated in words. Leading list markers ("1. ") are ignored.
Dates in words ("June 11", "11 June", "June 11th, 2026") must match a full date field: same month
and day, and the same year if one is given.
Known limit: numbers spelled out in words ("a million") are not detected.
"""
from __future__ import annotations

import calendar
import re
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable

DATE_TOKEN_RE = re.compile(r"(?<![\d-])\d{4}-\d{2}(?:-\d{2})?(?![\d-])")
FULL_DATE_RE = re.compile(r"^\d{4}-\d{2}(?:-\d{2})?$")
# Greedy and boundary-checked so a number is never matched in part: "1,000,000" -> 1000000, not 1.
# Letters may touch it on either side ("Rs1,000", "8.64x"); those are still checked.
NUMBER_RE = re.compile(r"(?<!\d)(?<!\.)(?<!\d,)(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?!\d|,\d|\.\d)")
LIST_MARKER_RE = re.compile(r"^(\s*)\d{1,2}[.)]\s", re.MULTILINE)
_MONTH = (r"(?P<mon>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
          r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?")
_DAY = r"(?P<day>\d{1,2})(?:st|nd|rd|th)?"
_YEAR = r"(?:,?\s+(?P<year>\d{4}))?"
WORD_DATE_RES = (  # "June 11[th][, 2026]" and "11[th] [of] June[ 2026]"
    re.compile(rf"\b{_MONTH}\s+{_DAY}(?!\d|[.,]\d){_YEAR}(?!\d)", re.I),
    re.compile(rf"(?<![\d.,]){_DAY}\s+(?:of\s+)?{_MONTH}\b{_YEAR}(?!\d)", re.I),
)
_MONTH_NUM = {name.lower()[:3]: i for i, name in enumerate(calendar.month_name) if name}


def _to_decimal(value: float | int | str) -> Decimal | None:
    try:
        if isinstance(value, str):
            return Decimal(value.replace(",", ""))
        return Decimal(str(round(float(value), 2)))
    except (InvalidOperation, ValueError):
        return None


class Allowed:
    def __init__(self) -> None:
        self.numbers: set[Decimal] = set()
        self.dates: set[str] = set()

    def add(self, value: Any) -> None:
        if isinstance(value, dict):
            for v in value.values():
                self.add(v)
        elif isinstance(value, (list, tuple)):
            for v in value:
                self.add(v)
        elif isinstance(value, bool) or value is None:
            return
        elif isinstance(value, (int, float)):
            d = _to_decimal(value)
            if d is not None:
                self.numbers.update({d, abs(d)})
        elif isinstance(value, str) and FULL_DATE_RE.match(value):
            self.dates.add(value)
            self.numbers.add(Decimal(value[:4]))  # the year, for "May 2026"


def allowed_from(sources: Iterable[Any]) -> Allowed:
    allowed = Allowed()
    for s in sources:
        allowed.add(s)
    return allowed


def _word_dates(text: str) -> tuple[list[re.Match], str]:
    """Word-form dates in `text`, and the text with them blanked out."""
    found: list[re.Match] = []
    for rx in WORD_DATE_RES:
        found += list(rx.finditer(text))
        text = rx.sub(" ", text)
    return found, text


def _word_date_ok(m: re.Match, allowed_dates: set[str]) -> bool:
    month, day, year = _MONTH_NUM[m.group("mon").lower()[:3]], int(m.group("day")), m.group("year")
    return any(len(d) == 10 and int(d[5:7]) == month and int(d[8:]) == day and (year is None or d[:4] == year)
               for d in allowed_dates)


def _split(text: str) -> tuple[list[str], list[re.Match], list[str]]:
    """(ISO date tokens, word-form date matches, other number tokens); list markers excluded."""
    iso = DATE_TOKEN_RE.findall(text)
    words, rest = _word_dates(DATE_TOKEN_RE.sub(" ", text))
    return iso, words, NUMBER_RE.findall(LIST_MARKER_RE.sub(r"\1", rest))


def extract_numbers(text: str) -> tuple[list[str], list[str]]:
    """(date tokens, other number tokens) in `text`. Date tokens are ISO ("2026-06-11", "2026-06")
    or word-form ("June 11"); list markers are excluded."""
    iso, words, numbers = _split(text)
    return iso + [m.group(0) for m in words], numbers


def unverified_numbers(answer: str, sources: Iterable[Any]) -> list[str]:
    """Every number/date token in `answer` that is not in `sources` (tool results and args)."""
    allowed = allowed_from(sources)
    iso, words, numbers = _split(answer)
    bad = [d for d in iso if d not in allowed.dates]
    bad += [m.group(0) for m in words if not _word_date_ok(m, allowed.dates)]
    bad += [n for n in numbers if _to_decimal(n) not in allowed.numbers]
    return bad
