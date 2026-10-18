import io
import math

import pandas as pd
import pytest

from scripts.generate_data import MONTHS
from src.parser import OUTPUT_COLUMNS, StatementParseError, parse_amount, parse_statement


def _parse(text: str) -> pd.DataFrame:
    return parse_statement(io.StringIO(text))


def test_both_date_formats_parse_to_same_day():
    df = _parse("Date,Description,Amount\n2026-03-05,A,-1.00\n05/03/2026,B,-2.00\n")
    assert list(df["date"]) == [pd.Timestamp("2026-03-05")] * 2


def test_unsupported_date_raises_with_line_number():
    with pytest.raises(StatementParseError, match="line 3"):
        _parse("Date,Description,Amount\n2026-03-05,A,-1\n03-05-2026,B,-2\n")


def test_bad_amount_raises():
    with pytest.raises(StatementParseError, match="line 2"):
        _parse("Date,Description,Amount\n2026-03-05,A,twelve\n")


def test_missing_column_raises():
    with pytest.raises(StatementParseError, match="amount"):
        _parse("Date,Description\n2026-03-05,A\n")


@pytest.mark.parametrize("raw, expected", [
    ("-1,650.00", -1650.0), ("4,850.00", 4850.0), ("(12.50)", -12.5),
    ("$9.99", 9.99), (" -0.5 ", -0.5), ("+3", 3.0), ("1 200.00", 1200.0),
])
def test_parse_amount(raw, expected):
    assert parse_amount(raw) == expected


@pytest.mark.parametrize("raw", ["", "abc", "1.2.3", "--5"])
def test_parse_amount_invalid_is_nan(raw):
    assert math.isnan(parse_amount(raw))


def test_debit_credit_columns_and_aliases():
    df = _parse("Posting Date,Details,Debit,Credit\n2026-03-01,RENT,1650.00,\n2026-03-25,PAY,,4850.00\n")
    assert list(df["amount"]) == [-1650.0, 4850.0]
    assert list(df["type"]) == ["debit", "credit"]


def test_blank_rows_dropped_and_whitespace_collapsed():
    df = _parse("Date,Description,Amount\n2026-03-05,  NETFLIX   .COM ,-15.49\n,,\n")
    assert len(df) == 1 and df.loc[0, "description"] == "NETFLIX .COM"


def test_generated_statement_parses(statement_csv, statement_txns):
    df = parse_statement(statement_csv)
    assert list(df.columns) == OUTPUT_COLUMNS
    assert len(df) == len(statement_txns)
    assert pd.api.types.is_datetime64_any_dtype(df["date"]) and df["date"].notna().all()
    assert pd.api.types.is_float_dtype(df["amount"]) and df["amount"].notna().all()
    assert sorted(set(zip(df["date"].dt.year, df["date"].dt.month))) == MONTHS
    assert ((df["amount"] < 0) == (df["type"] == "debit")).all()
    # Values survive the CSV round trip (incl. "1,650.00" thousands separators and mixed dates).
    assert list(df["amount"]) == [t.amount for t in statement_txns]
    assert list(df["date"].dt.date) == [t.date for t in statement_txns]
