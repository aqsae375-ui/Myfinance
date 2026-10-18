from scripts.print_month_report import report
from src.categorizer import categorize_transactions
from src.db import connect, ingest_csv
from src.llm.fake_provider import FakeProvider


def test_report_renders_for_first_and_last_month(statement_csv, tmp_path):
    db = tmp_path / "finance.db"
    ingest_csv(statement_csv, db)
    assert "run `python -m src.categorizer` first" in report(None, db)

    conn = connect(db)
    categorize_transactions(conn, provider=FakeProvider())
    conn.close()

    first = report("2026-03", db)
    assert "nothing (first month in data)" in first and "(no previous month)" in first
    last = report(None, db)
    assert "=== Report for 2026-08" in last and "POSSIBLY UNUSED" in last and "FITPRO GYM MEMBERSHIP" in last
