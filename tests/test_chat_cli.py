from scripts import chat_cli
from src.categorizer import categorize_transactions
from src.db import connect, ingest
from src.llm.fake_provider import FakeProvider
from src.parser import parse_statement


def test_demo_runs_offline(statement_csv, tmp_path, capsys):
    db = tmp_path / "finance.db"
    conn = connect(db)
    ingest(conn, parse_statement(statement_csv))
    categorize_transactions(conn, provider=FakeProvider())
    conn.close()

    assert chat_cli.main(["--db", str(db), "--demo", "--provider", "fake",
                          "--trace-file", str(tmp_path / "t.jsonl")]) == 0
    out = capsys.readouterr().out
    assert "Finance agent (fake)" in out and out.count("You: ") == len(chat_cli.DEMO_QUESTIONS)
    assert "get_category_spending" in out and "2026-04" in out and "FITPRO GYM MEMBERSHIP" in out
    assert len((tmp_path / "t.jsonl").read_text().splitlines()) == len(chat_cli.DEMO_QUESTIONS)


def test_missing_db(tmp_path, capsys):
    assert chat_cli.main(["--db", str(tmp_path / "none.db"), "--demo"]) == 1
    assert "No database" in capsys.readouterr().out
