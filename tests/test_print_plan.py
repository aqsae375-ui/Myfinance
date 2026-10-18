from scripts.print_plan import main
from src.advisor import load_goal
from src.categorizer import categorize_transactions
from src.db import connect, ingest_csv
from src.llm.fake_provider import FakeProvider


def test_print_plan_end_to_end(statement_csv, tmp_path, capsys):
    db = tmp_path / "finance.db"
    ingest_csv(statement_csv, db)
    assert "run `python -m src.categorizer` first" in main(["--db", str(db)])
    conn = connect(db)
    categorize_transactions(conn, provider=FakeProvider())
    conn.close()

    out = main(["--db", str(db)])          # no goal in the DB -> the sample goal
    assert "Goal: Emergency fund: Rs 10,000.00 by 2027-03-31" in out
    assert "[x] 1. Plan big Shopping purchases ahead" in out and "How I got this:" in out
    assert "Months to goal" in out and "8 mo (late)" in out and "6 mo (on track)" in out
    assert "5 model-written and verified, 0 replaced by template" in out

    out = main(["--db", str(db), "--monthly", "1500", "--save-goal", "--accept", "cancel:FITPRO GYM MEMBERSHIP"])
    assert "1 of 5 suggestions accepted" in out and "[ ] 1." in out and "[x] 4." in out
    conn = connect(db)
    stored = load_goal(conn)
    assert (stored.kind, stored.amount, stored.name) == ("monthly_savings", 1500.0, "Monthly savings")
    conn.close()
    assert "Monthly savings: save Rs 1,500.00 every month" in main(["--db", str(db)])  # stored goal reused
    assert "Goal: none set" in main(["--db", str(db), "--no-goal"])
