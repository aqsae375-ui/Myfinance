import pytest

from scripts.generate_data import build_statement, write_outputs


@pytest.fixture(scope="session")
def statement_txns():
    return build_statement(seed=42)


@pytest.fixture(scope="session")
def statement_csv(statement_txns, tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("raw")
    csv_path, _ = write_outputs(statement_txns, out_dir, seed=42)
    return csv_path


@pytest.fixture(scope="session")
def synthetic_df(statement_csv):
    """The synthetic statement through the real pipeline: parse, redact, ingest, categorize
    (rules + offline fake LLM), load. The shape the analytics engine reads in production."""
    from src.categorizer import categorize_transactions
    from src.db import connect, ingest, load_transactions
    from src.llm.fake_provider import FakeProvider
    from src.parser import parse_statement

    conn = connect(":memory:")
    ingest(conn, parse_statement(statement_csv))
    categorize_transactions(conn, provider=FakeProvider())
    df = load_transactions(conn)
    conn.close()
    return df


@pytest.fixture
def make_txns():
    """Factory for small hand-built transaction frames: rows of (date, description, amount, category)."""
    import pandas as pd

    def build(rows):
        df = pd.DataFrame(rows, columns=["date", "redacted_description", "amount", "category"])
        df.insert(0, "id", range(1, len(df) + 1))
        df["date"] = pd.to_datetime(df["date"])
        df["raw_description"] = df["redacted_description"]
        df["type"] = df["amount"].map(lambda a: "credit" if a >= 0 else "debit")
        return df

    return build
