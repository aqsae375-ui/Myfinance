"""Synthetic Pakistani bank statement (Jan-Dec 2025, PKR) with the same structure as real exports:
Date, Value Date, Description, Amount, Balance. Every row type the rules must handle appears at least once.
All names are fictional placeholders (ZARA FICTIONAL, ...); no real person, account or STAN code.

`ROWS` = (date, description, amount, expected category or None if it must go to the LLM).
`write_fixture(path)` writes the CSV; tests/fixtures/pk_statement.csv is a committed copy.
"""
from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "pk_statement.csv"
OPENING_BALANCE = 150000.00
FAKE_PEOPLE = ("ZARA FICTIONAL", "BILAL TESTNAME", "OMAR SAMPLE KH", "HINA PLACEHOLDER")
DINING = ("CHEEZIOUS", "KFC", "LAYERS BAKESHOP", "THE MONAL")
SOURCES = ("EASYPAISA", "ALFALAH", "HBL BANK", "JAZZCASH BANK")


def _rows() -> list[tuple[date, str, float, str | None]]:
    rows: list[tuple[date, str, float, str | None]] = []
    for m in range(1, 13):
        d = lambda day: date(2025, m, day)  # noqa: E731
        rows.append((d(1), "Payment of Profit", round(180 + m * 7.5, 2), "Interest"))
        rows.append((d(1), "Withholding Tax Debit", -round((180 + m * 7.5) * 0.15, 2), "Tax"))
        rows.append((d(5), f"1-Link ATM Cash Withdrawal STAN ({100100 + m}) TESTVILLE MAIN BRANCH - II", -10000.00,
                     "Cash Withdrawal"))
        src = SOURCES[m % 4]
        if src == "ALFALAH":
            desc = f"Money Received from ALFALAH STAN({200200 + m}) IBFT from ALFALAH"
        else:
            desc = f"Money Received from {FAKE_PEOPLE[m % 4]} STAN({200200 + m}) IBFT from {src}"
        rows.append((d(10), desc, 25000.00, "Transfers In"))
        rows.append((d(15), f"POS Purchase {DINING[m % 4]} STAN ({300300 + m})", -(1200.00 + m * 110), "Dining Out"))
        grocer = "AL-FATAH STORE" if m % 2 else "SABZI MANDI"
        rows.append((d(20), f"POS Purchase {grocer} STAN ({400400 + m})", -(3800.00 + m * 25), "Groceries"))
    rows += [
        (date(2025, 1, 25), "MasterCard Annual Fee 2025", -2500.00, "Bank Fees"),
        (date(2025, 3, 28), "BANK CHARGES", -150.00, "Bank Fees"),
        (date(2025, 4, 18), "1-Link Fee Charges ATM Charges", -23.40, "Bank Fees"),
        (date(2025, 5, 12), "RAAST PYMT STAN (500501) MEEZAN BANK LIMITED Raast P2P Fund transfer from HINA PLACEHOLDER",
         8000.00, "Transfers In"),
        (date(2025, 6, 28), "BANK CHARGES STAN (500601) FED ON CHARGES", -26.00, "Bank Fees"),
        (date(2025, 7, 14), "INTL. MASTER POS Purchase LaunchGood Inc. STAN (500701) USD 10.00 to PKR 2850",
         -2850.00, "Charity/Donations"),
        (date(2025, 7, 14), "Charges Taxes Plus FED STAN (500701) LaunchGood Inc. to PKR 128", -128.00, "Tax"),
        (date(2025, 8, 9), "POS Purchase ZEPHYR TRADING CO STAN (500801)", -1500.00, None),
        (date(2025, 9, 6), "ATM Cash Withdrawal STAN (500901) SAMPLETOWN MARKAZ", -50000.00, "Cash Withdrawal"),
        (date(2025, 10, 3), "POS Purchase SAVOUR FOODS (BLUE AREA) STAN (501001)", -1850.00, "Dining Out"),
        (date(2025, 11, 22), "BANK CHARGES STAN (501101) MASTERCARD ANNUAL FEE 2025", -1800.00, "Bank Fees"),
    ]
    rows.sort(key=lambda r: r[0])
    return rows


ROWS = _rows()


def write_fixture(path: Path, rows=ROWS) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    balance = OPENING_BALANCE
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Date", "Value Date", "Description", "Amount", "Balance"])
        for d, desc, amount, _ in rows:
            balance = round(balance + amount, 2)
            w.writerow([d.isoformat(), (d + timedelta(days=1)).isoformat(), desc, f"{amount:,.2f}", f"{balance:,.2f}"])
    return path


if __name__ == "__main__":
    print(write_fixture(FIXTURE))
