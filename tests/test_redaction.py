import re

import pytest
from faker import Faker

from scripts.generate_data import INJECTION_TEXT
from src.redaction import redact

# 7+ digits, allowing phone-style separators between them: nothing like this may survive.
DIGIT_RUN_RE = re.compile(r"\d(?:[\s().+-]*\d){6,}")


@pytest.fixture(scope="module")
def fake():
    f = Faker("en_US")
    f.seed_instance(1234)
    return f


def test_no_faker_phone_survives(fake):
    for _ in range(500):
        phone = fake.phone_number()
        out = redact(f"ZELLE TO JOHN SMITH {phone} MEMO LUNCH")
        assert not DIGIT_RUN_RE.search(out), (phone, out)


def test_no_faker_email_survives(fake):
    for _ in range(500):
        email = fake.email()
        out = redact(f"E-TRANSFER FROM {email} REF DINNER")
        assert "@" not in out and email not in out
        assert "[EMAIL]" in out


@pytest.mark.parametrize("phone", [
    "+44 20 7946 0958", "(555) 123-4567", "555.123.4567 ext. 89", "+1-555-123-4567x123",
    "001-555-123-4567", "5551234567",
])
def test_phone_formats(phone):
    out = redact(f"CALL {phone} TODAY")
    assert not DIGIT_RUN_RE.search(out), out


@pytest.mark.parametrize("text, secret, label", [
    ("RENT ACCT 1234567890", "1234567890", "[ACCOUNT]"),
    ("ACCOUNT NO: 00123456", "00123456", "[ACCOUNT]"),
    ("WIRE IBAN GB29NWBK60161331926819", "GB29NWBK60161331926819", "[ACCOUNT]"),
    ("WIRE IBAN GB29 NWBK 6016 1331 9268 19", "6016 1331", "[ACCOUNT]"),
    ("CARD 4111 1111 1111 1111 PAYMENT", "4111", "[CARD]"),
    ("CARD 4111111111111111 PAYMENT", "4111111111111111", "[CARD]"),
    ("PAYROLL REF 987654321", "987654321", "[ACCOUNT]"),
])
def test_account_numbers(text, secret, label):
    out = redact(text)
    assert secret not in out and label in out


@pytest.mark.parametrize("text, secret", [
    ("RENT PAYMENT TO Maria Gonzalez ACCT 1234", "Gonzalez"),
    ("ZELLE TO JOHN SMITH 555-123-4567", "SMITH"),
    ("E-TRANSFER FROM Kelly O'Neil kelly@example.com", "Neil"),
    ("INTL WIRE TO Ana Lopez IBAN GB29NWBK60161331926819", "Lopez"),
    ("VENMO FROM DEREK", "DEREK"),
])
def test_names(text, secret):
    out = redact(text)
    assert secret not in out and "[NAME]" in out


def test_known_names_always_masked():
    out = redact("PAYMENT FROM ALEX RIVERA CARD SERVICES; RIVERA A.", known_names=["Alex Rivera"])
    assert "RIVERA" not in out.upper() and "ALEX" not in out.upper()


@pytest.mark.parametrize("text", [
    "NETFLIX.COM", "GREENLEAF GROCERY #0412", "CITY POWER & LIGHT", "SHELL OIL 5741",
    "AMZN MKTP US*2K4L81", "LUIGI'S TRATTORIA", "TRANSFER TO SAVINGS", "OVERDRAFT FEE",
    INJECTION_TEXT,
])
def test_non_pii_unchanged(text):
    assert redact(text) == text


def test_generated_statement_has_no_pii_after_redaction(statement_txns):
    for t in statement_txns:
        out = redact(t.description)
        assert "@" not in out and not DIGIT_RUN_RE.search(out), out
        for secret in t.pii:
            assert secret.lower() not in out.lower(), (secret, out)
