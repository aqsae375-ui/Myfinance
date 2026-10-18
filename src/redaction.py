"""Mask PII in transaction descriptions before any text reaches a model.

Placeholders: [EMAIL] [ACCOUNT] [CARD] [PHONE] [NAME] [REF] [ADDRESS].
Order matters: specific patterns (email, IBAN, card) run before generic digit runs and phones,
and names run last so the name rules can stop at placeholders instead of eating numbers.

Pakistani bank exports ("Money Received from ZARA FICTIONAL KH STAN(200201) IBFT from EASYPAISA"):
- STAN codes and reference numbers -> [REF]; the branch address after an ATM withdrawal -> [ADDRESS].
- Whatever follows "from" / "to" / "Received from" is a person's name, in any casing and even truncated,
  unless it is a bank or wallet (EASYPAISA, HBL, ...), a currency, a known transaction word, or a
  merchant a categorization rule recognizes. Banks and merchants are kept: the categorizer needs them.
- Merchant names after "POS Purchase" are never masked.
Known limitation: an ALL-CAPS name with no "from"/"to" before it is not detected.
"""
from __future__ import annotations

import re
from typing import Iterable

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){3,7}(?: ?[A-Z0-9]{1,3})?\b")
CARD_RE = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")  # 13-19 digits, optional separators
ACCOUNT_KEYWORD_RE = re.compile(
    r"(?P<kw>\b(?:acct|account|a/c)\b\.?\s*(?:no\.?|number|#)?\s*[:#]?\s*)(?P<num>[\dX*-]*\d{4,})",
    re.IGNORECASE,
)
LONG_DIGITS_RE = re.compile(r"\b\d{8,}\b")
INTL_PHONE_RE = re.compile(r"(?<![\w+])\+\d{1,3}(?:[\s.-]?\(?\d{1,4}\)?){2,5}(?:\s*(?:x|ext\.?)\s*\d{1,6})?")
PHONE_RE = re.compile(
    r"(?<![\w+])(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}"
    r"(?:\s*(?:[xX]|ext\.?)\s*\d{1,6})?(?!\w)"
)
# "STAN(200201)", "STAN (300301)", "STAN: 100101"
STAN_RE = re.compile(r"\bSTAN\s*(?:\(\s*\d+\s*\)|[:#]?\s*\d{3,})", re.IGNORECASE)
REF_RE = re.compile(r"(?P<kw>\b(?:REF(?:ERENCE)?|TXN|TRX|RRN|TID|TRANS(?:ACTION)?\s*ID)\b\.?\s*(?:NO\.?|#|:)?\s*[:#]?\s*)"
                    r"(?P<ref>(?=[A-Z0-9-]*\d)[A-Z0-9-]{4,})", re.IGNORECASE)
# The branch street address an ATM export appends after the STAN code.
ATM_ADDRESS_RE = re.compile(r"(?P<head>\bCASH\s+WITHDRAWAL\b(?:\s+STAN\s+\[REF\])?)\s+(?P<addr>[^\[\s].*)$",
                            re.IGNORECASE)

# Person-to-person keywords: whatever follows them is a person's name, whatever the casing.
P2P_NAME_RE = re.compile(
    r"(?P<prefix>\b(?:ZELLE|VENMO|PAYPAL|E-?TRANSFER|INTERAC|TRANSFER|WIRE)(?:\s+(?:TO|FROM))?\s+)"
    r"(?P<name>[A-Z][A-Z'.-]*(?:\s+[A-Z][A-Z'.-]*){0,3})",
    re.IGNORECASE,
)
# "from X", "to X", "Received from X": up to 4 words, cut at the first stop word below.
FROM_TO_NAME_RE = re.compile(
    r"(?P<prefix>\b(?:FROM|TO)\s+)(?P<name>[A-Z][A-Z'.-]*(?:\s+[A-Z][A-Z'.-]*){0,3})", re.IGNORECASE)

NAME_STOPWORDS = {
    "SAVINGS", "CHECKING", "ACCOUNT", "ACCT", "IBAN", "REF", "MEMO", "ON", "FOR", "CONF",
    "ID", "VIA", "THE", "MY", "CARD", "LOAN", "BANK", "PAYMENT",
}
# Banks and wallets are institutions, not people: kept so rules can use them.
BANKS = {
    "EASYPAISA", "JAZZCASH", "JAZZ", "HBL", "ALFALAH", "MEEZAN", "UBL", "MCB", "NBP", "ABL", "ALLIED", "ASKARI",
    "FAYSAL", "BAHL", "HABIB", "BANKISLAMI", "SADAPAY", "NAYAPAY", "KEENU", "UPAISA", "STANDARD", "SCB", "JS",
    "SONERI", "SUMMIT", "SILKBANK", "DUBAI", "ISLAMIC", "RAAST", "IBFT", "1LINK", "1-LINK",
}
CURRENCIES = {"PKR", "USD", "EUR", "GBP", "AED", "SAR", "CAD", "AUD", "RS"}
# Transaction words that end a name ("from ZARA FICTIONAL STAN ..."), and that never form a name themselves.
TXN_WORDS = {
    "STAN", "IBFT", "RAAST", "P2P", "POS", "PURCHASE", "ATM", "CASH", "WITHDRAWAL", "FUND", "TRANSFER", "PYMT",
    "PAYMENT", "PAYMENTS", "MONEY", "RECEIVED", "PROFIT", "WITHHOLDING", "TAX", "TAXES", "DEBIT", "CREDIT",
    "CHARGES", "CHARGE", "PLUS", "FED", "FEE", "FEES", "ANNUAL", "INTL", "MASTER", "MASTERCARD", "VISA",
    "INC", "LTD", "LIMITED", "LLC", "CORP", "CO", "STORE", "STORES", "SHOP", "MART", "MARKET", "FOODS",
    "SUPER", "SERVICES", "SERVICE", "ONLINE", "MOBILE", "INTEREST", "SALARY", "REFUND", "REVERSAL", "LINK",
    "WALLET", "BRANCH", "INWARD", "OUTWARD", "REMITTANCE", "PURCHASES", "OF", "AND", "IN", "AT", "BY",
} | NAME_STOPWORDS | BANKS | CURRENCIES

_WORD = r"[A-Z][a-z]+(?:[-'][A-Z][a-z]+|[A-Z][a-z]+)?"
TITLE_RUN_RE = re.compile(rf"\b{_WORD}(?:\s+(?:[A-Z]\.\s+)?{_WORD})+\b")
POS_MERCHANT_RE = re.compile(r"\bPOS\s+PURCHASE\b.*?(?=\bSTAN\b|$)", re.IGNORECASE)


def redact(text: str, known_names: Iterable[str] = ()) -> str:
    """Return `text` with PII replaced by placeholders. `known_names` (e.g. the account holder)
    are always masked, including each part of 3+ letters on its own."""
    if not text:
        return text
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = IBAN_RE.sub("[ACCOUNT]", text)
    text = CARD_RE.sub("[CARD]", text)
    text = ACCOUNT_KEYWORD_RE.sub(lambda m: f"{m.group('kw')}[ACCOUNT]", text)
    text = LONG_DIGITS_RE.sub("[ACCOUNT]", text)
    text = STAN_RE.sub("STAN [REF]", text)
    text = REF_RE.sub(lambda m: f"{m.group('kw')}[REF]", text)
    text = INTL_PHONE_RE.sub("[PHONE]", text)
    text = PHONE_RE.sub("[PHONE]", text)
    text = ATM_ADDRESS_RE.sub(lambda m: f"{m.group('head')} [ADDRESS]", text)
    text = _redact_known_names(text, known_names)
    text = P2P_NAME_RE.sub(_mask_p2p_name, text)
    text = _outside_pos_merchant(text, lambda part: FROM_TO_NAME_RE.sub(_mask_from_to_name, part))
    text = _outside_pos_merchant(text, lambda part: TITLE_RUN_RE.sub(_mask_title_run, part))
    return text


def _redact_known_names(text: str, known_names: Iterable[str]) -> str:
    names = [n.strip() for n in known_names if n and n.strip()]
    parts = [p for n in names for p in n.split() if len(p) >= 3]
    for term in sorted(set(names + parts), key=len, reverse=True):  # full names before parts
        text = re.sub(rf"\b{re.escape(term)}\b", "[NAME]", text, flags=re.IGNORECASE)
    return text


def _outside_pos_merchant(text: str, fn) -> str:
    """Apply `fn` everywhere except the merchant segment after "POS Purchase" (up to STAN)."""
    out, last = [], 0
    for m in POS_MERCHANT_RE.finditer(text):
        out += [fn(text[last:m.start()]), m.group(0)]
        last = m.end()
    out.append(fn(text[last:]))
    return "".join(out)


def _mask_p2p_name(m: re.Match) -> str:
    tokens = m.group("name").split()
    n_name = 0
    for tok in tokens:
        if tok.upper().strip(".") in NAME_STOPWORDS:
            break
        n_name += 1
    if n_name == 0:  # e.g. "TRANSFER TO SAVINGS"
        return m.group(0)
    rest = tokens[n_name:]
    return m.group("prefix") + "[NAME]" + (" " + " ".join(rest) if rest else "")


def _is_merchant(words: list[str]) -> bool:
    from src.rules import find_rule  # local import: rules -> categories only, no cycle
    return find_rule(" ".join(words)) is not None  # any rule match, deciding or only a hint


def _mask_from_to_name(m: re.Match) -> str:
    tokens = m.group("name").split()
    n_name = 0
    for tok in tokens:
        if tok.upper().strip(".,") in TXN_WORDS:
            break
        n_name += 1
    if n_name == 0 or _is_merchant(tokens[:n_name]):
        return m.group(0)
    rest = tokens[n_name:]
    return m.group("prefix") + "[NAME]" + (" " + " ".join(rest) if rest else "")


def _mask_title_run(m: re.Match) -> str:
    """Mask runs of 2+ capitalized words ("Maria Gonzalez"), but not transaction phrases: a known
    word ("Money Received", "Withholding Tax Debit", "LaunchGood Inc") splits the run, and only
    pieces of 2+ unknown words are masked."""
    tokens = m.group(0).split()
    out, run = [], []

    def flush() -> None:
        if len(run) >= 2:
            out.append("[NAME]")
        else:
            out.extend(run)
        run.clear()

    for tok in tokens:
        if tok.upper().strip(".") in TXN_WORDS:
            flush()
            out.append(tok)
        else:
            run.append(tok)
    flush()
    return " ".join(out)
