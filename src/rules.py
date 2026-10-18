"""Keyword/regex categorization rules. They run on redacted_description, before any LLM.

Stage 2d: a rule only DECIDES when its confidence is at least RULE_THRESHOLD (0.8). Every rule has a
risk rating: "ambiguous" rules (a generic word that can name two kinds of business, e.g. MART, MANDI,
CAFE) sit below the threshold, so their match is passed to the LLM as a hint only.

To extend: add a Rule row to RULES. Among deciding rules the first match wins, so put specific rules
(UBER EATS) before general ones (UBER); an ambiguous match is used only if no deciding rule matches.
`direction` restricts a rule to money in ('credit') or out ('debit'). Every pattern is matched as whole
words: it may not start or end inside a word ("MART" does not match "MARTIN").
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.categories import is_valid_category

RULE_THRESHOLD = 0.8        # a rule decides only at or above this confidence
AMBIGUOUS_CONFIDENCE = 0.6  # ambiguous rules: hint for the LLM, never a decision
RISKS = ("clear", "ambiguous")


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: str
    category: str
    confidence: float = 0.95
    direction: str = "any"  # any | debit | credit
    risk: str = "clear"     # clear | ambiguous
    regex: re.Pattern = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not is_valid_category(self.category):
            raise ValueError(f"Rule {self.name!r}: {self.category!r} is not in the taxonomy")
        if self.direction not in ("any", "debit", "credit"):
            raise ValueError(f"Rule {self.name!r}: bad direction {self.direction!r}")
        if self.risk not in RISKS:
            raise ValueError(f"Rule {self.name!r}: bad risk {self.risk!r}")
        if self.risk == "ambiguous" and self.confidence >= RULE_THRESHOLD:
            raise ValueError(f"Rule {self.name!r}: an ambiguous rule must stay below {RULE_THRESHOLD}")
        # Whole-word matching for every rule: no word character directly before or after the match.
        object.__setattr__(self, "regex", re.compile(rf"(?<!\w)(?:{self.pattern})(?!\w)", re.IGNORECASE))

    @property
    def decides(self) -> bool:
        return self.confidence >= RULE_THRESHOLD


# Merchant keywords for "POS Purchase <merchant>" rows (Pakistani bank exports). Unknown POS merchants
# match no rule and go to the LLM. Clear words name one kind of business; generic words (and brand names
# that are also common words) can mean two things, so they are only a hint.
POS_DINING_CLEAR = (r"CHEEZIOUS|KFC|MCDONALD'?S?|HARDEE'?S?|DOMINO'?S?|OPTP|MONAL|BUNDU\s*KHAN|TEHZEEB|"
                    r"GLORIA\s*JEANS|PIZZA|BIRYANI|KARAHI|TIKKA|BBQ|RESTAURANT|BURGER|SHAWARMA|DHABA|"
                    r"FOODS?\s*(STREET|POINT|COURT)")
POS_DINING_GENERIC = (r"MANDI|DOUGH|LAYERS|SAVOUR|SUBWAY|BAKESHOP|BAKERS?|BAKERY|CAFE|COFFEE|CHAI|CHAAYE|SWEETS|"
                      r"GRILL|HOTEL")
POS_GROCERY_CLEAR = (r"SUPERMARKET|IMTIAZ|CARREFOUR|METRO\s*(CASH|C&C)|AL[-\s]?FATAH|CHASE\s*UP|JALAL\s*SONS|"
                     r"KARYANA|GROCER(Y|IES|S)?|SABZI")
POS_GROCERY_GENERIC = r"MART|SUPER\s*STORE|SUPERSTORE|GENERAL\s*STORE|STORE|GREEN\s*VALLEY|FRUITS?|METRO"

A = AMBIGUOUS_CONFIDENCE

RULES: tuple[Rule, ...] = (
    # ---- Pakistani bank exports: transaction-type phrases first (never people's names) ----
    # Tax before fees and charity: "Charges Taxes Plus FED ... <merchant>" is tax, whatever the merchant.
    Rule("withholding_tax", r"\bWITHHOLDING\s+TAX\b|\bW\.?H\.?T\.?\s+(DEBIT|DEDUCTION)\b", "Tax", 0.98, "debit"),
    Rule("charges_taxes_fed", r"\bCHARGES\s+TAXES\b|\bTAXES\s+PLUS\s+FED\b", "Tax", 0.95, "debit"),
    Rule("profit_payment", r"\bPAYMENT\s+OF\s+PROFIT\b|\bPROFIT\s+(PAID|CREDIT|PAYMENT)\b", "Interest", 0.98, "credit"),
    Rule("atm_cash", r"\b(ATM\s+)?CASH\s+WITHDRAWAL\b|\bATM\s+(WDL|WITHDRAWAL)\b", "Cash Withdrawal", 0.98, "debit"),
    Rule("pk_bank_fees", r"\bBANK\s+CHARGES\b|\b1-?LINK\s+FEE\b|\bANNUAL\s+FEE\b|\bSMS\s+(ALERT\s+)?CHARGES\b|"
                         r"\bDEBIT\s+CARD\s+(FEE|CHARGES)\b|\bFED\b", "Bank Fees", 0.95, "debit"),
    Rule("charity", r"\bLAUNCH\s*GOOD\b|\bDONAT(E|ION|IONS)\b|\bCHARITY\b|\bEDHI\b|\bSHAUKAT\s+KHANUM\b|"
                    r"\bAKHUWAT\b|\bSAYLANI\b|\bZAKAT\b", "Charity/Donations", 0.95, "debit"),
    Rule("money_received", r"\bMONEY\s+RECEIVED\b|\bRAAST\b|\bIBFT\b|\bP2P\b|\bEASYPAISA\b|\bJAZZ\s*CASH\b|"
                           r"\bFUND\s+TRANSFER\b|\bINWARD\s+(REMITTANCE|TRANSFER)\b", "Transfers In", 0.95, "credit"),
    Rule("money_sent", r"\bRAAST\b|\bIBFT\b|\bP2P\b|\bEASYPAISA\b|\bJAZZ\s*CASH\b|\bFUND\s+TRANSFER\b",
         "Transfers", 0.9, "debit"),
    Rule("pos_dining", rf"\bPOS\s+PURCHASE\b.*\b({POS_DINING_CLEAR})", "Dining Out", 0.9, "debit"),
    Rule("pos_grocery", rf"\bPOS\s+PURCHASE\b.*\b({POS_GROCERY_CLEAR})", "Groceries", 0.9, "debit"),
    Rule("pos_dining_generic", rf"\bPOS\s+PURCHASE\b.*\b({POS_DINING_GENERIC})", "Dining Out", A, "debit",
         "ambiguous"),
    Rule("pos_grocery_generic", rf"\bPOS\s+PURCHASE\b.*\b({POS_GROCERY_GENERIC})", "Groceries", A, "debit",
         "ambiguous"),

    # Food delivery before rideshare: "UBER EATS" is dining, "UBER TRIP" is transport.
    Rule("food_delivery", r"\b(UBER\s*EATS|DOORDASH|GRUBHUB|DELIVEROO|POSTMATES|JUST\s*EAT|SKIP\s*THE\s*DISHES)\b",
         "Dining Out"),
    Rule("amazon_prime_video", r"\b(PRIME\s*VIDEO|AMAZON\s*PRIME|AMZN\s*PRIME)\b", "Subscriptions", 0.9),

    # Income
    Rule("payroll", r"\b(PAYROLL|SALARY|DIRECT\s*DEP(OSIT)?|DIR\s*DEP|WAGES?)\b", "Income", direction="credit"),
    Rule("interest_dividend", r"\b(INTEREST\s*(PAID|EARNED|CREDIT)|DIVIDEND)\b", "Interest", 0.9, "credit"),
    Rule("tax_refund", r"\b(TAX\s*REF(UND)?|IRS\s*TREAS)\b", "Income", 0.9, "credit"),

    # Transfers (person-to-person and between accounts). Money in from a person is Transfers In.
    # A wallet cash-out moves your own money (not income), so it falls through to "p2p" -> Transfers.
    Rule("p2p_in", r"\b(ZELLE|VENMO|CASH\s*APP|E-?TRANSFER|INTERAC|PAYPAL\s*\*?\s*FROM\b)(?!.*CASH\s*OUT)",
         "Transfers In", direction="credit"),
    Rule("p2p", r"\b(ZELLE|VENMO|CASH\s*APP|E-?TRANSFER|INTERAC|PAYPAL\s*\*?\s*(TO|FROM)\b)", "Transfers"),
    Rule("wire", r"\b(WIRE|INTL\s*TRANSFER|SWIFT)\b", "Transfers", 0.9),
    Rule("account_transfer", r"\b(TRANSFER\s+(TO|FROM)|ONLINE\s*TRANSFER|XFER)\b", "Transfers", 0.85),

    # Housing. A bare RENT can be a rental ("RENT A CAR"), so only rent phrases decide.
    Rule("rent", r"\b(RENT\s+PAYMENT|MONTHLY\s+RENT|HOUSE\s+RENT|LEASE\s*PAYMENT|PROPERTY\s*MGMT|LANDLORD)\b",
         "Rent"),
    Rule("rent_word", r"\bRENT\b", "Rent", A, risk="ambiguous"),

    # Utilities. POWER/ELECTRIC/HYDRO/INTERNET/SPECTRUM also name shops and products (power tools, internet cafe).
    Rule("utilities", r"\b(ELECTRICITY|WATER\s*(UTILITY|DEPT|BILL)|GAS\s*(CO|COMPANY|UTILITY|BILL)|BROADBAND|"
                      r"COMCAST|XFINITY|VERIZON|AT&T|T-?MOBILE|UTILITY|UTILITIES|ELECTRIC\s+BILL)\b",
         "Utilities", 0.9),
    Rule("utilities_generic", r"\b(POWER|ELECTRIC|HYDRO|INTERNET|SPECTRUM)\b", "Utilities", A, risk="ambiguous"),

    # Subscriptions (recurring digital services and memberships)
    Rule("streaming", r"\b(NETFLIX|SPOTIFY|HULU|DISNEY\s*PLUS|DISNEY\+|HBO\s*MAX|APPLE\.COM/BILL|YOUTUBE\s*PREMIUM|"
                      r"PARAMOUNT\+?|AUDIBLE|DROPBOX|ICLOUD|ADOBE|MICROSOFT\s*365|CHATGPT|PATREON)\b", "Subscriptions"),
    Rule("membership", r"\b(GYM|FITNESS|SUBSCRIPTION)\b", "Subscriptions", 0.85),
    Rule("membership_generic", r"\bMEMBERSHIP\b", "Subscriptions", A, risk="ambiguous"),  # also store/club cards

    # Groceries
    Rule("groceries", r"\b(GROCER(Y|IES|S)?|SUPERMARKET|SUPERMKT|TRADER\s*JOE'?S?|KROGER|SAFEWAY|ALDI|LIDL|"
                      r"COSTCO|PUBLIX|WEGMANS|TESCO|SAINSBURY'?S?)", "Groceries", 0.9),
    Rule("groceries_generic", r"\bFOODS\b", "Groceries", A, risk="ambiguous"),  # also restaurants, food makers

    # Dining out. CAFE/COFFEE/GRILL/KITCHEN/BAKERY/SUBWAY/PUB can also be a shop, a product or transit.
    Rule("dining", r"\b(RESTAURANT|TRATTORIA|PIZZERIA|PIZZA|SUSHI|STARBUCKS|BISTRO|DINER|BURGER|TACO|BAR\s*&|"
                   r"MCDONALD'?S?|CHIPOTLE|KFC|WENDY'?S?)", "Dining Out", 0.9),
    Rule("dining_generic", r"\b(CAFE|COFFEE|GRILL|KITCHEN|BAKERY|SUBWAY|PUB)\b", "Dining Out", A, risk="ambiguous"),

    # Transport
    Rule("fuel", r"\b(SHELL|CHEVRON|EXXON|MOBIL|BP|TEXACO|ESSO|FUEL|GAS\s*STATION|PETRO(L|LEUM)?)\b", "Transport",
         0.9),
    Rule("rideshare_transit", r"\b(UBER|LYFT|TAXI|CAB|METRO\s*TRANSIT|TRANSIT|PARKING|TOLL|AMTRAK|RAIL)\b",
         "Transport", 0.9),

    # Health
    Rule("health", r"\b(PHARMACY|CVS|WALGREENS|RITE\s*AID|CLINIC|DENTAL|DENTIST|HOSPITAL|MEDICAL|DOCTOR|"
                   r"OPTOMETR(Y|IST))", "Health", 0.9),

    # Entertainment. A console can be a purchase (Shopping) or a game pass (Subscriptions).
    Rule("entertainment", r"\b(CINEMAS?|MOVIES?|THEATER|THEATRE|AMC|TICKETMASTER|STEAM\s*GAMES|CONCERT|BOWLING|"
                          r"MUSEUM)\b", "Entertainment", 0.9),
    Rule("entertainment_generic", r"\b(PLAYSTATION|XBOX)\b", "Entertainment", A, risk="ambiguous"),

    # Shopping
    Rule("shopping", r"\b(AMZN|AMAZON|WALMART|TARGET|EBAY|ETSY|BEST\s*BUY|IKEA|ELECTRONICS|APPAREL|CLOTHING|"
                     r"HOME\s*DEPOT|MACY'?S?)", "Shopping", 0.85),

    # Bank fees (US-style statements)
    Rule("bank_fee", r"\b(OVERDRAFT|NSF|(SERVICE|MONTHLY|MAINTENANCE|ATM|LATE)\s*FEE)\b", "Bank Fees", 0.9),
)


@dataclass(frozen=True)
class RuleMatch:
    rule: str
    category: str
    confidence: float

    @property
    def decides(self) -> bool:
        return self.confidence >= RULE_THRESHOLD


def find_rule(description: str, amount: float | None = None,
              rules: tuple[Rule, ...] = RULES) -> RuleMatch | None:
    """The first matching deciding rule; else the first matching ambiguous rule (a hint only); else None.
    An ambiguous match never hides a deciding rule further down the list."""
    if not description:
        return None
    direction = None if amount is None else ("credit" if amount >= 0 else "debit")
    hint = None
    for rule in rules:
        if rule.direction != "any" and direction is not None and rule.direction != direction:
            continue
        if rule.regex.search(description):
            if rule.decides:
                return RuleMatch(rule.name, rule.category, rule.confidence)
            hint = hint or RuleMatch(rule.name, rule.category, rule.confidence)
    return hint


def match_rule(description: str, amount: float | None = None,
               rules: tuple[Rule, ...] = RULES) -> tuple[str, float] | None:
    """(category, confidence) of the first deciding rule, or None. Ambiguous matches are not returned:
    use find_rule() to get them as a hint."""
    hit = find_rule(description, amount, rules)
    return (hit.category, hit.confidence) if hit and hit.decides else None
