"""The one place money, percentages and months are formatted for display. Negative money is always
"-Rs 1,234" (sign before the currency); `signed=True` also puts "+" on positive values."""
from __future__ import annotations

from datetime import date

import pandas as pd

from src.analytics.common import CURRENCY  # the one currency setting
MINUS = "-"
# Plotly hover formats, so tooltips match money() exactly.
HOVER_MONEY_X = CURRENCY + " %{x:,.0f}"
HOVER_MONEY_Y = CURRENCY + " %{y:,.0f}"
NA = "—"


def money(value: float | None, cents: bool = False, signed: bool = False) -> str:
    """'Rs 1,234' (or 'Rs 1,234.56' with cents). None -> an em dash."""
    if value is None:
        return NA
    body = f"{CURRENCY} {abs(value):,.2f}" if cents else f"{CURRENCY} {abs(value):,.0f}"
    if round(value, 2 if cents else 0) == 0:
        return body
    if value < 0:
        return MINUS + body
    return ("+" + body) if signed else body


def pct(value: float | None, decimals: int = 1, signed: bool = False) -> str:
    """'12.3%' for a 0-100 percentage."""
    if value is None:
        return NA
    text = f"{abs(value):,.{decimals}f}%"
    if value < 0:
        return MINUS + text
    return ("+" + text) if signed and value > 0 else text


def points(value: float | None, decimals: int = 1) -> str:
    """Change of a percentage, in percentage points: '+3.2 pts'."""
    if value is None:
        return NA
    sign = "+" if value > 0 else (MINUS if value < 0 else "")
    return f"{sign}{abs(value):.{decimals}f} pts"


def month_label(month: str, short: bool = False) -> str:
    """'2026-08' -> 'August 2026' (or 'Aug 2026')."""
    return pd.Period(month, freq="M").strftime("%b %Y" if short else "%B %Y")


def month_tick(month: str, with_year: bool = False) -> str:
    """Short axis label: '2026-08' -> 'Aug' (or "Aug '26" where the year needs saying)."""
    return pd.Period(month, freq="M").strftime("%b '%y" if with_year else "%b")


def month_name(month: str) -> str:
    """'2026-07' -> 'July'."""
    return pd.Period(month, freq="M").strftime("%B")


def day_label(d: date) -> str:
    return f"{d:%d %b}"
