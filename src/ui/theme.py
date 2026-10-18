"""Brand colors, the per-category palette, light/dark surfaces and the shared Plotly template."""
from __future__ import annotations

import copy

import plotly.graph_objects as go

APP_NAME = "Myfinance"
TAGLINE = "Understand. Manage. Grow."
FONT = "Inter, -apple-system, Segoe UI, sans-serif"
CHART_HEIGHT = 320
SPARK_HEIGHT = 56

NAVY, TEAL = "#0B1F3A", "#14B8A6"
GOOD, BAD, WARNING, MUTED = "#16A34A", "#DC2626", "#F59E0B", "#64748B"
CLOUD_BLUE = "#2563EB"

# One fixed color per category, identical in every chart.
CATEGORY_COLORS = {
    "Rent": "#1E3A8A", "Groceries": "#14B8A6", "Dining Out": "#F97316", "Transport": "#6366F1",
    "Utilities": "#0EA5E9", "Subscriptions": "#A855F7", "Entertainment": "#EC4899", "Shopping": "#EAB308",
    "Health": "#10B981", "Transfers": "#94A3B8", "Income": "#22C55E", "Other": "#CBD5E1",
    "Interest": "#15803D", "Transfers In": "#0891B2", "Charity/Donations": "#84CC16", "Cash Withdrawal": "#8B5E34",
    "Unaccounted cash": "#C4A484",
    "Bank Fees": "#F43F5E", "Tax": "#475569", "Uncategorized": "#CBD5E1",
}

PALETTES = {
    "light": {"page": "#F8FAFC", "card": "#FFFFFF", "text": NAVY, "muted": MUTED, "border": "#E2E8F0",
              "grid": "#EEF2F6", "shadow": "0 1px 3px rgba(15,23,42,.06), 0 1px 2px rgba(15,23,42,.04)",
              "good_bg": "#DCFCE7", "bad_bg": "#FEE2E2", "warn_bg": "#FEF3C7", "info_bg": "#DBEAFE",
              "neutral_bg": "#F1F5F9", "highlight": "#F0FDFA"},
    "dark": {"page": "#0F172A", "card": "#1E293B", "text": "#E2E8F0", "muted": "#94A3B8", "border": "#334155",
             "grid": "#2B3A50", "shadow": "0 1px 3px rgba(0,0,0,.4)",
             "good_bg": "rgba(22,163,74,.18)", "bad_bg": "rgba(220,38,38,.18)", "warn_bg": "rgba(245,158,11,.18)",
             "info_bg": "rgba(37,99,235,.22)", "neutral_bg": "#273449", "highlight": "rgba(20,184,166,.12)"},
}

# Privacy label per provider. groq is listed for a future cloud provider (not implemented in src/llm yet).
PROVIDER_INFO = {
    "fake": ("Local", "Local - nothing leaves this machine", "Offline demo model (keyword planner, no LLM)"),
    "ollama": ("Local", "Local - nothing leaves this machine", "Ollama model running on this computer"),
    "groq": ("Cloud", "Cloud - only redacted text is sent", "Groq cloud API"),
}
AVAILABLE_PROVIDERS = ("fake", "ollama")
REAL_MODELS = frozenset({"ollama", "groq"})


def palette(mode: str | None) -> dict[str, str]:
    return PALETTES["dark" if mode == "dark" else "light"]


def category_color(category: str) -> str:
    return CATEGORY_COLORS.get(category, CATEGORY_COLORS["Other"])


def plotly_template(mode: str | None) -> go.layout.Template:
    """Transparent background, light horizontal gridlines only, Inter, bottom legends, Rs hovers."""
    p = palette(mode)
    # automargin: the 8px margins grow to fit tick labels, so "Rs 1,000" and merchant names are never cut off.
    axis = dict(showgrid=False, zeroline=False, showline=False, ticks="", color=p["muted"], tickangle=0, automargin=True,
                tickfont=dict(size=12, color=p["muted"]), title=dict(font=dict(size=12, color=p["muted"])))
    t = go.layout.Template()
    t.layout = go.Layout(
        font=dict(family=FONT, size=13, color=p["text"]),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=12, b=8),
        xaxis=dict(axis), yaxis=dict(axis, showgrid=True, gridcolor=p["grid"], gridwidth=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,  # top: never meets tick labels
                    font=dict(size=12, color=p["muted"]), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=p["card"], bordercolor=p["border"], font=dict(family=FONT, size=13, color=p["text"])),
        separators=".,", barcornerradius=4, bargap=0.28, bargroupgap=0.08,
        hovermode="closest", colorway=list(CATEGORY_COLORS.values()),
    )
    return t


def tint(hex_color: str, amount: float, toward: str = "#FFFFFF") -> str:
    """Blend `hex_color` toward `toward` (amount 0 = unchanged, 1 = `toward`): a pale version of a color."""
    a = [int(hex_color.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]
    b = [int(toward.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]
    return "#" + "".join(f"{round(x + (y - x) * amount):02X}" for x, y in zip(a, b))


def apply(fig: go.Figure, mode: str | None, height: int = CHART_HEIGHT, **layout) -> go.Figure:
    fig.update_layout(template=copy.deepcopy(plotly_template(mode)), height=height, **layout)
    fig.update_xaxes(tickangle=0)  # never rotated or slanted labels, on every axis
    fig.update_yaxes(tickangle=0)
    return fig
