"""Shared building blocks: the one CSS helper, cards, section titles, KPI cards, pills and badges.
Pure presentation: values arrive already computed (src/analytics) and are formatted with src/ui/format.py."""
from __future__ import annotations

from contextlib import contextmanager
from html import escape

import streamlit as st

from src.analytics.schema import Change
from src.ui import format as fmt
from src.ui.theme import BAD, CLOUD_BLUE, FONT, GOOD, MUTED, NAVY, TEAL, WARNING, palette

NO_EARLIER = "No earlier month to compare"

# Metrics where a rise is bad news; for every other KPI a rise is good news.
UP_IS_BAD = frozenset({"total_spent"})

INSIGHT_KIND_LABEL = {"category_increase": "Spending up", "category_decrease": "Spending down",
                      "unused_subscription": "Subscription", "anomaly": "Unusual charge",
                      "cash_share": "Cash"}


def theme_mode() -> str:
    try:
        return st.context.theme.type or "light"
    except Exception:  # no browser context (e.g. AppTest)
        return "light"


def inject_css(mode: str) -> None:
    p = palette(mode)
    st.markdown(f"""<style>
html, body, [class*="css"], .stMarkdown {{ font-family: {FONT}; }}
.stApp {{ background:{p['page']}; }}
#MainMenu, footer, .stDeployButton, [data-testid="stAppDeployButton"], [data-testid="stToolbarActions"],
[data-testid="stDecoration"], [data-testid="stMainMenu"] {{ display:none !important; }}
/* stToolbar itself must stay visible: it holds the "open sidebar" button (stExpandSidebarButton). */
header[data-testid="stHeader"] {{ background:transparent; }}
.block-container {{ padding-top:1.6rem; padding-bottom:3rem; max-width:1400px; }}
[data-testid="stPlotlyChart"], [data-testid="stPlotlyChart"] .main-svg {{ background:transparent !important; }}
[class*="st-key-card-"] {{ background:{p['card']}; border:1px solid {p['border']}; border-radius:12px;
  box-shadow:{p['shadow']}; padding:18px 20px; }}
[class*="st-key-card-"] [data-testid="stVerticalBlock"] {{ gap:.5rem; }}
.fa-section {{ margin:24px 0 2px; }}
.fa-section h3 {{ font-size:1.12rem; font-weight:600; color:{p['text']}; margin:0; padding:0; }}
.fa-section p {{ font-size:.86rem; color:{p['muted']}; margin:2px 0 0; }}
.fa-header .name {{ font-size:1.55rem; font-weight:700; color:{p['text']}; letter-spacing:-.01em; line-height:1.2; }}
.fa-header .tag {{ font-size:.82rem; letter-spacing:.01em; color:{p['muted']}; white-space:nowrap; overflow:hidden;
  text-overflow:ellipsis; }}
.fa-kpi .label {{ font-size:.78rem; font-weight:500; color:{p['muted']}; text-transform:uppercase; letter-spacing:.05em; }}
.fa-kpi .value {{ font-size:1.75rem; font-weight:700; color:{p['text']}; margin:4px 0 6px; line-height:1.15;
  font-variant-numeric:tabular-nums; }}
.fa-kpi .row {{ display:flex; align-items:center; gap:8px; flex-wrap:wrap; min-height:24px; }}
.fa-kpi .caption {{ font-size:.8rem; color:{p['muted']}; }}
.fa-pill {{ display:inline-flex; align-items:center; gap:4px; padding:2px 9px; border-radius:999px;
  font-size:.78rem; font-weight:600; line-height:1.5; white-space:nowrap; }}
.fa-pill.good, .fa-pill.local {{ background:{p['good_bg']}; color:{GOOD}; }}
.fa-pill.bad {{ background:{p['bad_bg']}; color:{BAD}; }}
.fa-pill.warn {{ background:{p['warn_bg']}; color:{WARNING}; }}
.fa-pill.info, .fa-pill.cloud {{ background:{p['info_bg']}; color:{CLOUD_BLUE}; }}
.fa-pill.flat, .fa-pill.neutral {{ background:{p['neutral_bg']}; color:{p['muted']}; }}
.fa-right {{ display:flex; justify-content:flex-end; align-items:center; height:100%; padding-top:30px; }}
.fa-insight {{ border-left:4px solid {MUTED}; padding:2px 0 2px 12px; font-size:.92rem; color:{p['text']}; }}
.fa-insight.bad {{ border-color:{BAD}; }}
.fa-insight.good {{ border-color:{GOOD}; }}
.fa-insight.warn {{ border-color:{WARNING}; }}
.fa-insight .kind {{ font-size:.7rem; font-weight:600; text-transform:uppercase; letter-spacing:.06em;
  color:{p['muted']}; margin-bottom:2px; }}
.fa-table {{ width:100%; border-collapse:collapse; font-size:.9rem; color:{p['text']}; }}
.fa-table th {{ text-align:right; font-weight:500; color:{p['muted']}; font-size:.75rem; text-transform:uppercase;
  letter-spacing:.05em; padding:8px 12px; border-bottom:1px solid {p['border']}; }}
.fa-table td {{ text-align:right; padding:9px 12px; border-bottom:1px solid {p['border']};
  font-variant-numeric:tabular-nums; }}
.fa-table th:first-child, .fa-table td:first-child {{ text-align:left; }}
.fa-table tr:last-child td {{ border-bottom:none; }}
.fa-table tr.selected td {{ background:{p['highlight']}; font-weight:600; }}
.fa-table td.good {{ color:{GOOD}; font-weight:600; }}
.fa-table td.bad {{ color:{BAD}; font-weight:600; }}
.fa-table td.flat {{ color:{p['muted']}; }}
.fa-empty {{ text-align:center; padding:56px 16px; color:{p['muted']}; }}
.fa-empty .big {{ font-size:1.25rem; font-weight:600; color:{p['text']}; margin-bottom:6px; }}
.fa-muted {{ color:{p['muted']}; font-size:.86rem; }}
.fa-opp-title {{ font-weight:600; font-size:1rem; color:{p['text']}; }}
.fa-opp-save {{ font-size:1.35rem; font-weight:700; color:{TEAL}; font-variant-numeric:tabular-nums; }}
.fa-calc {{ font-family:ui-monospace, Consolas, monospace; font-size:.8rem; color:{p['muted']};
  background:{p['neutral_bg']}; border-radius:8px; padding:8px 10px; }}
.fa-stat .label {{ font-size:.75rem; color:{p['muted']}; text-transform:uppercase; letter-spacing:.05em; }}
.fa-stat .value {{ font-size:1.3rem; font-weight:700; color:{p['text']}; font-variant-numeric:tabular-nums; }}
.fa-stat .sub {{ font-size:.8rem; color:{p['muted']}; }}
.fa-footer {{ text-align:center; color:{p['muted']}; font-size:.8rem; margin-top:32px; }}
/* transactions: filter pills, HTML ledger, fix box */
[class*="st-key-txn-filter-"] button {{ border-radius:999px !important; border:1px solid {p['border']} !important;
  background:{p['card']} !important; color:{p['text']} !important; }}
[class*="st-key-txn-filter-"] button[aria-checked="true"], [class*="st-key-txn-filter-"] button[aria-pressed="true"],
[class*="st-key-txn-filter-"] button[data-selected="true"], [class*="st-key-txn-filter-"] button[kind="pillsActive"],
[class*="st-key-txn-filter-"] button[data-testid="stBaseButton-pillsActive"] {{ background:{TEAL} !important;
  border-color:{TEAL} !important; color:#FFFFFF !important; }}
[class*="st-key-txn-filter-"] button[aria-checked="true"] p, [class*="st-key-txn-filter-"] button[data-selected="true"] p,
[class*="st-key-txn-filter-"] button[data-testid="stBaseButton-pillsActive"] p {{ color:#FFFFFF !important; font-weight:600; }}
.fa-ledger {{ width:100%; border-collapse:collapse; font-size:.88rem; color:{p['text']}; }}
.fa-ledger, .fa-ledger tr, .fa-ledger th, .fa-ledger td {{ background:transparent; border:none; }}
.fa-ledger th {{ text-align:left; font-weight:500; font-size:.76rem; color:{p['muted']}; padding:8px 10px;
  border-bottom:1px solid {p['border']} !important; }}
.fa-ledger td {{ padding:10px; border-bottom:1px solid {p['border']} !important; vertical-align:middle; }}
.fa-ledger tbody tr:hover td {{ background:{p['highlight']}; }}
.fa-ledger tbody tr:last-child td {{ border-bottom:none !important; }}
.fa-ledger td.date {{ color:{p['muted']}; white-space:nowrap; }}
.fa-ledger td.merchant {{ font-weight:600; color:{p['text']}; }}
.fa-ledger .amt {{ text-align:right; white-space:nowrap; font-variant-numeric:tabular-nums; }}
.fa-ledger td.amt {{ font-weight:700; }}
.fa-ledger td.amt.in {{ color:{GOOD}; }}
.fa-cat {{ display:inline-flex; align-items:center; padding:2px 10px; border-radius:999px; font-size:.8rem;
  border:1px solid {p['border']}; background:{p['card']}; color:{p['text']}; white-space:nowrap; }}
.fa-src {{ display:inline-block; padding:1px 8px; border-radius:6px; font-size:.72rem; font-weight:600;
  background:{p['neutral_bg']}; color:{p['muted']}; }}
.fa-src.warn {{ background:{p['warn_bg']}; color:{WARNING}; }}
.fa-src.cash {{ background:{p['info_bg']}; color:{p['text']}; }}
/* cash wallet card */
.fa-cash-stats {{ display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:16px; margin:12px 0 16px; }}
.fa-cash-stats .l {{ font-size:.78rem; color:{p['muted']}; }}
.fa-cash-stats .v {{ font-size:1.35rem; font-weight:700; color:{p['text']}; font-variant-numeric:tabular-nums; }}
.fa-cash-stats .v.warn {{ color:{WARNING}; }}
.fa-help {{ display:inline-flex; align-items:center; justify-content:center; width:16px; height:16px; margin-left:4px;
  border-radius:50%; border:1px solid {p['border']}; color:{p['muted']}; font-size:.68rem; font-weight:600;
  cursor:help; vertical-align:middle; }}
.fa-cash-bar {{ margin:0 0 16px; }}
.fa-cash-bar .track {{ height:12px; border-radius:999px; background:{p['neutral_bg']}; overflow:hidden; }}
.fa-cash-bar .fill {{ height:100%; border-radius:999px; background:{TEAL}; }}
.fa-cash-bar .l {{ font-size:.8rem; color:{p['muted']}; margin-top:6px; }}
.fa-cash-confirm {{ font-weight:600; color:{p['text']}; }}
.fa-cash-what {{ color:{p['text']}; }}
.fa-cash-amt {{ text-align:right; font-weight:600; color:{p['text']}; font-variant-numeric:tabular-nums; }}
.st-key-card-cash [data-testid="stExpander"] details {{ border:1px solid {p['border']} !important; border-radius:10px; }}
.st-key-cash-recent {{ border-top:1px solid {p['border']}; padding-top:12px; margin-top:4px; }}
.st-key-cash-edit-box {{ background:{p['neutral_bg']}; border-radius:10px; padding:12px; }}
@media (max-width:640px) {{ .fa-cash-stats {{ gap:8px; }} .fa-cash-stats .v {{ font-size:1.05rem; }} }}
.st-key-card-ask [data-testid="stExpander"] details {{ border:none !important; background:transparent !important; }}
.st-key-card-ask [data-testid="stExpander"] summary {{ padding:2px 4px !important; min-height:0 !important; }}
.st-key-card-ask [data-testid="stExpander"] summary p {{ font-size:.8rem !important; color:{GOOD}; font-weight:600; }}
.st-key-fix-box {{ border-top:1px solid {p['border']}; padding-top:12px; margin-top:4px; }}
/* ---- reference design: top bar, underline tabs, card headers ---- */
.st-key-topbar {{ background:{NAVY}; border-radius:12px; padding:12px 20px; margin-bottom:4px; }}
.st-key-topbar [data-testid="stHorizontalBlock"] {{ align-items:center; }}
.fa-brand {{ display:flex; align-items:center; gap:12px; min-width:0; }}
.fa-brand .logo {{ width:40px; height:40px; border-radius:10px; background:{TEAL}; display:flex; align-items:center;
  justify-content:center; flex:none; }}
.fa-brand .name {{ font-size:1.15rem; font-weight:700; color:#FFFFFF; line-height:1.2; }}
.fa-brand .tag {{ font-size:.8rem; color:#CBD5E1; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.fa-privacy {{ display:inline-flex; align-items:center; gap:8px; padding:5px 12px; border-radius:999px;
  background:rgba(22,163,74,.16); color:#4ADE80; font-size:.8rem; font-weight:600; white-space:nowrap; }}
.fa-privacy.cloud {{ background:rgba(37,99,235,.22); color:#93C5FD; }}
.fa-privacy .dot {{ width:8px; height:8px; border-radius:50%; background:currentColor; }}
.st-key-topbar .react-aria-ComboBox [role="group"] {{ background:rgba(255,255,255,.06) !important;
  border:1px solid rgba(255,255,255,.22) !important; border-radius:8px; }}
.st-key-topbar .react-aria-ComboBox input {{ color:#FFFFFF !important; -webkit-text-fill-color:#FFFFFF;
  font-weight:600; background:transparent !important; }}
.st-key-topbar .react-aria-ComboBox button {{ color:#FFFFFF !important; background:transparent !important; }}
.st-key-topbar [data-testid="stWidgetLabel"] p {{ color:#CBD5E1; }}
.st-key-nav {{ border-bottom:1px solid {p['border']}; margin-bottom:6px; width:100% !important; }}
.st-key-nav [data-testid="stButtonGroup"] {{ gap:8px; }}
.st-key-nav button {{ background:transparent !important; border:none !important; border-bottom:2px solid transparent !important;
  border-radius:0 !important; color:{p['muted']} !important; padding:8px 14px !important; box-shadow:none !important; }}
.st-key-nav button[aria-checked="true"] {{ border-bottom:2px solid {TEAL} !important; color:{p['text']} !important; }}
.st-key-nav button[aria-checked="true"] p {{ font-weight:600; }}
.fa-card-head {{ margin:0 0 6px; }}
.fa-card-head .title {{ font-size:1rem; font-weight:600; color:{p['text']}; }}
.fa-card-head .sub {{ font-size:.8rem; color:{p['muted']}; margin-top:1px; }}
.fa-kpi .label.plain {{ text-transform:none; letter-spacing:0; font-size:.82rem; }}
/* donut legend */
.fa-legend {{ width:100%; border-collapse:collapse; font-size:.84rem; color:{p['text']}; }}
.fa-legend, .fa-legend tr, .fa-legend td {{ background:transparent !important; border:none !important; }}
.fa-legend td {{ padding:3px 4px; white-space:nowrap; }}
.fa-legend td.amt {{ text-align:right; font-weight:600; font-variant-numeric:tabular-nums; }}
.fa-legend td.pct {{ text-align:right; color:{p['muted']}; font-variant-numeric:tabular-nums; }}
.fa-dot {{ display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:8px; vertical-align:middle; }}
/* insight cards with a Fix this button (one keyed container per insight) */
[class*="st-key-insight-"] {{ background:{p['neutral_bg']}; border-radius:10px; border-left:4px solid {MUTED};
  padding:12px 14px; }}
[class*="st-key-insight-bad-"] {{ border-left-color:{BAD}; }}
[class*="st-key-insight-good-"] {{ border-left-color:{GOOD}; }}
[class*="st-key-insight-warn-"] {{ border-left-color:{WARNING}; }}
.fa-ins-kind {{ font-size:.7rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase; }}
.fa-ins-kind.bad {{ color:{BAD}; }} .fa-ins-kind.good {{ color:{GOOD}; }} .fa-ins-kind.warn {{ color:{WARNING}; }}
.fa-ins-text {{ font-size:.9rem; color:{p['text']}; margin:2px 0 4px; }}
[class*="st-key-insight-"] [data-testid="stVerticalBlock"] {{ gap:.65rem; }}
[class*="st-key-insight-"] [data-testid="stMarkdownContainer"] {{ margin-bottom:0; }}
[class*="st-key-insight-"] [class*="st-key-fix-"] button {{ border:1px solid {TEAL} !important; color:{TEAL} !important; background:{p['card']} !important;
  border-radius:8px !important; padding:2px 12px !important; min-height:0 !important; font-size:.8rem !important; }}
/* top merchants as labelled progress bars */
.fa-merchant {{ margin:4px 0 14px; }}
.fa-merchant .row {{ display:flex; justify-content:space-between; gap:12px; font-size:.86rem; color:{p['text']}; }}
.fa-merchant .row .n {{ font-weight:600; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
.fa-merchant .row .v {{ font-variant-numeric:tabular-nums; white-space:nowrap; }}
.fa-merchant .track {{ height:8px; border-radius:999px; background:{p['neutral_bg']}; margin-top:6px; overflow:hidden; }}
.fa-merchant .fill {{ height:100%; border-radius:999px; }}
/* badges and chips */
.fa-badge {{ display:inline-block; padding:1px 8px; border-radius:6px; font-size:.72rem; font-weight:600;
  background:{p['neutral_bg']}; color:{p['muted']}; border:1px solid {p['border']}; }}
.fa-chip {{ display:inline-flex; align-items:center; padding:2px 10px; border-radius:999px; font-size:.78rem;
  border:1px solid {p['border']}; color:{p['muted']}; background:{p['card']}; }}
/* chat */
.fa-msg {{ display:flex; margin:10px 0; }}
.fa-msg.user {{ justify-content:flex-end; }}
.fa-bubble {{ max-width:86%; padding:12px 16px; border-radius:12px; font-size:.92rem; line-height:1.5; }}
.fa-msg.user .fa-bubble {{ background:{NAVY if mode != 'dark' else '#334155'}; color:#FFFFFF; }}
.fa-msg.bot .fa-bubble {{ background:{p['neutral_bg']}; color:{p['text']}; border:1px solid {p['border']}; }}
.fa-bubble .badges {{ margin-top:8px; display:flex; flex-wrap:wrap; gap:6px; }}
[class*="st-key-suggest-"] button {{ border-radius:999px !important; border:1px solid {p['border']} !important;
  background:{p['card']} !important; color:{p['text']} !important; font-size:.84rem !important; }}
[data-testid="stFormSubmitButton"] button {{ background:{TEAL} !important; color:#FFFFFF !important;
  border:none !important; font-weight:600; }}
[class*="st-key-suggest-"] button p {{ white-space:normal !important; text-align:left; }}
.st-key-ask-input div:has(> input) {{ border:1px solid {p['border']} !important; border-radius:8px;
  background:{p['neutral_bg']} !important; }}
/* How I got this panel */
.fa-step {{ border:1px solid {p['border']}; border-radius:10px; padding:12px 14px; margin-bottom:10px; }}
.fa-step .h {{ display:flex; align-items:center; gap:8px; font-weight:600; font-size:.9rem; color:{p['text']}; margin-bottom:8px; }}
.fa-step .n {{ width:22px; height:22px; border-radius:50%; background:{TEAL}; color:#FFFFFF; font-size:.75rem;
  display:inline-flex; align-items:center; justify-content:center; }}
.fa-code {{ background:{p['neutral_bg']}; color:{p['text']}; border-radius:8px; padding:10px 12px; font-size:.8rem;
  margin:0 0 6px; white-space:pre-wrap; font-family:ui-monospace, Consolas, monospace; line-height:1.5; }}
.fa-fields {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; }}
.fa-field {{ background:{p['neutral_bg']}; border-radius:8px; padding:8px 10px; }}
.fa-field .k {{ font-size:.72rem; color:{p['muted']}; overflow:hidden; text-overflow:ellipsis; }}
.fa-field .v {{ font-size:1rem; font-weight:700; color:{p['text']}; font-variant-numeric:tabular-nums; }}
.fa-checks {{ color:{GOOD}; font-weight:600; font-size:.86rem; }}
.fa-checks .no {{ color:{BAD}; }}
/* plan */
.fa-goal-title {{ font-size:1.45rem; font-weight:700; color:{p['text']}; line-height:1.25; margin:2px 0 8px; }}
.fa-save-pill {{ display:inline-block; padding:2px 10px; border-radius:999px; background:{p['good_bg']}; color:{GOOD};
  font-weight:700; font-size:.82rem; white-space:nowrap; }}
.fa-opp-text {{ font-size:.88rem; color:{p['text']}; }}
[data-result] .value {{ font-size:1.55rem; white-space:nowrap; }}
[class*="st-key-card-opp-"] [data-testid="stCheckbox"] p {{ white-space:normal !important; overflow:visible !important;
  text-overflow:clip !important; }}
.st-key-goal-edit button, .st-key-goal-clear button {{ padding:2px 10px !important; min-height:0 !important;
  font-size:.8rem !important; border-radius:999px !important; }}
.st-key-goal-edit button p, .st-key-goal-clear button p {{ white-space:nowrap; }}
.fa-note {{ display:flex; gap:8px; align-items:flex-start; font-size:.78rem; color:{p['muted']}; }}
.fa-note .dot {{ width:7px; height:7px; border-radius:50%; background:{WARNING}; margin-top:5px; flex:none; }}
.fa-honest {{ border-left:4px solid {WARNING}; background:{p['warn_bg']}; color:{p['text']}; border-radius:8px;
  padding:10px 12px; font-size:.88rem; }}
.fa-empty-card {{ text-align:center; padding:28px 12px; color:{p['muted']}; font-size:.9rem; }}
.fa-empty-card .big {{ font-weight:600; color:{p['text']}; margin-bottom:4px; }}
</style>""", unsafe_allow_html=True)


@contextmanager
def card(key: str):
    """A styled card: rounded 12px, 1px soft border, subtle shadow (CSS matches the container key)."""
    with st.container(key=f"card-{key}"):
        yield


def section(title: str, subtitle: str = "") -> None:
    sub = f"<p>{escape(subtitle)}</p>" if subtitle else ""
    st.markdown(f'<div class="fa-section"><h3>{escape(title)}</h3>{sub}</div>', unsafe_allow_html=True)


def pill(text: str, tone: str = "neutral") -> str:
    return f'<span class="fa-pill {tone}">{escape(text)}</span>'


def privacy_pill(kind: str) -> str:
    return pill("● Local", "local") if kind == "Local" else pill("● Cloud - redacted", "cloud")


def delta_tone(metric: str, abs_change: float | None) -> str:
    """good | bad | flat. Spending going up is bad; income, net saved and savings rate going up is good."""
    if abs_change is None or round(abs_change, 2) == 0:
        return "flat"
    up = abs_change > 0
    return ("bad" if up else "good") if metric in UP_IS_BAD else ("good" if up else "bad")


def delta_text(ch: Change, is_rate: bool = False) -> str:
    """'▲ 12.3%' (relative change), points for a rate, or the Rs change when the % is undefined
    (previous value 0)."""
    arrow = "▲" if ch.abs_change > 0 else ("▼" if ch.abs_change < 0 else "■")
    if is_rate:
        return f"{arrow} {fmt.points(abs(ch.abs_change)).lstrip('+')}"
    if ch.pct_change is not None:
        return f"{arrow} {fmt.pct(abs(ch.pct_change))}"
    return f"{arrow} {fmt.money(abs(ch.abs_change))}"


def kpi_html(metric: str, label: str, value: str, ch: Change, previous_month: str | None,
             is_rate: bool = False) -> str:
    """KPI card body. data-kpi / data-tone attributes make the delta logic testable."""
    if previous_month is None or ch.abs_change is None:
        tone = "none"
        row = f'<span class="caption">{NO_EARLIER}</span>'
    else:
        tone = delta_tone(metric, ch.abs_change)
        row = (pill(delta_text(ch, is_rate), tone)
               + f'<span class="caption">vs {escape(fmt.month_name(previous_month))}</span>')
    return (f'<div class="fa-kpi" data-kpi="{metric}" data-tone="{tone}">'
            f'<div class="label plain">{escape(label)}</div><div class="value">{escape(value)}</div>'
            f'<div class="row">{row}</div></div>')


def stat_html(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="sub">{escape(sub)}</div>' if sub else ""
    return (f'<div class="fa-stat"><div class="label">{escape(label)}</div>'
            f'<div class="value">{escape(value)}</div>{sub_html}</div>')


def insight_tone(kind: str) -> str:
    return {"category_increase": "bad", "category_decrease": "good"}.get(kind, "warn")


def table_html(headers: list[str], rows: list[list[tuple[str, str]]], selected: int | None = None) -> str:
    """rows = list of [(text, css_class), ...]; the `selected` row is highlighted."""
    head = "".join(f"<th>{escape(h)}</th>" for h in headers)
    body = []
    for i, row in enumerate(rows):
        cells = "".join(f'<td class="{c}">{escape(t)}</td>' for t, c in row)
        body.append(f'<tr class="{"selected" if i == selected else ""}">{cells}</tr>')
    return f'<table class="fa-table"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>'


def empty_state(title: str, body: str) -> None:
    st.markdown(f'<div class="fa-empty"><div class="big">{escape(title)}</div>{escape(body)}</div>',
                unsafe_allow_html=True)


def muted(text: str) -> None:
    st.markdown(f'<div class="fa-muted">{escape(text)}</div>', unsafe_allow_html=True)


def footer() -> None:
    sample = "Sample synthetic data. " if "sample" in str(st.session_state.get("db_label", "")).lower() else ""
    st.markdown(f'<div class="fa-footer">{sample}Not financial advice. Every figure is computed from your statement '
                'by deterministic code; suggestions are estimates based on your own history.</div>',
                unsafe_allow_html=True)


# ---- reference-design building blocks -----------------------------------------------------------

LOGO_SVG = ('<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2" '
            'stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>'
            '<path d="M9 12l2 2 4-4"/></svg>')


def brand_html(name: str, tagline: str) -> str:
    return (f'<div class="fa-brand"><div class="logo">{LOGO_SVG}</div><div style="min-width:0">'
            f'<div class="name">{escape(name)}</div><div class="tag">{escape(tagline)}</div></div></div>')


def privacy_badge(kind: str, label: str) -> str:
    """Top-bar privacy pill: green for local models, blue for cloud ones (label from theme.PROVIDER_INFO)."""
    tone = "" if kind == "Local" else " cloud"
    return f'<span class="fa-privacy{tone}"><span class="dot"></span>{escape(label)}</span>'


def card_head(title: str, subtitle: str = "") -> None:
    """Title + muted subtitle at the top of a card (reference design: headers live inside the cards)."""
    sub = f'<div class="sub">{escape(subtitle)}</div>' if subtitle else ""
    st.markdown(f'<div class="fa-card-head"><div class="title">{escape(title)}</div>{sub}</div>',
                unsafe_allow_html=True)


def empty_card(title: str, body: str = "") -> None:
    st.markdown(f'<div class="fa-empty-card"><div class="big">{escape(title)}</div>{escape(body)}</div>',
                unsafe_allow_html=True)


def legend_html(rows: list[tuple[str, str, str, str]]) -> str:
    """Donut legend: rows of (color, name, amount text, percent text)."""
    body = "".join(f'<tr><td><span class="fa-dot" style="background:{c}"></span>{escape(n)}</td>'
                   f'<td class="amt">{escape(a)}</td><td class="pct">{escape(pc)}</td></tr>' for c, n, a, pc in rows)
    return f'<table class="fa-legend">{body}</table>'


def merchant_bars_html(rows: list[tuple[str, str, str, float]]) -> str:
    """Top merchants: rows of (name, amount text, color, share of the largest 0-1)."""
    return "".join(f'<div class="fa-merchant"><div class="row"><span class="n">{escape(n)}</span>'
                   f'<span class="v">{escape(v)}</span></div><div class="track"><div class="fill" '
                   f'style="width:{max(2.0, share * 100):.1f}%;background:{c}"></div></div></div>'
                   for n, v, c, share in rows)


def badge(text: str) -> str:
    return f'<span class="fa-badge">{escape(text)}</span>'


def chip(text: str, color: str | None = None) -> str:
    dot = f'<span class="fa-dot" style="background:{color}"></span>' if color else ""
    return f'<span class="fa-chip">{dot}{escape(text)}</span>'
