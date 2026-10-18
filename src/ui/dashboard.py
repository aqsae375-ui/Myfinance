"""Dashboard tab: the home screen. Reads Stage 3 analytics only (through the cached src/ui/data.py);
it never imports or calls an LLM provider. Layout follows the reference design: KPI cards, donut +
month comparison, trend + "What changed", top merchants + daily spending, month history, transactions.
Every label that depends on the data (months, amounts, merchants, sentences) comes from src/analytics."""
from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st

from src.analytics.cash import CASH_SOURCE, OUTSIDE_FLAG, UNACCOUNTED
from src.analytics.dashboard import CASH
from src.cash_wallet import ENTRY_CATEGORIES, CashEntryError, parse_quick_line
from src.categories import CATEGORIES
from src.ui import charts, data
from src.ui import format as fmt
from src.ui.components import (NO_EARLIER, card, card_head, delta_tone, empty_card, insight_tone, kpi_html,
                               legend_html, merchant_bars_html, muted, table_html)
from src.ui.theme import WARNING, category_color

SPARK_MONTHS = 12
REMEMBER_KEY = "remember-merchant"
CASH_NOTE = "Cash spending is not tracked after withdrawal unless you log it in the Cash wallet card."
# Short uppercase label per insight kind, as on the reference cards (UP / DOWN / CHECK ...).
INSIGHT_TAG = {"category_increase": "Up", "category_decrease": "Down", "unused_subscription": "Check",
               "anomaly": "Unusual", "cash_share": "Cash"}


def go_to_plan(insight_text: str, kind: str) -> None:
    """'Fix this': open Your Plan with the insight as context."""
    st.session_state["nav"] = "Your Plan"
    st.session_state["focus"] = {"text": insight_text, "kind": kind}


def render(db_path: str, key: str, month: str, mode: str) -> None:
    with st.spinner("Crunching your numbers..."):
        history = data.wallet_history(db_path, key)   # counts logged + unaccounted cash (analytics.apply_cash)
        view = data.wallet_month_view(db_path, key, month)
        flagged = data.anomaly_months(db_path, key)
        cash = data.cash_view(db_path, key, month)
    mom = view["mom"]
    _kpis(history, month, mom.previous_month, mode)
    _categories(mom, mode)
    if cash is not None:  # only when the data has cash withdrawals
        _cash_wallet(db_path, key, month, cash)
    _trend_and_insights(history, month, flagged, view["insights"], mode)
    _merchants_and_days(view, month, mode)
    _history_table(history, month)
    _transactions(db_path, key, month)


# ---- 1. KPI row ---------------------------------------------------------------------------------

def _kpis(history, month: str, previous_month: str | None, mode: str) -> None:
    idx = next(i for i, r in enumerate(history) if r.month == month)
    row = history[idx]
    window = history[max(0, idx - SPARK_MONTHS + 1): idx + 1]
    months = [r.month for r in window]
    specs = [
        ("total_spent", "Total Spent", fmt.money(row.total_spent), row.spent, [r.total_spent for r in window], False),
        ("total_income", "Income", fmt.money(row.total_income), row.income, [r.total_income for r in window], False),
        ("net_saved", "Net Saved", fmt.money(row.net_saved), row.net, [r.net_saved for r in window], False),
        ("savings_rate", "Savings Rate", fmt.pct(row.savings_rate), row.rate, [r.savings_rate for r in window], True),
    ]
    for col, (metric, label, value, ch, series, is_rate) in zip(st.columns(4, gap="medium"), specs):
        with col, card(f"kpi-{metric}"):
            st.markdown(kpi_html(metric, label, value, ch, previous_month, is_rate), unsafe_allow_html=True)
            if len([v for v in series if v is not None]) >= 2:
                st.plotly_chart(charts.sparkline(months, series, mode, as_pct=is_rate), config=charts.CONFIG, theme=None,
                                key=f"spark-{metric}")


# ---- 2. Categories ------------------------------------------------------------------------------

def _categories(mom, mode: str) -> None:
    left, right = st.columns([5, 7], gap="medium")
    summary = mom.current
    with left, card("donut"):
        card_head("Where your money went", f"{fmt.month_label(mom.month)}, by category")
        if summary.categories:
            pie, legend = st.columns([1, 1.25], vertical_alignment="center")
            with pie:
                st.plotly_chart(charts.category_donut(summary, mode, legend=False), config=charts.CONFIG, theme=None, key="donut")
            legend.markdown(legend_html([(category_color(c.category), c.category, fmt.money(c.amount),
                                          fmt.pct(c.pct_of_spend)) for c in summary.categories]),
                            unsafe_allow_html=True)
        else:
            empty_card("No spending this month")
        for name in (UNACCOUNTED, CASH):
            if summary.category(name):
                muted(f"{name}: {CASH_NOTE}")
                break
    with right, card("cat-bars"):
        if mom.previous_month:
            card_head("This month vs last month",
                      f"{fmt.month_label(mom.month)} compared with {fmt.month_label(mom.previous_month)}")
        else:
            card_head("This month vs last month", f"{NO_EARLIER}: showing {fmt.month_label(mom.month)} only")
        if any(c.amount.current or c.amount.previous for c in mom.categories):
            st.plotly_chart(charts.category_bars(mom, mode), config=charts.CONFIG, theme=None, key="cat-bars")
        else:
            empty_card("No spending to compare")


# ---- 2b. Cash wallet ----------------------------------------------------------------------------

CASH_MSG = "cash-msg"


RECENT_CASH = 3
CASH_EMPTY = "No cash logged yet. Try: chai 150"
CASH_HELP = "Unaccounted cash counts as spending until you log it."


def _cash_stats_html(cash) -> str:
    items = [("withdrawn", "Withdrawn", fmt.money(cash.withdrawn), ""),
             ("logged", "Logged", fmt.money(cash.logged), ""),
             ("unaccounted", "Unaccounted", fmt.money(cash.unaccounted), "warn")]
    return '<div class="fa-cash-stats">' + "".join(
        f'<div data-cash="{name}"><div class="l">{escape(label)}</div><div class="v {tone}">{escape(value)}</div></div>'
        for name, label, value, tone in items) + "</div>"


def _cash_head_html(month: str, cash) -> str:
    """Title, month and a (?) tooltip holding the wallet balance (information only) and the one rule."""
    tip = (f"Wallet balance (all months): {fmt.money(cash.wallet_balance, signed=cash.wallet_balance < 0)}. "
           f"{CASH_HELP}")
    return (f'<div class="fa-card-head"><div class="title">Cash wallet '
            f'<span class="fa-help" data-cash="balance" title="{escape(tip)}">?</span></div>'
            f'<div class="sub">{escape(fmt.month_label(month))}</div></div>')


def _cash_bar_html(pct: float) -> str:
    width = min(max(pct, 0.0), 100.0)
    return (f'<div class="fa-cash-bar" data-cash="bar"><div class="track"><div class="fill" style="width:{width:.1f}%">'
            f'</div></div><div class="l">{escape(fmt.pct(pct, 0))} accounted for</div></div>')


def _cash_done(text: str) -> None:
    st.session_state[CASH_MSG] = None
    st.toast(text)


def _cash_date(default):
    return st.session_state.get("cash-date") or default


def _add_cash(db_path: str) -> None:
    """'More options': amount, category, date and note typed in full."""
    s = st.session_state
    try:
        data.add_cash(db_path, s.get("cash-amount") or 0, s.get("cash-cat") or "", s.get("cash-date"),
                      s.get("cash-note") or "")
    except CashEntryError as e:
        s[CASH_MSG] = str(e)
        return
    s["cash-amount"], s["cash-cat"], s["cash-note"] = None, None, ""
    _cash_done("Cash entry added. Figures updated.")


def _propose_quick() -> None:
    """Parse the quick line and propose a category with the keyword rules only (no model)."""
    s = st.session_state
    try:
        line = parse_quick_line(s.get("cash-quick") or "")
    except CashEntryError as e:
        s["cash-proposal"], s[CASH_MSG] = None, str(e)
        return
    s[CASH_MSG] = None
    s["cash-proposal"] = {"amount": line.amount, "note": line.note, "category": line.category}
    s["cash-quick-cat"] = line.category  # None when no rule matched: the user picks


def _confirm_quick(db_path: str, default_date) -> None:
    s = st.session_state
    p = s.get("cash-proposal") or {}
    try:
        data.add_cash(db_path, p.get("amount") or 0, s.get("cash-quick-cat") or "", _cash_date(default_date),
                      p.get("note", ""))
    except CashEntryError as e:
        s[CASH_MSG] = str(e)
        return
    s["cash-proposal"], s["cash-quick"] = None, ""
    _cash_done("Cash entry added. Figures updated.")


def _edit_cash(entry_id: int | None) -> None:
    st.session_state["cash-edit-id"] = entry_id


def _save_cash_edit(db_path: str, entry_id: int) -> None:
    s = st.session_state
    try:
        data.update_cash(db_path, entry_id, s.get(f"cash-edit-amount-{entry_id}") or 0,
                         s.get(f"cash-edit-cat-{entry_id}") or "", s.get(f"cash-edit-date-{entry_id}"),
                         s.get(f"cash-edit-note-{entry_id}") or "")
    except CashEntryError as e:
        s[CASH_MSG] = str(e)
        return
    s["cash-edit-id"] = None
    _cash_done("Cash entry saved. Figures updated.")


def _delete_cash(db_path: str, entry_id: int) -> None:
    data.delete_cash(db_path, entry_id)
    if st.session_state.get("cash-edit-id") == entry_id:
        st.session_state["cash-edit-id"] = None
    _cash_done("Cash entry deleted. Figures updated.")


def _cash_wallet(db_path: str, key: str, month: str, cash) -> None:
    first, last = data.date_range(db_path, key)
    with card("cash"):
        st.markdown(_cash_head_html(month, cash), unsafe_allow_html=True)
        st.markdown(_cash_stats_html(cash), unsafe_allow_html=True)
        if cash.accounted_pct is not None:
            st.markdown(_cash_bar_html(cash.accounted_pct), unsafe_allow_html=True)
        else:
            muted(f"No cash withdrawn in {fmt.month_name(month)}.")
        if cash.outside:
            st.markdown(f'<div class="fa-pill warn" data-cash="outside">{escape(fmt.money(cash.outside))} logged '
                        f'above withdrawals: {escape(OUTSIDE_FLAG)}, counted as spending</div>', unsafe_allow_html=True)
        if st.session_state.get(CASH_MSG):
            st.warning(st.session_state[CASH_MSG])
        _cash_quick_add(db_path, last)
        _cash_more_options(db_path, first, last)
        _cash_recent(db_path, key, first, last)


def _cash_quick_add(db_path: str, last) -> None:
    with st.form("cash-quick-form", border=False):
        q1, q2 = st.columns([5, 1], vertical_alignment="bottom")
        q1.text_input("Add cash", key="cash-quick", placeholder="e.g. chai 150", label_visibility="collapsed")
        q2.form_submit_button("Add", type="primary", key="cash-quick-go", width="stretch", on_click=_propose_quick)
    proposal = st.session_state.get("cash-proposal")
    if not proposal:
        return
    amount = fmt.money(proposal["amount"], cents=proposal["amount"] < 100)
    with st.container(key="cash-confirm"):
        c1, c2, c3 = st.columns([3, 2, 1], vertical_alignment="center")
        found = st.session_state.get("cash-quick-cat")
        line = f"{found} - {amount}" if found else f"Pick a category for {amount}"
        c1.markdown(f'<div class="fa-cash-confirm" data-cash="proposal">{escape(line)}</div>', unsafe_allow_html=True)
        c2.selectbox("Change" if proposal["category"] else "Category", list(ENTRY_CATEGORIES), key="cash-quick-cat",
                     placeholder="Pick a category", label_visibility="collapsed")
        c3.button("Confirm", key="cash-quick-save", type="primary", width="stretch", on_click=_confirm_quick,
                  args=(db_path, last))


def _cash_more_options(db_path: str, first, last) -> None:
    with st.expander("More options", expanded=False):
        c1, c2 = st.columns(2)
        c1.number_input("Amount", min_value=0.0, step=10.0, value=None, key="cash-amount", placeholder="0")
        c2.selectbox("Category", list(ENTRY_CATEGORIES), index=None, key="cash-cat", placeholder="Pick a category")
        c3, c4 = st.columns(2)
        c3.date_input("Date", value=last, min_value=first, max_value=last, key="cash-date")
        c4.text_input("Note (optional)", key="cash-note", max_chars=80,
                      help="Names are masked before saving. Notes are never sent to an AI model.")
        st.button("Add entry", key="cash-add", on_click=_add_cash, args=(db_path,))


def _cash_recent(db_path: str, key: str, first, last) -> None:
    """The newest cash entries (all months) with edit and delete icons."""
    entries = data.cash_entries(db_path, key)
    with st.container(key="cash-recent"):
        if entries.empty:
            st.markdown(f'<div class="fa-muted" data-cash="empty">{escape(CASH_EMPTY)}</div>', unsafe_allow_html=True)
            return
        recent = entries.sort_values(["date", "id"], ascending=False).head(RECENT_CASH)
        for e in recent.itertuples():
            eid, day = int(e.id), pd.Timestamp(e.date).date()
            c1, c2, c3, c4, c5 = st.columns([1.2, 3, 1.6, 0.5, 0.5], vertical_alignment="center")
            c1.markdown(f'<span class="fa-muted">{escape(_short_date(day))}</span>', unsafe_allow_html=True)
            c2.markdown(f'<span class="fa-cash-what">{escape(str(e.note or "") or e.category)}</span>',
                        unsafe_allow_html=True)
            c3.markdown(f'<div class="fa-cash-amt">{escape(fmt.money(e.amount, cents=e.amount < 100))}</div>',
                        unsafe_allow_html=True)
            c4.button("", icon=":material/edit:", key=f"cash-edit-{eid}", help="Edit", type="tertiary",
                      on_click=_edit_cash, args=(eid,))
            c5.button("", icon=":material/delete:", key=f"cash-del-{eid}", help="Delete", type="tertiary",
                      on_click=_delete_cash, args=(db_path, eid))
        editing = st.session_state.get("cash-edit-id")
        if editing in set(entries["id"].astype(int)):
            _cash_edit_form(db_path, entries.set_index("id").loc[editing], int(editing), first, last)


def _cash_edit_form(db_path: str, e, entry_id: int, first, last) -> None:
    with st.container(key="cash-edit-box"):
        c1, c2, c3, c4 = st.columns([1.2, 1.6, 1.3, 2])
        c1.number_input("Amount", min_value=0.0, step=10.0, value=float(e["amount"]), key=f"cash-edit-amount-{entry_id}")
        c2.selectbox("Category", list(ENTRY_CATEGORIES), key=f"cash-edit-cat-{entry_id}",
                     index=ENTRY_CATEGORIES.index(e["category"]) if e["category"] in ENTRY_CATEGORIES else None)
        day = min(max(pd.Timestamp(e["date"]).date(), first), last)
        c3.date_input("Date", value=day, min_value=first, max_value=last, key=f"cash-edit-date-{entry_id}")
        c4.text_input("Note", value=str(e["note"] or ""), key=f"cash-edit-note-{entry_id}", max_chars=80)
        b1, b2, _ = st.columns([1, 1, 4])
        b1.button("Save", key="cash-edit-save", type="primary", width="stretch", on_click=_save_cash_edit,
                  args=(db_path, entry_id))
        b2.button("Cancel", key="cash-edit-cancel", width="stretch", on_click=_edit_cash, args=(None,))


# ---- 3. Trend + insights ------------------------------------------------------------------------

def _trend_subtitle(history, month: str, flagged: list[str]) -> str:
    shown = {r.month for r in history}
    marks = [fmt.month_name(m) for m in flagged if m in shown]
    text = f"Total spent per month. {fmt.month_name(month)} is selected"
    if marks:
        text += f"; {', '.join(marks)} {'has' if len(marks) == 1 else 'have'} an unusual charge"
    return text + "."


def _trend_and_insights(history, month: str, flagged: list[str], insights, mode: str) -> None:
    left, right = st.columns([7, 5], gap="medium")
    with left, card("trend"):
        card_head("Spending trend", _trend_subtitle(history, month, flagged) if len(history) > 1 else "")
        if len(history) < 2:
            empty_card("Not enough history for a trend yet",
                       "A trend needs at least two months of data. Upload a longer statement to see one.")
        else:
            st.plotly_chart(charts.spending_trend(history, month, flagged, mode), config=charts.CONFIG, theme=None, key="trend")
    with right, card("insights"):
        card_head("What changed", "Computed from your statement, ranked by money impact")
        if not insights:
            empty_card("Nothing unusual this month", "No big changes, unused subscriptions or odd charges.")
        for i, ins in enumerate(insights):
            tone = insight_tone(ins.kind)
            with st.container(key=f"insight-{tone}-{i}"):
                st.markdown(f'<div class="fa-insight" data-kind="{ins.kind}"><div class="fa-ins-kind {tone}">'
                            f'{escape(INSIGHT_TAG.get(ins.kind, ins.kind))}</div>'
                            f'<div class="fa-ins-text">{escape(ins.text)}</div></div>', unsafe_allow_html=True)
                if tone != "good":  # a praise card has nothing to fix
                    st.button("Fix this", key=f"fix-{i}", on_click=go_to_plan, args=(ins.text, ins.kind),
                              help="Open Your Plan to see what this is worth")


# ---- 4. Merchants + daily -----------------------------------------------------------------------

def _merchants_and_days(view, month: str, mode: str) -> None:
    left, right = st.columns(2, gap="medium")
    merchants = view["merchants"]
    with left, card("merchants"):
        card_head(f"Top {len(merchants) or 5} merchants", f"Where the most money went in {fmt.month_name(month)}")
        if merchants:
            top = max(m.amount for m in merchants) or 1
            st.markdown(merchant_bars_html([(m.merchant.title(), fmt.money(m.amount), category_color(m.category),
                                             m.amount / top) for m in merchants]), unsafe_allow_html=True)
        else:
            empty_card("No spending this month")
    daily = view["daily"]
    with right, card("daily"):
        peak = daily.peak
        card_head("Daily spending", f"Highest day: {fmt.day_label(peak.date)}, {fmt.money(peak.amount)}"
                  if peak else "No spending this month")
        if peak:
            st.plotly_chart(charts.daily_bars(daily, mode), config=charts.CONFIG, theme=None, key="daily")
        else:
            empty_card("No spending this month")


# ---- 5. Month history ---------------------------------------------------------------------------

def _history_table(history, month: str) -> None:
    rows = []
    for r in history:
        change = r.spent
        if change.abs_change is None:
            cell = ("No previous month", "flat")
        else:
            arrow = "▲" if change.abs_change > 0 else ("▼" if change.abs_change < 0 else "■")
            text = (f"{arrow} {fmt.pct(abs(change.pct_change))}" if change.pct_change is not None
                    else f"{arrow} {fmt.money(abs(change.abs_change))}")
            cell = (text, delta_tone("total_spent", change.abs_change))
        rows.append([(fmt.month_name(r.month) if len({h.month[:4] for h in history}) == 1 else fmt.month_label(r.month),
                      "month"), (fmt.money(r.total_spent), ""), (fmt.money(r.total_income), ""),
                     (fmt.money(r.net_saved), ""), (fmt.pct(r.savings_rate), ""), cell])
    selected = next(i for i, r in enumerate(history) if r.month == month)
    with card("history"):
        card_head("Month history", "Every month side by side. Change is spending versus the month before.")
        st.markdown(table_html(["Month", "Spent", "Income", "Saved", "Savings rate", "Change"], rows, selected),
                    unsafe_allow_html=True)


# ---- 6. Transactions ----------------------------------------------------------------------------

ALL = "All"
REVIEW_FILTER = "Needs review"
PAGE_ROWS = 25
EMPTY_FILTER = "No transactions in this category"
SOURCE_BADGE = {"rule": "Rule", "llm": "AI", "manual": "Manual", "learned": "Learned", CASH_SOURCE: "Cash"}


def _short_date(d) -> str:
    return f"{d:%b} {d.day}"


def _amount(value: float) -> str:
    """Signed money for the ledger: "-Rs 355" out, "+Rs 4,850" in; cents only below 100."""
    return fmt.money(value, cents=abs(value) < 100, signed=True)


def _review_label(n: int) -> str:
    return f"{REVIEW_FILTER} ({n})"


def ledger_table_html(rows) -> str:
    """The ledger as a styled HTML table (reference design). Values come from analytics LedgerRow objects."""
    body = []
    for r in rows:
        badge_cls, badge = (("warn", REVIEW_FILTER) if r.needs_review else
                            ("cash" if r.source == CASH_SOURCE else "", SOURCE_BADGE.get(r.source, "None")))
        tone = "in" if r.amount > 0 else "out"
        body.append(
            f'<tr data-txn="{r.txn_id}"><td class="date">{escape(_short_date(r.date))}</td>'
            f'<td class="merchant">{escape(r.merchant.title())}</td>'
            f'<td><span class="fa-cat"><span class="fa-dot" style="background:{category_color(r.category)}"></span>'
            f'{escape(r.category)}</span></td>'
            f'<td><span class="fa-src {badge_cls}">{escape(badge)}</span></td>'
            f'<td class="amt {tone}">{escape(_amount(r.amount))}</td></tr>')
    return ('<table class="fa-ledger"><thead><tr><th>Date</th><th>Merchant</th><th>Category</th><th>Source</th>'
            f'<th class="amt">Amount</th></tr></thead><tbody>{"".join(body)}</tbody></table>')


def _filter_css(options: list[str], key: str) -> str:
    """Colored dots on the filter pills, generated from the categories present (same colors as the donut)."""
    rules = []
    for i, opt in enumerate(options, start=1):
        if opt == ALL:
            continue
        color = WARNING if opt.startswith(REVIEW_FILTER) else category_color(opt)
        rules.append(f'.st-key-{key} button:nth-of-type({i}) p::before {{ content:""; display:inline-block; width:8px; '
                     f'height:8px; border-radius:50%; margin-right:7px; vertical-align:middle; background:{color}; }}')
    return "<style>" + "\n".join(rules) + "</style>"


def _save_fix(db_path: str) -> None:
    """Same override path as before: data.set_category -> categorizer.set_manual_category (+ merchant rule)."""
    s = st.session_state
    txn_id = s.get("fix-txn")
    category = s.get(f"fix-cat-{txn_id}")
    if txn_id is None or not category:
        return
    data.set_category(db_path, int(txn_id), category, remember=bool(s.get(REMEMBER_KEY)))
    st.toast(f"Saved: {category}. Figures updated.")


def _transactions(db_path: str, key: str, month: str) -> None:
    rows = data.ledger(db_path, key, month)  # newest first
    present = [c for c in CATEGORIES if any(r.category == c for r in rows)]
    present += sorted({r.category for r in rows} - set(present))
    n_review = sum(r.needs_review for r in rows)
    options = [ALL, *present] + ([_review_label(n_review)] if n_review else [])
    filter_key = f"txn-filter-{month}"
    prev = st.session_state.get(filter_key)
    if prev not in options:
        # A changed review count maps to the new label; a filter whose last row was just re-labelled
        # (or an emptied review queue) switches back to All.
        if isinstance(prev, str) and prev.startswith(REVIEW_FILTER) and n_review:
            st.session_state[filter_key] = _review_label(n_review)
        else:
            st.session_state[filter_key] = ALL
    with card("ledger"):
        card_head("Transactions", "Filter by category. Fix a label below and it is saved as a manual override.")
        st.markdown(_filter_css(options, filter_key), unsafe_allow_html=True)
        chosen = st.pills("Filter by category", options, key=filter_key, label_visibility="collapsed") or ALL
        if chosen == ALL:
            shown = rows
        elif chosen.startswith(REVIEW_FILTER):
            shown = [r for r in rows if r.needs_review]
        else:
            shown = [r for r in rows if r.category == chosen]
        if not shown:
            muted(EMPTY_FILTER)
        else:
            pages_key = f"txn-pages-{month}-{chosen}"
            limit = PAGE_ROWS * st.session_state.get(pages_key, 1)
            st.markdown(ledger_table_html(shown[:limit]), unsafe_allow_html=True)
            left, right = st.columns([3, 1], vertical_alignment="center")
            left.markdown(f'<div class="fa-muted">Showing {min(limit, len(shown))} of {len(shown)} transactions · '
                          'descriptions are shown redacted, as a model would see them</div>', unsafe_allow_html=True)
            if len(shown) > limit:
                right.button("Show more", key="txn-more", width="stretch",
                             on_click=lambda: st.session_state.update({pages_key: limit // PAGE_ROWS + 1}))
        # cash entries are edited in the Cash wallet card, not here
        _fix_control(db_path, [r for r in (shown or rows) if r.source != CASH_SOURCE])


def _fix_control(db_path: str, rows) -> None:
    if not rows:
        return
    with st.container(key="fix-box"):
        st.markdown('<div class="fa-card-head"><div class="title">Fix a category</div></div>', unsafe_allow_html=True)
        by_id = {r.txn_id: r for r in rows}
        if st.session_state.get("fix-txn") not in by_id:
            st.session_state["fix-txn"] = rows[0].txn_id
        c1, c2, c3 = st.columns([5, 3, 1.2], vertical_alignment="bottom")
        txn_id = c1.selectbox("Transaction", list(by_id), key="fix-txn",
                              format_func=lambda i: f"{_short_date(by_id[i].date)} · {by_id[i].merchant.title()} · "
                                                    f"{_amount(by_id[i].amount)}")
        current = by_id[txn_id].category
        c2.selectbox("New category", list(CATEGORIES), key=f"fix-cat-{txn_id}",
                     index=CATEGORIES.index(current) if current in CATEGORIES else None, placeholder="Pick a category")
        c3.button("Save", key="fix-save", type="primary", width="stretch", on_click=_save_fix, args=(db_path,))
        st.toggle("Apply my fix to every transaction from the same merchant", key=REMEMBER_KEY,
                  help="Saves a merchant rule; the AI also learns from it")
