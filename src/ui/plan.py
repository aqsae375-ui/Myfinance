"""Your Plan tab (goal-driven).

No goal yet: only a "Set your goal" form plus a short explanation; every goal-based card shows "Set a goal
to see this". With a goal: the goal card, three result cards, ranked opportunities with checkboxes and the
current-vs-plan chart with the goal line, all from src/advisor on the user's own data and goal. The goal is
validated (positive amount, future deadline) and an unrealistic goal is called out in plain words.

Speed: the opportunities are computed once per data state (data.base_plan, no model call); toggling a
checkbox or changing the goal only runs advisor.reproject, which is pure arithmetic. Friendly sentences
come from plan.py's verified LLM phrasing only when a real model is selected and the user asks for it.
"""
from __future__ import annotations

import logging
from datetime import date
from html import escape

import streamlit as st

from src.advisor import Goal, clear_goals, load_goal, reproject, save_goal
from src.advisor.goals import BY_DATE, MONTHLY
from src.advisor.projector import GOAL_MET, INSUFFICIENT, cumulative_path
from src.db import connect
from src.ui import charts, data
from src.ui import format as fmt
from src.ui.components import card, card_head, empty_card, footer, muted, pill
from src.ui.theme import REAL_MODELS

log = logging.getLogger("finance_agent.ui")

KIND_LABELS = {BY_DATE: "Target by date", MONTHLY: "Monthly target"}
SET_GOAL_FIRST = "Set a goal to see this"
TOO_LONG_MONTHS = 120  # beyond 10 years at the current pace, a goal is called unrealistic
FORM_KEYS = ("goal_kind", "goal_name", "goal_amount", "goal_date", "goal_saved")


# ---- goal storage and the form ------------------------------------------------------------------

def _goal(db_path: str) -> Goal | None:
    conn = connect(db_path)
    try:
        return load_goal(conn)
    finally:
        conn.close()


def goal_title(goal: Goal) -> str:
    if goal.kind == BY_DATE:
        d = goal.target_date
        return f"Save {fmt.money(goal.amount)} by {d:%b} {d.day}, {d.year}"
    return f"Save {fmt.money(goal.amount)} every month"


def validate_goal(kind: str, amount, target_date, saved, today: date | None = None) -> str | None:
    """A plain-English problem with the form values, or None when they make a valid goal."""
    today = today or date.today()
    if amount is None or amount <= 0:
        return "Enter a target amount above zero."
    if kind == BY_DATE:
        if target_date is None:
            return "Pick a deadline."
        if target_date <= today:
            return "The deadline must be in the future."
        if saved is not None and saved < 0:
            return "Already saved cannot be negative."
        if saved is not None and saved >= amount:
            return "You have already saved the full target: pick a bigger target."
    return None


def _save_goal(db_path: str) -> None:
    s = st.session_state
    kind = s.get("goal_kind") or BY_DATE
    error = validate_goal(kind, s.get("goal_amount"), s.get("goal_date"), s.get("goal_saved"))
    if error:
        s["goal_error"] = error
        return
    name = (s.get("goal_name") or "").strip()
    try:
        goal = (Goal.by_date(s["goal_amount"], s["goal_date"], s.get("goal_saved") or 0.0, name=name or "Savings goal")
                if kind == BY_DATE else Goal.monthly(s["goal_amount"], name=name or "Monthly savings"))
    except ValueError as e:
        s["goal_error"] = str(e)
        return
    conn = connect(db_path)
    try:
        save_goal(conn, goal)
    finally:
        conn.close()
    s.pop("goal_error", None)
    s["goal_editing"] = False
    st.toast("Goal saved.")


def _edit_goal(goal: Goal) -> None:
    s = st.session_state
    s.update(goal_editing=True, goal_kind=goal.kind, goal_name=goal.name, goal_amount=float(goal.amount),
             goal_date=goal.target_date, goal_saved=float(goal.saved_so_far))


def _remove_goal(db_path: str) -> None:
    conn = connect(db_path)
    try:
        clear_goals(conn)
    finally:
        conn.close()
    for k in (*FORM_KEYS, "goal_editing", "goal_error"):
        st.session_state.pop(k, None)


def _cancel_edit() -> None:
    st.session_state["goal_editing"] = False
    st.session_state.pop("goal_error", None)


def _empty(key: str) -> dict:
    """Start a form field empty (no made-up default); once the key is in session state, it holds the value."""
    return {} if key in st.session_state else {"value": None}


def _goal_form(db_path: str, editing: bool) -> None:
    s = st.session_state
    s.setdefault("goal_kind", BY_DATE)
    with card("goal-form"):
        card_head("Edit your goal" if editing else "Set your goal",
                  "Your plan is built around one goal: an amount by a date, or an amount to save every month. "
                  "Every result is then calculated from your own statement.")
        st.segmented_control("Goal type", list(KIND_LABELS), format_func=KIND_LABELS.get, key="goal_kind",
                             required=True)
        by_date = (s.get("goal_kind") or BY_DATE) == BY_DATE
        cols = st.columns([3, 2, 2, 2] if by_date else [3, 2])
        cols[0].text_input("Name (optional)", key="goal_name", placeholder="e.g. Emergency fund")
        cols[1].number_input(f"Target ({fmt.CURRENCY})" if by_date else f"Save per month ({fmt.CURRENCY})",
                             min_value=0.0, step=500.0, key="goal_amount", format="%.0f", placeholder="Amount",
                             **_empty("goal_amount"))
        if by_date:
            cols[2].date_input("Deadline", key="goal_date", min_value=date.today(), **_empty("goal_date"))
            cols[3].number_input(f"Already saved ({fmt.CURRENCY})", min_value=0.0, step=500.0, key="goal_saved",
                                 format="%.0f", placeholder="0", **_empty("goal_saved"))
        b1, b2, _ = st.columns([1, 1, 4])
        b1.button("Save goal", key="goal-save", type="primary", on_click=_save_goal, args=(db_path,), width="stretch")
        if editing:
            b2.button("Cancel", key="goal-cancel", on_click=_cancel_edit, width="stretch")
        if s.get("goal_error"):
            st.error(s["goal_error"])


# ---- page ---------------------------------------------------------------------------------------

def _matches(focus: dict | None, opp) -> bool:
    if not focus:
        return False
    text = focus["text"].lower()
    return opp.target.lower() in text or opp.category.lower() in text


def render(db_path: str, key: str, provider_name: str, mode: str) -> None:
    goal = _goal(db_path)
    focus = st.session_state.get("focus")
    editing = bool(st.session_state.get("goal_editing"))
    with st.spinner("Finding ways to save..."):
        base = data.base_plan(db_path, key)
    _focus_banner(focus, base)

    if goal is None or editing:
        _goal_form(db_path, editing and goal is not None)
    if goal is None:
        _no_goal_cards()
        footer()
        return

    # Stored outside widget state: Streamlit drops a widget's state while its tab isn't rendered.
    choices = st.session_state.setdefault("accepted", {})
    accepted = [i.opportunity.id for i in base.items if choices.get(i.opportunity.id, True)]
    plan = reproject(base, accepted, goal)
    p = plan.projection
    if not editing:
        _top_row(db_path, goal, p)
    if honest := realism_message(p):
        st.markdown(f'<div class="fa-honest" data-testid="goal-honest">{escape(honest)}</div>', unsafe_allow_html=True)

    phrased = _phrased(db_path, key, provider_name)
    left, right = st.columns([11, 10], gap="medium")
    with left:
        st.markdown('<div class="fa-card-head"><div class="title">Ways to save, ranked</div><div class="sub">'
                    'Tick the ones you will do. The chart updates instantly.</div></div>', unsafe_allow_html=True)
        if not plan.items:
            with card("no-opps"):
                empty_card("No savings opportunities found yet", "More months of history may reveal some.")
        for item in plan.items:
            _opportunity_card(item, phrased, _matches(focus, item.opportunity))
        if provider_name in REAL_MODELS and plan.items:
            st.button(f"✦ Write friendly explanations with {provider_name}", key="phrase",
                      help="One model call per suggestion; every number it writes is checked. "
                           "Local models can take a few minutes on CPU.",
                      on_click=_phrase, args=(db_path, key, provider_name))
            if err := st.session_state.pop("phrase_error", None):
                st.warning(f"The {provider_name} model couldn't phrase the plan ({err}); showing template sentences.")
    with right:
        _projection(plan, mode)
    footer()


def _no_goal_cards() -> None:
    cols = st.columns(3, gap="medium")
    for col, (k, label) in zip(cols, [("res-surplus", "Monthly surplus"), ("res-months", "Months to goal"),
                                      ("res-12m", "12-month savings")]):
        with col, card(k):
            st.markdown(f'<div class="fa-kpi"><div class="label plain">{label}</div></div>', unsafe_allow_html=True)
            empty_card(SET_GOAL_FIRST)
    left, right = st.columns([11, 10], gap="medium")
    with left, card("opps-empty"):
        card_head("Ways to save, ranked", "Tick the ones you will do. The chart updates instantly.")
        empty_card(SET_GOAL_FIRST, "Your ranked savings ideas appear once a goal is set.")
    with right, card("projection-empty"):
        card_head("Current path vs your plan", "Cumulative savings over 12 months")
        empty_card(SET_GOAL_FIRST)


def _focus_banner(focus: dict | None, base) -> None:
    if not focus:
        return
    hit = any(_matches(focus, i.opportunity) for i in base.items)
    with card("focus"):
        st.markdown(f"{pill('From your dashboard', 'info')} {escape(focus['text'])}", unsafe_allow_html=True)
        muted("The matching suggestion is highlighted below." if hit else
              "No plan item matches this one directly; here are your best options.")
        st.button("Dismiss", key="focus-dismiss", type="tertiary", on_click=lambda: st.session_state.pop("focus", None))


def _result(label: str, value: str, delta: str = "", tone: str = "neutral", caption: str = "") -> str:
    row = (pill(delta, tone) if delta else "") + (f'<span class="caption">{escape(caption)}</span>' if caption else "")
    return (f'<div class="fa-kpi" data-result="{escape(label)}"><div class="label plain">{escape(label)}</div>'
            f'<div class="value">{escape(value)}</div><div class="row">{row}</div></div>')


def _top_row(db_path: str, goal: Goal, p) -> None:
    c0, c1, c2, c3 = st.columns([6, 3, 3, 3], gap="medium")
    with c0, card("goal"):
        st.markdown(f'<div class="fa-kpi"><div class="label plain">Your goal</div></div>'
                    f'<div class="fa-goal-title">{escape(goal_title(goal))}</div>', unsafe_allow_html=True)
        b1, b2, b3 = st.columns([1.3, 1, 1], gap="small")
        b1.markdown(pill(KIND_LABELS[goal.kind], "good"), unsafe_allow_html=True)
        b2.button("Edit goal", key="goal-edit", on_click=_edit_goal, args=(goal,), width="stretch")
        b3.button("Remove", key="goal-clear", on_click=_remove_goal, args=(db_path,), width="stretch")
    ready = p.status != INSUFFICIENT and p.current_surplus is not None
    with c1, card("res-surplus"):
        st.markdown(_result("Monthly surplus", fmt.money(p.new_surplus) if ready else fmt.NA,
                            f"▲ {fmt.money(p.added_saving)}" if ready and p.added_saving else "", "good",
                            f"vs {fmt.money(p.current_surplus)} now" if ready else "Needs 2+ months of history"),
                    unsafe_allow_html=True)
    with c2, card("res-months"):
        if goal.kind == BY_DATE:
            after, before, saved = p.months_to_goal_after, p.months_to_goal_before, p.months_saved
            value = fmt.NA if not ready else ("Reached" if p.status == GOAL_MET else
                                              (str(after) if after is not None else "Never"))
            delta = f"▼ {saved} month{'s' if saved != 1 else ''}" if ready and saved else ""
            caption = (f"vs {before if before is not None else 'never'} now" if ready and p.status != GOAL_MET else "")
            st.markdown(_result("Months to goal", value, delta, "good", caption), unsafe_allow_html=True)
        else:
            st.markdown(_result("Monthly target", fmt.money(goal.amount),
                                ("On track with the plan" if p.on_track_after else "Not reached yet") if ready else "",
                                "good" if p.on_track_after else "bad"), unsafe_allow_html=True)
    with c3, card("res-12m"):
        after12, before12 = p.cumulative_after[12], p.cumulative_before[12]
        st.markdown(_result("12-month savings", fmt.money(after12) if ready else fmt.NA,
                            f"▲ {fmt.money(p.plan_only[12])}" if ready and p.plan_only[12] else "", "good",
                            f"vs {fmt.money(before12)} now" if ready else ""), unsafe_allow_html=True)


def realism_message(p) -> str | None:
    """Say plainly when the goal does not fit the data (uses only the projection's fields)."""
    g = p.goal
    if g is None:
        return None
    if p.status == INSUFFICIENT:
        return (f"There is only {p.months_of_history} month of history, so your savings can't be projected yet. "
                "Upload a statement with at least two months to see when you'd reach this goal.")
    if p.status == GOAL_MET:
        return None
    if p.new_surplus is not None and p.new_surplus <= 0:
        return (f"On average you spend more than you earn ({fmt.money(p.current_surplus)} a month), so this goal "
                "can't be reached at your current pace, even with every suggestion ticked.")
    if g.kind == BY_DATE:
        after = p.months_to_goal_after
        if after is not None and after > TOO_LONG_MONTHS:
            return (f"At this pace the goal would take about {after} months (over ten years). That is unrealistic "
                    "for this budget: consider a smaller target.")
        if not p.on_track_after and after is not None:
            return (f"Even with the ticked suggestions you'd need about {after} months, but the deadline is "
                    f"{p.months_left} months away. Consider a later deadline or a smaller target.")
    elif g.kind == MONTHLY and not p.on_track_after:
        return (f"Your goal is {fmt.money(g.amount)} a month, but with the ticked suggestions you'd save about "
                f"{fmt.money(p.new_surplus)} a month. The goal is not reachable yet with these changes.")
    return None


def _opportunity_card(item, phrased: dict, highlighted: bool) -> None:
    o = item.opportunity
    sentence, source = phrased.get(o.id, (item.explanation, item.explanation_source))
    with card(f"opp-{o.rank}"):
        top, save = st.columns([3, 1.3], vertical_alignment="center")
        with top:
            st.checkbox(f"**{o.action}**", value=item.accepted, key=f"accept-{o.id}", on_change=_toggle, args=(o.id,))
        with save:
            st.markdown(f'<div style="text-align:right"><span class="fa-save-pill">'
                        f'{escape(fmt.money(o.monthly_saving))} / month</span></div>', unsafe_allow_html=True)
        badges = []
        if highlighted:
            badges.append(pill("★ From your dashboard", "warn"))
        if source == "llm":
            badges.append(pill("✦ Model-written, numbers verified", "info"))
        if badges:
            st.markdown(" ".join(badges), unsafe_allow_html=True)
        st.markdown(f'<div class="fa-opp-text">{escape(sentence)}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="fa-calc">{escape(o.calculation)}</div>', unsafe_allow_html=True)


def _toggle(opp_id: str) -> None:
    st.session_state.setdefault("accepted", {})[opp_id] = st.session_state[f"accept-{opp_id}"]


def _projection(plan, mode: str) -> None:
    p = plan.projection
    g = p.goal
    with card("projection"):
        card_head("Current path vs your plan", "Cumulative savings over 12 months")
        if p.current_surplus is None:
            empty_card("Not enough history to project", "At least two months of data are needed.")
            return
        before = cumulative_path(p.current_surplus)
        after = cumulative_path(p.new_surplus)
        goal_line = p.remaining if g is not None and g.kind == BY_DATE and p.remaining else None
        label = (f"Goal {fmt.money(goal_line)}" if goal_line and not g.saved_so_far
                 else f"Goal: {fmt.money(goal_line)} to go" if goal_line else "")
        st.plotly_chart(charts.plan_paths(before, after, goal_line, label, mode), config=charts.CONFIG, theme=None,
                        key="plan-chart")
        parts = [f"6-month total: <b>{escape(fmt.money(p.cumulative_after[6]))}</b> "
                 f"({escape(fmt.money(p.cumulative_before[6]))} now)"]
        if g is not None and g.kind == BY_DATE and p.status != GOAL_MET:
            parts.append(f"Goal reached around month {p.months_to_goal_after} with the plan"
                         if p.months_to_goal_after is not None else "Goal not reached at this pace")
        st.markdown(f'<div class="fa-muted">{" &nbsp;·&nbsp; ".join(parts)}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="fa-calc">Method: {escape(p.method)}</div>', unsafe_allow_html=True)


def _phrased(db_path: str, key: str, provider_name: str) -> dict:
    return st.session_state.get("phrased", {}).get((db_path, key, provider_name), {})


def _phrase(db_path: str, key: str, provider_name: str) -> None:
    """Ask the selected model for one verified sentence per opportunity (plan.build_plan does the checking)."""
    from src.advisor import build_plan
    from src.llm import get_provider
    try:
        with st.spinner(f"Asking {provider_name} to phrase each suggestion..."):
            plan = build_plan(data.txns(db_path, key), goal=None, provider=get_provider(provider_name))
    except Exception as e:
        log.exception("phrasing failed")
        st.session_state["phrase_error"] = str(e).splitlines()[0] if str(e) else type(e).__name__
        st.toast(f"The {provider_name} model couldn't phrase the plan; using template sentences.")
        return
    st.session_state.setdefault("phrased", {})[(db_path, key, provider_name)] = {
        i.opportunity.id: (i.explanation, i.explanation_source) for i in plan.items}
