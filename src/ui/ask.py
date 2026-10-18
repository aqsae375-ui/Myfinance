"""Ask tab: chat with the Stage 4 agent at full width. Under every answer a collapsed "How I got this"
expander (labelled with the answer's verification state) shows that answer's own trace: tool called,
result returned by code, number check. Older answers keep their traces and can be opened too.

Suggested questions are generated from the loaded statement (its real months, categories and
subscriptions), never fixed text. The agent and provider are created lazily, on the first question."""
from __future__ import annotations

import json
import logging
import re
from html import escape

import streamlit as st

from src.ui import data
from src.ui import format as fmt
from src.ui.components import badge, card, card_head, muted, pill

log = logging.getLogger("finance_agent.ui")

MAX_SOURCE_ROWS = 50
MAX_FIELDS = 8
MAX_SUGGESTIONS = 4
LOCAL_SLOW = frozenset({"ollama"})
EXCLUDED_FROM_SUGGESTIONS = frozenset({"Transfers", "Transfers In", "Income", "Interest", "Other", "Uncategorized",
                                       "Rent", "Tax", "Bank Fees"})  # fixed or non-spending: dull questions
MONEY_RE = re.compile(rf"(-?{re.escape(fmt.CURRENCY)}\s?[\d,]+(?:\.\d+)?)")
HINT = ("Every answer has a \"How I got this\" link underneath: which tool ran, what the code returned, and how "
        "every number in the reply was checked.")


def _md(text: str) -> str:
    """User/model text as inert HTML: escaped, with amounts in bold and line breaks kept."""
    return MONEY_RE.sub(r"<b>\1</b>", escape(text, quote=False)).replace("\n", "<br>")


def _ask_suggestion(question: str) -> None:
    st.session_state["pending_question"] = question


def suggestions(db_path: str, key: str) -> list[str]:
    """Questions about this statement: its latest month, its biggest spending categories, its most
    expensive month and (only if any are found) its subscriptions."""
    history = data.history(db_path, key)
    if not history:
        return []
    one_year = len({r.month[:4] for r in history}) == 1
    name = fmt.month_name if one_year else fmt.month_label
    latest = history[-1].month
    cats = [c.category for c in data.month_view(db_path, key, latest)["mom"].current.categories
            if c.category not in EXCLUDED_FROM_SUGGESTIONS]
    out = []
    if cats:
        out.append(f"How much did I spend on {cats[0].lower()} in {name(latest)}?")
    if len(history) >= 2:
        prev = history[-2].month
        out.append(f"Compare {name(prev)} with {name(latest)}")
        if len(cats) > 1:
            out.append(f"Compare {cats[1].lower()} in {name(prev)} and {name(latest)}")
    subs = data.subscriptions(db_path, key)
    if any(s.possibly_unused for s in subs):
        out.append("Which subscriptions might be unused?")
    elif subs:
        out.append("Which subscriptions am I paying for?")
    peak = max(history, key=lambda r: r.total_spent)
    out.append(f"Show my biggest transactions in {name(peak.month)}")
    return out[:MAX_SUGGESTIONS]


def _agent(db_path: str, key: str, provider_name: str):
    """One FinanceAgent per (data, provider); rebuilt when either changes, which also clears its memory."""
    sig = (db_path, key, provider_name)
    if st.session_state.get("agent_sig") != sig:
        from src.agent.chat import FinanceAgent
        from src.llm import get_provider
        st.session_state["agent"] = FinanceAgent(get_provider(provider_name), txns=data.txns(db_path, key),
                                                      cash=data.cash_entries(db_path, key))
        st.session_state["agent_sig"] = sig
    return st.session_state["agent"]


def render(db_path: str, key: str, provider_name: str) -> None:
    chat: list[dict] = st.session_state.setdefault("chat", [])
    with card("ask"):
        card_head("Ask about your spending",
                  "Every number in an answer comes from a tool call. If it cannot be proven, it is not shown.")
        for i, turn in enumerate(chat):
            _render_turn(turn, i, db_path, key)
        pending = st.empty()
        qs = suggestions(db_path, key)
        if qs:
            cols = st.columns(2)
            for i, q in enumerate(qs):
                cols[i % 2].button(q, key=f"suggest-{i}", on_click=_ask_suggestion, args=(q,), width="stretch")
        with st.form("ask-form", clear_on_submit=True, border=False):
            box, send = st.columns([5, 1], vertical_alignment="bottom")
            typed = box.text_input("Question", key="ask-input", placeholder="Ask a question about your statement",
                                   label_visibility="collapsed")
            submitted = send.form_submit_button("Send", width="stretch")
        muted(HINT if not chat else "Instruction-like text inside a transaction is flagged with a warning chip "
                                      "and never followed.")
        if provider_name in LOCAL_SLOW:
            muted("Local models can take 1 to 2 minutes per answer on a laptop CPU.")
        question = (typed.strip() if submitted and typed else None) or st.session_state.pop("pending_question", None)
        if question:
            with pending.container(), st.spinner("Thinking..."):
                _answer(question, chat, db_path, key, provider_name)
            st.rerun()


def _answer(question: str, chat: list[dict], db_path: str, key: str, provider_name: str) -> None:
    try:
        reply = _agent(db_path, key, provider_name).ask(question)
        chat.append({"question": question, "answer": reply.text, "trace": reply.trace})
    except Exception as e:  # provider misconfigured or unreachable: say so, never a traceback
        log.exception("agent failed")
        chat.append({"question": question, "answer": None,
                     "error": f"The {provider_name} model couldn't answer: "
                              f"{str(e).splitlines()[0] if str(e) else type(e).__name__}. "
                              "Check the model is running, or switch to another one in the sidebar."})


def trace_label(trace) -> str:
    """The small line under an answer: its verification state + "How I got this"."""
    if trace.refused:
        state = "Refused (out of scope)"
    elif trace.fallback_used:
        return ":orange[⚠ Verification failed: answer rebuilt from tool results · How I got this]"
    elif trace.retried:
        state = "✓ Verified after one retry"
    else:
        state = "✓ Verified"
    return f"{state} · How I got this"


def _render_turn(turn: dict, i: int, db_path: str, key: str) -> None:
    st.markdown(f'<div class="fa-msg user"><div class="fa-bubble">{_md(turn["question"])}</div></div>',
                unsafe_allow_html=True)
    if turn.get("error"):
        st.warning(turn["error"])
        return
    trace = turn["trace"]
    badges = "".join(_badges(trace))
    st.markdown(f'<div class="fa-msg bot"><div class="fa-bubble">{_md(turn["answer"])}'
                f'{f"<div class=badges>{badges}</div>" if badges else ""}</div></div>', unsafe_allow_html=True)
    with st.expander(trace_label(trace), expanded=False):
        _how_i_got_this(trace, i, db_path, key)


def _badges(trace) -> list[str]:
    """Chips kept on the answer itself: source count and any prompt-injection warning (the verification
    state lives in the expander label)."""
    out = []
    n_src = len({i for c in trace.tool_calls for i in c.txn_ids})
    if n_src:
        out.append(badge(f"{n_src} source transaction{'s' if n_src != 1 else ''}"))
    if trace.injection_flags:
        out.append(pill(f"⚠ Prompt-injection text flagged ({len(trace.injection_flags)})", "warn"))
    return out


def _call_text(call) -> str:
    if not call.args:
        return f"{call.name}()"
    args = ",\n".join(f"  {k}={json.dumps(v, ensure_ascii=False)}" for k, v in call.args.items())
    return f"{call.name}(\n{args})"


def _fields(result: dict) -> list[tuple[str, str]]:
    """Numeric fields of a tool result, as the code returned them (top level, then one level down)."""
    out = []
    for k, v in result.items():
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            out.append((k, f"{v:,.2f}" if isinstance(v, float) else f"{v:,}"))
        elif isinstance(v, dict):
            out += [(f"{k}.{k2}", f"{v2:,.2f}" if isinstance(v2, float) else f"{v2:,}")
                    for k2, v2 in v.items() if isinstance(v2, (int, float)) and not isinstance(v2, bool)]
        elif isinstance(v, list):
            out.append((f"{k} (rows)", f"{len(v):,}"))
    return out[:MAX_FIELDS]


def _step(n: int, title: str, body: str) -> str:
    return f'<div class="fa-step"><div class="h"><span class="n">{n}</span>{escape(title)}</div>{body}</div>'


def _how_i_got_this(trace, i: int, db_path: str, key: str) -> None:
    if trace.refused:
        st.markdown(_step(1, "Refused by a fixed rule", '<div class="fa-muted">The question was outside what this '
                                                         'assistant covers: no model or tool was called.</div>'),
                    unsafe_allow_html=True)
        return
    calls = trace.tool_calls
    if not calls:
        st.markdown(_step(1, "Tool called", '<div class="fa-muted">No tool was called for this answer.</div>'),
                    unsafe_allow_html=True)
    else:
        code = "".join(f'<div class="fa-code">{escape(_call_text(c))}</div>' for c in calls)
        st.markdown(_step(1, "Tool called" if len(calls) == 1 else f"Tools called ({len(calls)})", code),
                    unsafe_allow_html=True)
        fields = [f for c in calls for f in _fields(c.result)][:MAX_FIELDS]
        grid = "".join(f'<div class="fa-field"><div class="k">{escape(k)}</div><div class="v">{escape(v)}</div></div>'
                       for k, v in fields) or '<div class="fa-muted">The result has no numeric fields.</div>'
        st.markdown(_step(2, "Result returned by code", f'<div class="fa-fields">{grid}</div>'),
                    unsafe_allow_html=True)
    from src.agent.verify import extract_numbers
    dates, numbers = extract_numbers(trace.answer)
    ok = " ".join(f"{escape(n)} ✓" for n in [*numbers, *dates]) or "No numbers in the reply."
    bad = "".join(f' <span class="no">{escape(n)} ✗</span>' for n in trace.unverified_numbers)
    note = ("Every number in the reply was matched to a field above. A number that does not match replaces the "
            "whole answer.")
    if trace.fallback_used:
        note = ("The model's draft had numbers no tool returned (✗), so it was replaced by a summary built "
                "directly from the tool results.")
    st.markdown(_step(3, "Number check", f'<div class="fa-muted">{note}</div><div class="fa-checks">{ok}{bad}</div>'),
                unsafe_allow_html=True)
    for flag in trace.injection_flags:
        st.markdown(pill("⚠ " + flag, "warn"), unsafe_allow_html=True)
    if trace.retried:
        muted("The model's first draft had numbers but no tool call, so it was asked once to use the tools.")
    if calls:  # expanders can't nest: full results are collapsed JSON instead
        st.caption("Full tool results and source rows")
        df = data.txns(db_path, key)
        for n, call in enumerate(calls, 1):
            st.caption(f"{n}. {call.name} · source: {call.source or '-'} · {call.duration_ms:.0f} ms")
            st.json(call.result, expanded=False)
            if call.txn_ids:
                rows = df[df["id"].isin(call.txn_ids)].head(MAX_SOURCE_ROWS)
                st.dataframe({"Date": [fmt.day_label(d.date()) for d in rows["date"]],
                              "Description (redacted)": list(rows["redacted_description"]),
                              "Category": list(rows["category"]),
                              "Amount": [fmt.money(a, cents=True) for a in rows["amount"]]},
                             hide_index=True, key=f"src-{i}-{n}")
