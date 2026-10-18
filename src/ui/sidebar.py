"""Sidebar: app name, data loading (sample DB or CSV upload) and the Privacy panel."""
from __future__ import annotations

import hashlib
import logging
import os
from html import escape

import streamlit as st
from dotenv import load_dotenv

from src.ui import data
from src.ui.components import muted, pill
from src.ui.theme import APP_NAME, AVAILABLE_PROVIDERS, PROVIDER_INFO, TAGLINE

log = logging.getLogger("finance_agent.ui")


def default_provider() -> str:
    load_dotenv()
    name = (os.getenv("LLM_PROVIDER") or "fake").strip().lower()
    return name if name in AVAILABLE_PROVIDERS else "fake"


def use_db(path, label: str) -> None:
    """Switch the app to another DB: reset everything that belongs to the old data."""
    st.session_state["db_path"] = str(path)
    st.session_state["db_label"] = label
    for k in ("month", "chat", "agent", "agent_sig", "focus", "accepted", "phrased"):
        st.session_state.pop(k, None)


def load_sample() -> None:
    try:
        use_db(data.ensure_sample_db(), "Sample data (synthetic)")
    except Exception:
        log.exception("sample load failed")
        st.session_state["sidebar_error"] = "Couldn't build the sample data. Check the logs and try again."


def render() -> str:
    """Draw the sidebar; returns the selected provider name."""
    st.session_state.setdefault("provider", default_provider())
    with st.sidebar:
        st.markdown(f'<div class="fa-header"><div class="name">{escape(APP_NAME)}</div>'
                    f'<div class="tag">{escape(TAGLINE)}</div></div>', unsafe_allow_html=True)
        st.divider()
        st.markdown("**Your data**")
        st.button("Load sample data", key="load-sample", type="primary", on_click=load_sample, width="stretch",
                  help="A synthetic 6-month statement, already categorized: loads instantly, no model call")
        upload = st.file_uploader("Upload a bank statement (CSV)", type=["csv"], key="upload",
                                  help="Columns: Date (YYYY-MM-DD or DD/MM/YYYY), Description, and Amount "
                                       "(or Debit + Credit)")
        if upload is not None:
            _handle_upload(upload)
        if err := st.session_state.pop("sidebar_error", None):
            st.error(err)
        if st.session_state.get("db_path"):
            muted(f"Using: {st.session_state.get('db_label', 'your data')}")
        st.divider()
        _privacy_panel()
    return st.session_state["provider"]


def _handle_upload(upload) -> None:
    content = upload.getvalue()
    digest = hashlib.sha256(content).hexdigest()
    if st.session_state.get("upload_digest") == digest:  # already processed this file
        return
    st.session_state["upload_digest"] = digest
    provider = st.session_state["provider"]
    bar = st.progress(0.0, text="Starting...")
    note = st.empty()

    def on_needs_llm(n: int) -> None:
        if n:
            note.info(f"{n} transaction{'s' if n != 1 else ''} didn't match a rule and will go to the "
                      f"{provider} model. On a local CPU model this can take a few minutes.")

    try:
        result = data.ingest_upload(content, upload.name, provider,
                                    step=lambda p, msg: bar.progress(p, text=msg), on_needs_llm=on_needs_llm)
    except data.UploadError as e:
        bar.empty()
        st.error(str(e))
        return
    except Exception:
        log.exception("upload failed")
        bar.empty()
        st.error("Something went wrong while processing this file. Please check it is a bank-statement CSV.")
        return
    bar.empty()
    use_db(result.db_path, f"{upload.name} ({result.rows_total} transactions)")
    if result.llm_error:
        st.warning(f"The {provider} model wasn't reachable ({result.llm_error.splitlines()[0]}). Rules categorized "
                   f"{result.by_rule} rows; the rest are marked Other: fix them in the Transactions table or retry "
                   "with another model.")
    else:
        st.success(f"Loaded {result.rows_total} transactions: {result.by_rule} by rules, {result.by_llm} by the model.")


def _privacy_panel() -> None:
    st.markdown("**Privacy**")
    st.selectbox("AI model", AVAILABLE_PROVIDERS, key="provider",
                 format_func=lambda n: f"{n} - {PROVIDER_INFO[n][2]}",
                 help="Used by Ask, for uncategorized upload rows, and for plan sentences. "
                      "The Dashboard never uses a model.")
    kind, label, _ = PROVIDER_INFO[st.session_state["provider"]]
    st.markdown(pill(label, "local" if kind == "Local" else "cloud"), unsafe_allow_html=True)
    muted("Names, emails, phone, card and account numbers are masked before any text reaches a model. "
          "Every figure is computed by code, never by the model.")
