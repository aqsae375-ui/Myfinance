"""Streamlit entry point: header, sidebar and the three views Dashboard | Ask | Your Plan.

Run:  streamlit run app.py

The views are switched with a segmented control rather than st.tabs so that (a) only the active view
runs, keeping the Dashboard free of any model code, and (b) "Fix this" on an insight can open Your Plan.
"""
from __future__ import annotations

import logging
import streamlit as st

from src.ui import dashboard, data, sidebar
from src.ui import format as fmt
from src.ui.components import brand_html, empty_state, footer, inject_css, privacy_badge, theme_mode
from src.ui.theme import APP_NAME, PROVIDER_INFO, TAGLINE

log = logging.getLogger("finance_agent.ui")
VIEWS = ("Dashboard", "Ask", "Your Plan")

st.set_page_config(page_title=APP_NAME, page_icon="💹", layout="wide", initial_sidebar_state="expanded")
mode = theme_mode()
inject_css(mode)
provider = sidebar.render()
db_path = st.session_state.get("db_path")
months = data.months(db_path, data.db_key(db_path)) if db_path else []

# ---- header bar: logo + name + tagline | privacy badge | month selector (months that exist in the data) ----
with st.container(key="topbar"):
    left, mid, right = st.columns([5, 2.6, 1.9], vertical_alignment="center")
    left.markdown(brand_html(APP_NAME, TAGLINE), unsafe_allow_html=True)
    kind, label, _ = PROVIDER_INFO[provider]
    mid.markdown(f'<div style="text-align:right">{privacy_badge(kind, label)}</div>', unsafe_allow_html=True)
    if months:
        if st.session_state.get("month") not in months:
            st.session_state["month"] = months[-1]
        right.selectbox("Month", months, key="month", format_func=fmt.month_label, label_visibility="collapsed")

st.session_state.setdefault("nav", "Dashboard")
st.segmented_control("View", VIEWS, key="nav", required=True, label_visibility="collapsed")
view = st.session_state.get("nav") or "Dashboard"

if not db_path:
    empty_state("Upload a statement or load sample data",
                "Use the sidebar: upload a bank-statement CSV, or click \"Load sample data\" to explore a demo.")
elif not months:
    empty_state("No transactions yet", "This statement has no rows. Upload another CSV or load the sample data.")
else:
    key = data.db_key(db_path)
    try:
        if view == "Dashboard":
            dashboard.render(db_path, key, st.session_state["month"], mode)
            footer()
        elif view == "Ask":
            from src.ui import ask  # imported only when the Ask view runs: the Dashboard never loads model code
            ask.render(db_path, key, provider)
        else:
            from src.ui import plan
            plan.render(db_path, key, provider, mode)
    except Exception:  # st.rerun/st.stop raise BaseException subclasses, so they pass through
        log.exception("view %s failed", view)
        st.error(f"Something went wrong while building the {view} view. Try reloading the data from the sidebar.")
