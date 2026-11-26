"""Institutional dark quant terminal CSS theme for Streamlit."""

from __future__ import annotations

import streamlit as st

QUANT_TERMINAL_CSS = """
<style>
    /* Global institutional typography and dark slate surfaces */
    .stApp {
        background-color: #0b1120;
        color: #e2e8f0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
    }
    .block-container {
        padding-top: 1.35rem;
        padding-bottom: 2.5rem;
        max-width: 1440px;
    }
    /* Crisp metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, #111827 0%, #0f172a 100%);
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 12px 14px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.35);
    }
    div[data-testid="stMetric"] label {
        color: #94a3b8 !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    div[data-testid="stMetricValue"] {
        font-variant-numeric: tabular-nums;
        font-size: 1.45rem !important;
        font-weight: 700 !important;
        color: #f8fafc !important;
    }
    /* Horizontal pill workspace navigation bar */
    div[data-testid="stRadio"] > div[role="radiogroup"] {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        background: #0f172a;
        padding: 0.45rem 0.6rem;
        border-radius: 10px;
        border: 1px solid #1e293b;
        margin-bottom: 1rem;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] > label {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 0.4rem 0.85rem;
        cursor: pointer;
        transition: all 0.15s ease;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {
        border-color: #38bdf8;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"],
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) {
        background: #0284c7;
        border-color: #38bdf8;
        color: #ffffff !important;
    }
    /* Justification & status cards */
    .qa-card {
        background: #111827;
        border: 1px solid #1e293b;
        border-left: 3px solid #38bdf8;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.65rem;
    }
    .qa-card-warn {
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.35);
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.9rem;
    }
    .qa-card-ok {
        background: rgba(16, 185, 129, 0.09);
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-left: 4px solid #10b981;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.9rem;
    }
    .qa-card-title {
        font-size: 0.82rem;
        font-weight: 700;
        color: #38bdf8;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.25rem;
    }
    .qa-card-body {
        font-size: 0.92rem;
        color: #cbd5e1;
        line-height: 1.45;
    }
</style>
"""


def inject_quant_theme() -> None:
    """Inject custom institutional quant terminal styling cleanly into the active page."""
    st.markdown(QUANT_TERMINAL_CSS, unsafe_allow_html=True)
