"""Modular UI view-model formatters and presentation helpers for the Streamlit dashboard."""

from quant_alpha.ui.components import (
    build_metals_spotlight_df,
    build_screener_display_df,
    build_upside_downside_watchlists,
    compute_capital_scenario,
    format_pulse_value,
    get_currency_symbol,
    summarize_gate_status,
)

__all__ = [
    "build_metals_spotlight_df",
    "build_screener_display_df",
    "build_upside_downside_watchlists",
    "compute_capital_scenario",
    "format_pulse_value",
    "get_currency_symbol",
    "summarize_gate_status",
]
