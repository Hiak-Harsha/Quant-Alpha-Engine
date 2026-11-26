"""Modular UI view-model formatters, styling, and plain-English justification helpers."""

from quant_alpha.ui.components import (
    build_metals_spotlight_df,
    build_screener_display_df,
    build_signal_justification,
    build_upside_downside_watchlists,
    compute_capital_scenario,
    format_pulse_value,
    get_currency_symbol,
    humanize_feature_name,
    summarize_gate_status,
)
from quant_alpha.ui.styles import inject_quant_theme

__all__ = [
    "build_metals_spotlight_df",
    "build_screener_display_df",
    "build_signal_justification",
    "build_upside_downside_watchlists",
    "compute_capital_scenario",
    "format_pulse_value",
    "get_currency_symbol",
    "humanize_feature_name",
    "inject_quant_theme",
    "summarize_gate_status",
]
