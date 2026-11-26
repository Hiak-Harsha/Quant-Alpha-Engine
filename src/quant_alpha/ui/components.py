"""Modular, plain-English data formatters and watchlists for Quant Alpha Engine UI."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import pandas as pd

from quant_alpha.predictor import LivePredictionReport


def get_currency_symbol(ticker: str) -> str:
    """Return display currency symbol based on exchange suffix."""
    up = ticker.upper()
    if up.endswith((".NS", ".BO")) or up in {"^NSEI", "^NSEBANK", "^CNXIT", "^INDIAVIX"}:
        return "INR "
    if up == "USDINR=X":
        return "INR "
    return "$"


def format_pulse_value(ticker: str, value: float) -> str:
    """Format market pulse numbers without Streamlit metric ellipsis truncation."""
    up = ticker.upper()
    if up in {"^NSEI", "^NSEBANK", "^CNXIT"}:
        return f"{value:,.0f}"
    if up in {"^INDIAVIX", "^VIX", "USDINR=X"}:
        return f"{value:.2f}"
    prefix = "Rs " if up.endswith((".NS", ".BO")) else "$"
    return f"{prefix}{value:,.2f}"


def summarize_gate_status(track_record: Dict[str, Any]) -> Dict[str, Any]:
    """Summarize statistical edge gate state and underlying backtest numbers."""
    passed = bool(track_record.get("gate_passed", False))
    alpha = float(track_record.get("alpha_ann_pct", 0.0))
    hac_t = float(track_record.get("hac_t_stat", 0.0))
    dsr = float(track_record.get("dsr_prob", 0.0))
    sharpe = float(track_record.get("sharpe", 0.0))
    verdict = str(track_record.get("edge_verdict", "No Edge vs Benchmark"))

    if passed:
        headline = "VALIDATED EDGE (Live Portfolio Allocations Active)"
    else:
        headline = "RESEARCH / GATED MODE (0.0% Live Allocation — Unproven Alpha vs. Buy & Hold)"

    detail = (
        f"OOS Meta-Ensemble: Sharpe={sharpe:.3f} | Alpha={alpha:+.2f}%/yr | "
        f"HAC t={hac_t:.2f} (req >= 2.00) | DSR={dsr:.3f} (req >= 0.95) | Status: {verdict}"
    )
    return {
        "passed": passed,
        "headline": headline,
        "detail": detail,
        "alpha_ann_pct": alpha,
        "hac_t_stat": hac_t,
        "dsr_prob": dsr,
        "sharpe": sharpe,
    }


def _execution_rule_label(row: pd.Series) -> str:
    verdict = str(row.get("verdict", ""))
    raw_v = str(row.get("raw_verdict", ""))
    comp_ret = float(row.get("composite_ret_pct", 0.0))
    cash_only = bool(row.get("cash_long_only", False))
    gated = verdict == "NO EDGE / GATED"

    if cash_only and comp_ret < 0.0:
        return "Avoid / Exit Risk (Cash Long-Only)"
    if gated:
        return "Gated Research Only (0% Alloc)"
    if raw_v == "NEUTRAL / HOLD":
        return "Hold / Below Cost Hurdle"
    return "Active Allocation"


def _prepare_enriched_scan(scan_df: pd.DataFrame) -> pd.DataFrame:
    """Enrich raw scan DataFrame with projected price, 1-sigma range text, and execution rule."""
    if scan_df.empty:
        return scan_df.copy()

    df = scan_df.copy()
    close_s = pd.to_numeric(df["close"], errors="coerce").fillna(0.0)
    ret_s = pd.to_numeric(df["composite_ret_pct"], errors="coerce").fillna(0.0)
    df["Est. Target"] = (close_s * (1.0 + ret_s / 100.0)).round(2)

    if "range_low_pct" in df.columns and "range_high_pct" in df.columns:
        df["Expected Range"] = [
            f"[{lo:+.2f}%, {hi:+.2f}%]"
            for lo, hi in zip(
                pd.to_numeric(df["range_low_pct"], errors="coerce").fillna(0.0),
                pd.to_numeric(df["range_high_pct"], errors="coerce").fillna(0.0),
            )
        ]
    else:
        df["Expected Range"] = ""

    if "as_of" not in df.columns:
        df["as_of"] = ""

    df["Execution Note"] = df.apply(_execution_rule_label, axis=1)
    return df


def build_upside_downside_watchlists(
    scan_df: pd.DataFrame,
    top_n: int = 12,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split scan into Potential Profit (Upside) and Potential Loss / Risk (Downside) watchlists."""
    if scan_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = _prepare_enriched_scan(scan_df)
    cols = [
        "ticker",
        "market",
        "as_of",
        "close",
        "chg_1_pct",
        "raw_verdict",
        "composite_ret_pct",
        "Est. Target",
        "Expected Range",
        "prob_up_pct",
        "consensus_pct",
        "garch_vol_pct",
        "stop_loss",
        "take_profit",
        "alloc_pct",
        "verdict",
        "Execution Note",
    ]
    avail_cols = [c for c in cols if c in df.columns]

    rename_map = {
        "ticker": "Ticker",
        "market": "Segment",
        "as_of": "Bar Date",
        "close": "Last Close",
        "chg_1_pct": "1-Bar Chg %",
        "raw_verdict": "Model View",
        "composite_ret_pct": "Est. Move %",
        "prob_up_pct": "Up Prob %",
        "consensus_pct": "Model Agree %",
        "garch_vol_pct": "Ann Vol %",
        "stop_loss": "Risk Stop",
        "take_profit": "Target Price",
        "alloc_pct": "Live Alloc %",
        "verdict": "Gate Status",
    }

    up_mask = pd.to_numeric(df["composite_ret_pct"], errors="coerce").fillna(0.0) > 0.0
    gainers = (
        df.loc[up_mask]
        .assign(_score=lambda x: x["composite_ret_pct"] * (x["consensus_pct"] / 100.0))
        .sort_values(["_score", "prob_up_pct"], ascending=False)
        .head(top_n)[avail_cols]
        .rename(columns=rename_map)
        .reset_index(drop=True)
    )

    down_mask = pd.to_numeric(df["composite_ret_pct"], errors="coerce").fillna(0.0) < 0.0
    decliners = (
        df.loc[down_mask]
        .sort_values(["composite_ret_pct", "prob_up_pct"], ascending=[True, True])
        .head(top_n)[avail_cols]
        .rename(columns=rename_map)
        .reset_index(drop=True)
    )
    return gainers, decliners


def build_metals_spotlight_df(scan_df: pd.DataFrame) -> pd.DataFrame:
    """Return formatted Gold & Silver ETF + Bullion spotlight DataFrame."""
    if scan_df.empty:
        return pd.DataFrame()
    df = _prepare_enriched_scan(scan_df)
    metals = df[df["market"] == "Gold / Silver ETF"].copy()
    if metals.empty:
        return pd.DataFrame()
    cols = [
        "ticker",
        "as_of",
        "close",
        "chg_1_pct",
        "raw_verdict",
        "composite_ret_pct",
        "Expected Range",
        "prob_up_pct",
        "consensus_pct",
        "garch_vol_pct",
        "alloc_pct",
        "verdict",
        "Execution Note",
    ]
    avail = [c for c in cols if c in metals.columns]
    return metals[avail].rename(
        columns={
            "ticker": "Ticker",
            "as_of": "Bar Date",
            "close": "Last Close",
            "chg_1_pct": "1-Bar Chg %",
            "raw_verdict": "Model View",
            "composite_ret_pct": "Est. Move %",
            "prob_up_pct": "Up Prob %",
            "consensus_pct": "Model Agree %",
            "garch_vol_pct": "Ann Vol %",
            "alloc_pct": "Live Alloc %",
            "verdict": "Gate Status",
        }
    ).reset_index(drop=True)


def build_screener_display_df(scan_df: pd.DataFrame) -> pd.DataFrame:
    """Return full formatted table with plain-English headers for filtering and exploration."""
    if scan_df.empty:
        return pd.DataFrame()
    df = _prepare_enriched_scan(scan_df)
    cols = [
        "ticker",
        "market",
        "as_of",
        "close",
        "chg_1_pct",
        "raw_verdict",
        "composite_ret_pct",
        "Est. Target",
        "Expected Range",
        "prob_up_pct",
        "consensus_pct",
        "garch_vol_pct",
        "regime",
        "buzz_score",
        "alloc_pct",
        "stop_loss",
        "take_profit",
        "verdict",
        "Execution Note",
    ]
    avail = [c for c in cols if c in df.columns]
    return df[avail].rename(
        columns={
            "ticker": "Ticker",
            "market": "Segment",
            "as_of": "Bar Date",
            "close": "Last Close",
            "chg_1_pct": "1-Bar Chg %",
            "raw_verdict": "Model View",
            "composite_ret_pct": "Est. Move %",
            "prob_up_pct": "Up Prob %",
            "consensus_pct": "Model Agree %",
            "garch_vol_pct": "Ann Vol %",
            "regime": "HMM Regime",
            "buzz_score": "News Buzz",
            "alloc_pct": "Live Alloc %",
            "stop_loss": "Risk Stop",
            "take_profit": "Target Price",
            "verdict": "Gate Status",
        }
    ).reset_index(drop=True)


def compute_capital_scenario(report: LivePredictionReport, capital: float) -> Dict[str, float]:
    """Compute objective expected P&L, 1-sigma outcome range, and ATR stop/target amounts."""
    cap = max(0.0, float(capital))
    last_c = max(float(report.last_close), 1e-8)

    expected_pnl = cap * (float(report.composite_expected_ret_pct) / 100.0)
    range_low_pnl = cap * (float(report.expected_range_low_pct) / 100.0)
    range_high_pnl = cap * (float(report.expected_range_high_pct) / 100.0)

    stop_dist_pct = abs(float(report.stop_loss_price) - last_c) / last_c
    tp_dist_pct = abs(float(report.take_profit_price) - last_c) / last_c
    stop_loss_risk = -cap * stop_dist_pct
    take_profit_gain = cap * tp_dist_pct
    gated_capital = cap * (abs(float(report.suggested_allocation_pct)) / 100.0)

    return {
        "capital": round(cap, 2),
        "expected_pnl": round(expected_pnl, 2),
        "range_low_pnl": round(range_low_pnl, 2),
        "range_high_pnl": round(range_high_pnl, 2),
        "stop_loss_risk": round(stop_loss_risk, 2),
        "take_profit_gain": round(take_profit_gain, 2),
        "gated_capital": round(gated_capital, 2),
    }
