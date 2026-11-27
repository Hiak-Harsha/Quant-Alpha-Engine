"""Modular, plain-English data formatters, justification engine, and watchlists."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd

from quant_alpha.paper.costs import DEFAULT_FEE_SCHEDULE, estimate_round_trip_breakeven_pct
from quant_alpha.portfolio.construct import is_cash_long_only_asset
from quant_alpha.predictor import LivePredictionReport

_FEATURE_LABELS: Dict[str, str] = {
    "ret_1": "1-Bar Return",
    "ret_2": "2-Bar Return",
    "ret_3": "3-Bar Return",
    "ret_5": "5-Bar Return",
    "ret_10": "10-Bar Return",
    "ret_21": "21-Bar Momentum",
    "ret_63": "63-Bar Trend",
    "rsi_14": "14-Bar RSI",
    "macd_hist": "MACD Histogram",
    "bb_zscore_20": "Bollinger 20D Z-Score",
    "atr_14_pct": "14D ATR %",
    "adx_14": "14D ADX Trend Strength",
    "rv_5": "5D Realized Vol",
    "rv_21": "21D Realized Vol",
    "gk_vol_21": "21D Garman-Klass Vol",
    "garch_vol_forecast": "GARCH(1,1) Forecast Vol",
    "vol_ratio_5_21": "Short/Medium Vol Ratio",
    "beta_market_63": "63D Market Beta",
    "idio_ret_5": "5D Idiosyncratic Return",
    "idio_ret_21": "21D Idiosyncratic Return",
    "corr21_nsei": "21D Correlation to NIFTY 50",
    "corr21_gold_fut": "21D Correlation to COMEX Gold",
    "corr21_silver_fut": "21D Correlation to COMEX Silver",
    "corr21_usdinr": "21D Correlation to USD/INR",
    "corr21_spy": "21D Correlation to S&P 500",
    "corr21_vix": "21D Correlation to VIX",
    "drv_usdinr_ret5": "USD/INR 5D FX Move",
    "drv_gold_fut_ret5": "COMEX Gold 5D Move",
    "drv_silver_fut_ret5": "COMEX Silver 5D Move",
    "drv_nsei_ret5": "NIFTY 50 5D Move",
    "sig_classical_trend": "Classical Trend Signal",
    "sig_classical_mean_rev": "Classical Mean-Reversion Signal",
    "sig_classical_momentum": "Classical Momentum Signal",
    "sig_macro_risk_on": "Macro Risk-On Composite",
    "cs_rank_ret1": "1D Peer Return Rank (0-1)",
    "cs_rank_ret5": "5D Peer Momentum Rank (0-1)",
    "cs_rank_ret21": "21D Peer Momentum Rank (0-1)",
    "cs_rel_ret5_diff": "5D Excess Return vs Peers",
    "cs_rel_ret21_diff": "21D Excess Return vs Peers",
    "cs_rank_rv21": "21D Peer Volatility Rank",
    "cs_rank_rsi14": "14D Peer RSI Rank",
    "cs_rank_amihud21": "21D Liquidity Rank",
    "cs_rank_idio21": "21D Idiosyncratic Alpha Rank",
}


def humanize_feature_name(feature: str) -> str:
    """Translate internal snake_case feature identifiers into crisp English labels."""
    return _FEATURE_LABELS.get(feature, feature.replace("_", " ").title())


def get_currency_symbol(ticker: str) -> str:
    """Return display currency symbol based on exchange suffix."""
    up = ticker.upper()
    if up.endswith((".NS", ".BO")) or up in {"^NSEI", "^NSEBANK", "^CNXIT", "^INDIAVIX", "USDINR=X"}:
        return "Rs "
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
        headline = "VALIDATED STATISTICAL EDGE (Live Portfolio Allocations Active)"
    else:
        headline = "RESEARCH / PAPER-TRADING MODE (0.0% Auto-Allocation — Edge Unproven vs. Buy & Hold)"

    detail = (
        f"OOS Meta-Ensemble: Sharpe = {sharpe:.3f} | Alpha = {alpha:+.2f}%/yr | "
        f"HAC t-stat = {hac_t:.2f} (requires >= 2.00) | DSR = {dsr:.3f} (requires >= 0.95) | Status: {verdict}"
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
        return "Research Only (Gated 0% Alloc)"
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
        lo_s = pd.to_numeric(df["range_low_pct"], errors="coerce").fillna(0.0)
        hi_s = pd.to_numeric(df["range_high_pct"], errors="coerce").fillna(0.0)
        df["Pred. Range Price"] = [
            f"{(c * (1.0 + lo / 100.0)):,.2f} - {(c * (1.0 + hi / 100.0)):,.2f}"
            for c, lo, hi in zip(close_s, lo_s, hi_s)
        ]
        df["Expected Range"] = [
            f"[{lo:+.2f}%, {hi:+.2f}%]"
            for lo, hi in zip(lo_s, hi_s)
        ]
    else:
        df["Pred. Range Price"] = ""
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
        "Pred. Range Price",
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
        "Est. Target",
        "Pred. Range Price",
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
        "Pred. Range Price",
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


def build_signal_justification(
    report: LivePredictionReport,
    position_capital: float = 100000.0,
) -> List[Dict[str, str]]:
    """Build 5 plain-English justification cards explaining WHY the prediction & gate status occurred."""
    cards: List[Dict[str, str]] = []
    cur = get_currency_symbol(report.ticker)

    # 1. Quantitative drivers & ensemble agreement
    top_drv_txt = "No local SHAP tree drivers computed for this call."
    if report.top_drivers:
        top3 = report.top_drivers[:3]
        parts = [
            f"{humanize_feature_name(str(d['feature']))} ({d['direction']}, impact {float(d['shap_impact']):+.3f})"
            for d in top3
        ]
        top_drv_txt = "Top model drivers: " + "; ".join(parts) + "."
    cards.append(
        {
            "title": "1. Quantitative Signal & Model Agreement",
            "body": (
                f"Pure Quant Forecast is **{report.quant_expected_ret_pct:+.2f}%** with **{report.prob_up_pct:.1f}%** "
                f"upside probability and **{report.model_agreement_pct:.0f}%** consensus across 6 model families "
                f"(raw classification: `{report.raw_verdict}`). {top_drv_txt}"
            ),
        }
    )

    # 2. Volatility, expected range & ATR risk bounds
    vol_v = float(report.garch_ann_vol_pct)
    vol_tier = "Low" if vol_v < 18.0 else ("Moderate" if vol_v < 30.0 else "Elevated / High Risk")
    cards.append(
        {
            "title": "2. Risk, GARCH Volatility & Expected Band",
            "body": (
                f"GARCH(1,1) annualized volatility is **{vol_v:.1f}% ({vol_tier})** vs. 21-bar realized vol "
                f"**{report.realized_vol21_pct:.1f}%**. Expected 1-sigma return range is "
                f"**[{report.expected_range_low_pct:+.2f}%, {report.expected_range_high_pct:+.2f}%]**. "
                f"ATR risk stop sits at **{cur}{report.stop_loss_price:,.2f}** and target at **{cur}{report.take_profit_price:,.2f}**."
            ),
        }
    )

    # 3. Market Regime & Cross-Asset Context
    beta_val = (report.cross_asset_snapshot or {}).get("beta_market_63")
    beta_str = f"63-day Market Beta = `{beta_val:.2f}`" if beta_val is not None else "Market Beta unavailable"
    vix_str = f"`{report.vix_level}`" if report.vix_level is not None else "`N/A`"
    cards.append(
        {
            "title": "3. HMM Market Regime & Macro Backdrop",
            "body": (
                f"Causal Market-Level HMM classifies current conditions as **{report.regime_label}** "
                f"({report.regime_probs}). Reference VIX = {vix_str}; {beta_str}."
            ),
        }
    )

    # 4. News Sentiment Overlay
    ns = report.news_summary or {}
    n_art = int(ns.get("asset_article_count", 0)) + int(ns.get("macro_article_count", 0))
    comp_sent = float(ns.get("composite_sentiment", 0.0))
    sent_word = "Positive" if comp_sent > 0.08 else ("Negative" if comp_sent < -0.08 else "Neutral")
    blend_note = (
        f"Sentiment overlay blended into composite return (**{report.composite_expected_ret_pct:+.2f}%**)."
        if n_art >= 2
        else "Fewer than 2 matching headlines found; composite forecast equals pure quant score."
    )
    cards.append(
        {
            "title": "4. Live News & Narrative Context",
            "body": f"Detected **{n_art}** recent asset/macro headlines with score **{comp_sent:+.3f} ({sent_word})**. {blend_note}",
        }
    )

    # 5. Execution Rules, Real Broker Cost & PassGate Reliability
    be_pct = estimate_round_trip_breakeven_pct(report.ticker, max(1000.0, position_capital), DEFAULT_FEE_SCHEDULE)
    cash_rule = (
        "NSE/BSE Cash Delivery is strictly **Long/Flat** (delivery shorting disabled)."
        if is_cash_long_only_asset(report.ticker)
        else "Both Long and Short directional trades are supported for this instrument."
    )
    cards.append(
        {
            "title": "5. Real-World Fees & Statistical Reliability Gate",
            "body": (
                f"Estimated round-trip delivery breakeven on {cur}{position_capital:,.0f} (including STT, Stamp, "
                f"NSE Txn, GST & Rs 15.34 DP sell charge where applicable) is **{be_pct:.3f}%** vs. expected move "
                f"**{report.composite_expected_ret_pct:+.2f}%**. {cash_rule} **Gate Status:** `{report.verdict}` "
                f"({report.gate_reason})."
            ),
        }
    )
    return cards
