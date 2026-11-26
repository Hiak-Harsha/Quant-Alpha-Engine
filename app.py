"""Fast, modular Institutional Quant Terminal & Paper-Trading Desk (Streamlit)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import pandas as pd
import plotly.express as px
import streamlit as st

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quant_alpha.backtest.engine import run_backtest_comparison
from quant_alpha.config import (
    DATA_CACHE_DIR,
    GLOBAL_DRIVERS,
    HORIZON_CONFIGS,
    UNIVERSES,
    EngineSettings,
    configure_logging,
    ensure_directories,
)
from quant_alpha.data.macro_feed import load_all_macro_series
from quant_alpha.data.market_feed import fetch_ohlcv, fetch_universe_ohlcv
from quant_alpha.data.news_feed import fetch_all_news
from quant_alpha.data.trending_feed import discover_trending_stocks
from quant_alpha.data.validator import run_full_feed_audit
from quant_alpha.features.sentiment import score_news_dataframe
from quant_alpha.models.ensemble import load_ensemble_artifacts
from quant_alpha.paper.costs import DEFAULT_FEE_SCHEDULE
from quant_alpha.paper.ledger import PaperLedger, log_paper_signals
from quant_alpha.paper.simulator import PaperBrokerSimulator, compute_prediction_market_correlation
from quant_alpha.predictor import (
    LivePredictionReport,
    ensure_horizon_bundle,
    load_scan_snapshot,
    predict_asset,
    predict_multi_horizon_matrix,
    scan_market_predictions,
)
from quant_alpha.ui import (
    build_metals_spotlight_df,
    build_screener_display_df,
    build_signal_justification,
    build_upside_downside_watchlists,
    compute_capital_scenario,
    format_pulse_value,
    get_currency_symbol,
    humanize_feature_name,
    inject_quant_theme,
    summarize_gate_status,
)

st.set_page_config(
    page_title="Quant Alpha Terminal | Research & Paper Desk",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(ttl=900, show_spinner=False)
def _cached_market_bundle(
    universe_name: str,
    horizon_name: str,
    include_trending: bool,
    max_trending: int,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and cache universe + driver market data independently of single-asset selection."""
    hcfg = HORIZON_CONFIGS[horizon_name]
    base_tickers = list(UNIVERSES[universe_name])
    india_focus = universe_name.startswith("india_")
    macro_df = load_all_macro_series()
    news_tickers = list(dict.fromkeys(base_tickers[:35]))
    news_df = score_news_dataframe(fetch_all_news(news_tickers))
    trending_df = discover_trending_stocks(
        news_df=news_df,
        candidate_tickers=base_tickers,
        india_focus=india_focus,
    )

    scan_tickers = list(base_tickers)
    if include_trending and not trending_df.empty:
        cands = [s for s in trending_df["symbol"].tolist() if s not in scan_tickers]
        if india_focus:
            cands = [
                s for s in cands
                if s.endswith((".NS", ".BO"))
                or s in {"GOLDBEES.NS", "SILVERBEES.NS", "GC=F", "SI=F", "GLD", "SLV", "USDINR=X", "^NSEI", "^NSEBANK"}
            ]
        scan_tickers.extend(cands[:max_trending])

    tickers = list(dict.fromkeys(scan_tickers + GLOBAL_DRIVERS))
    ohlcv_map, _ = fetch_universe_ohlcv(tickers, period=hcfg.period, interval=hcfg.interval, max_workers=10)
    return ohlcv_map, macro_df, news_df, trending_df


@st.cache_resource(show_spinner=False)
def _cached_ensemble_bundle(horizon_name: str) -> dict:
    """Keep large trained model bundle pinned in memory across clicks."""
    return load_ensemble_artifacts(horizon_name)


@st.cache_data(ttl=1800, show_spinner=False)
def _cached_market_scan(
    horizon_name: str,
    universe_name: str,
    include_trending: bool,
    max_trending: int,
    _ohlcv_map: dict,
    _macro_df: pd.DataFrame,
    _scored_news: pd.DataFrame,
    _bundle: dict,
) -> tuple[pd.DataFrame, dict, dict]:
    ttl = 0.5 if horizon_name == "intraday" else 6.0
    snap = load_scan_snapshot(horizon_name, universe_name, ttl_hours=ttl)
    if snap is not None:
        return snap

    out = scan_market_predictions(
        horizon_name=horizon_name,
        universe_name=universe_name,
        include_trending=include_trending,
        max_trending_extras=max_trending,
        ohlcv_map=_ohlcv_map,
        macro_df=_macro_df,
        scored_news=_scored_news,
        bundle=_bundle,
    )
    return out.scan_df, out.skipped_tickers, out.horizon_track_record


@st.cache_data(ttl=900, show_spinner=False)
def _cached_single_asset_report(
    ticker: str,
    horizon_name: str,
    universe_name: str,
    latest_bar_key: str,
    _ohlcv_map: dict,
    _macro_df: pd.DataFrame,
    _scored_news: pd.DataFrame,
    _bundle: dict,
) -> LivePredictionReport:
    """Cache SHAP-enriched single-asset forecast so clicking back and forth is instantaneous."""
    return predict_asset(
        ticker=ticker,
        horizon_name=horizon_name,
        universe_name=universe_name,
        ohlcv_map=_ohlcv_map,
        macro_df=_macro_df,
        scored_news=_scored_news,
        bundle=_bundle,
        compute_shap=True,
    )


@st.cache_data(ttl=1800, show_spinner=False)
def _cached_backtest(
    horizon_name: str,
    _oof_df: pd.DataFrame,
    model_names: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from quant_alpha.config import ARTIFACTS_DIR

    hdir = ARTIFACTS_DIR / horizon_name
    lb_p = hdir / "leaderboard.parquet"
    eq_p = hdir / "equity_curves.parquet"
    dd_p = hdir / "drawdown_curves.parquet"
    as_p = hdir / "per_asset_summary.parquet"
    if lb_p.exists() and eq_p.exists() and dd_p.exists() and as_p.exists():
        lb_cached = pd.read_parquet(lb_p)
        as_cached = pd.read_parquet(as_p)
        if (
            {"alpha_ann_pct", "hac_t_stat", "dsr_prob", "gate_passed"}.issubset(lb_cached.columns)
            and {"hit_edge_pct", "gate_passed"}.issubset(as_cached.columns)
        ):
            return lb_cached, pd.read_parquet(eq_p), pd.read_parquet(dd_p), as_cached

    hcfg = HORIZON_CONFIGS[horizon_name]
    bt = run_backtest_comparison(_oof_df, list(model_names), hcfg, EngineSettings(bootstrap_samples=150))
    try:
        hdir.mkdir(parents=True, exist_ok=True)
        bt.leaderboard.to_parquet(lb_p, index=False)
        bt.equity_curves.to_parquet(eq_p)
        bt.drawdown_curves.to_parquet(dd_p)
        bt.per_asset_summary.to_parquet(as_p, index=False)
    except Exception:
        pass
    return bt.leaderboard, bt.equity_curves, bt.drawdown_curves, bt.per_asset_summary


def _nse_market_status_str() -> str:
    now_ist = pd.Timestamp.now(tz="Asia/Kolkata")
    is_weekday = now_ist.day_of_week < 5
    in_hours = (now_ist.hour > 9 or (now_ist.hour == 9 and now_ist.minute >= 15)) and (
        now_ist.hour < 15 or (now_ist.hour == 15 and now_ist.minute <= 30)
    )
    status = "NSE OPEN (09:15–15:30 IST)" if (is_weekday and in_hours) else "NSE CLOSED / AFTER-HOURS"
    return f"{now_ist.strftime('%Y-%m-%d %H:%M IST')} | {status}"


def _render_gate_banner(track_record: dict) -> dict:
    gate_info = summarize_gate_status(track_record)
    if gate_info["passed"]:
        st.markdown(
            f'<div class="qa-card-ok"><strong>{gate_info["headline"]}</strong><br>'
            f'<span style="color:#cbd5e1;font-size:0.88rem;">{gate_info["detail"]}</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="qa-card-warn"><strong>{gate_info["headline"]}</strong><br>'
            f'<span style="color:#cbd5e1;font-size:0.88rem;">{gate_info["detail"]}. '
            f'Raw closed-bar forecasts are shown below strictly for research &amp; paper-trading validation (live auto-allocation locked at 0.0%).</span></div>',
            unsafe_allow_html=True,
        )
    return gate_info


def _render_market_radar(
    horizon_name: str,
    universe_name: str,
    include_trending: bool,
    max_trending: int,
    ohlcv_map: dict,
    macro_df: pd.DataFrame,
    scored_news: pd.DataFrame,
    trending_df: pd.DataFrame,
    bundle: dict,
) -> None:
    hcfg = HORIZON_CONFIGS[horizon_name]
    st.subheader(f"Market Radar & Watchlists — {hcfg.display_name}")

    pulse_tickers = ["^NSEI", "^NSEBANK", "^INDIAVIX", "USDINR=X", "GOLDBEES.NS", "SILVERBEES.NS"]
    p_cols = st.columns(len(pulse_tickers))
    for idx_p, pt in enumerate(pulse_tickers):
        pdf = ohlcv_map.get(pt)
        if pdf is not None and len(pdf) >= 2:
            last_v = float(pdf["Close"].iloc[-1])
            chg_v = float(pdf["Close"].pct_change(1).iloc[-1] * 100.0)
            p_cols[idx_p].metric(pt, format_pulse_value(pt, last_v), f"{chg_v:+.2f}%")
        else:
            p_cols[idx_p].metric(pt, "N/A")

    scan_df, skipped_map, horizon_tr = _cached_market_scan(
        horizon_name,
        universe_name,
        include_trending,
        max_trending,
        ohlcv_map,
        macro_df,
        scored_news,
        bundle,
    )
    gate_info = _render_gate_banner(horizon_tr)

    if scan_df.empty:
        st.warning("No predictions generated across scanned tickers.")
        return

    raw_bull = int(scan_df["raw_verdict"].astype(str).str.contains("BULLISH").sum())
    raw_bear = int(scan_df["raw_verdict"].astype(str).str.contains("BEARISH").sum())
    raw_neut = int((scan_df["raw_verdict"] == "NEUTRAL / HOLD").sum())
    gross_alloc = float(scan_df["alloc_pct"].abs().sum())

    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Assets Scanned", len(scan_df), f"Skipped: {len(skipped_map)}")
    s2.metric("Raw Bull / Bear", f"{raw_bull} / {raw_bear}", "Directional counts")
    s3.metric("Raw Neutral / Hold", raw_neut)
    s4.metric("Live Portfolio Gross", f"{gross_alloc:.1f}%", "Locked 0%" if not gate_info["passed"] else "Cap <= 100%")
    s5.metric("News-Active Tickers", int((scan_df["buzz_score"] > 0).sum()), f"Trending: {len(trending_df)}")

    gainers_df, decliners_df = build_upside_downside_watchlists(scan_df, top_n=12)

    col_up, col_dn = st.columns(2)
    with col_up:
        st.markdown("#### Potential Profit Candidates (Top Raw Model Upside)")
        if not gainers_df.empty:
            st.dataframe(
                gainers_df[["Ticker", "Bar Date", "Last Close", "Model View", "Est. Move %", "Est. Target", "Up Prob %", "Model Agree %", "Risk Stop", "Target Price"]],
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("No positive expected-return candidates on this scan.")

    with col_dn:
        st.markdown("#### Potential Loss / Risk Candidates (Avoid Cash Longs / Short Watch)")
        if not decliners_df.empty:
            st.dataframe(
                decliners_df[["Ticker", "Bar Date", "Last Close", "Model View", "Est. Move %", "Expected Range", "Up Prob %", "Execution Note"]],
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("No negative expected-return candidates on this scan.")

    metals_df = build_metals_spotlight_df(scan_df)
    if not metals_df.empty:
        st.markdown("#### Gold & Silver ETFs + Global Bullion Spotlight")
        st.dataframe(metals_df, width="stretch", hide_index=True)

    with st.expander("Full Universe Screener (Filter, Search & Raw Return Bar Chart)", expanded=False):
        f1, f2, f3 = st.columns([1.2, 1.0, 1.0])
        with f1:
            mkt_filter = st.multiselect(
                "Market Segment",
                sorted(scan_df["market"].unique().tolist()),
                default=sorted(scan_df["market"].unique().tolist()),
            )
        with f2:
            search_q = st.text_input("Search Symbol", value="").strip().upper()
        with f3:
            sig_filter = st.selectbox(
                "Signal Filter",
                ["All Assets", "Raw Bullish Only", "Raw Bearish Only", "Passed Edge Gate Only", "Trending / Buzzed Only"],
            )

        filtered = scan_df[scan_df["market"].isin(mkt_filter)].copy()
        if search_q:
            filtered = filtered[filtered["ticker"].astype(str).str.contains(search_q, regex=False)]
        if sig_filter == "Raw Bullish Only":
            filtered = filtered[filtered["raw_verdict"].astype(str).str.contains("BULLISH")]
        elif sig_filter == "Raw Bearish Only":
            filtered = filtered[filtered["raw_verdict"].astype(str).str.contains("BEARISH")]
        elif sig_filter == "Passed Edge Gate Only":
            filtered = filtered[filtered["gate_passed"] == True]  # noqa: E712
        elif sig_filter == "Trending / Buzzed Only":
            filtered = filtered[filtered["buzz_score"] > 0]

        st.dataframe(build_screener_display_df(filtered), width="stretch", hide_index=True)
        plot_src = filtered[filtered["raw_verdict"] != "NEUTRAL / HOLD"].sort_values("composite_ret_pct", key=lambda s: s.abs(), ascending=False).head(16)
        if not plot_src.empty:
            fig_scan = px.bar(
                plot_src.sort_values("composite_ret_pct"),
                x="composite_ret_pct",
                y="ticker",
                orientation="h",
                color="raw_verdict",
                color_discrete_map={
                    "STRONG BULLISH": "#10b981",
                    "LEAN BULLISH": "#34d399",
                    "LEAN BEARISH": "#f87171",
                    "STRONG BEARISH": "#dc2626",
                },
                title=f"Raw Model Expected Return (%) — Top 16 Directional Moves ({hcfg.display_name})",
            )
            fig_scan.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_scan, width="stretch")


def _render_deep_dive(
    ticker: str,
    horizon_name: str,
    universe_name: str,
    scenario_capital: float,
    run_all_horizons: bool,
    ohlcv_map: dict,
    macro_df: pd.DataFrame,
    scored_news: pd.DataFrame,
    bundle: dict,
    settings: EngineSettings,
) -> None:
    st.subheader(f"Signal Deep Dive & Justification — {ticker}")

    # Ensure custom ticker exists in map without invalidating global market cache
    work_map = dict(ohlcv_map)
    if ticker not in work_map:
        hcfg = HORIZON_CONFIGS[horizon_name]
        extra = fetch_ohlcv(ticker, period=hcfg.period, interval=hcfg.interval, settings=settings)
        if not extra.empty:
            work_map[ticker] = extra

    t_df = work_map.get(ticker)
    latest_bar_key = str(t_df.index[-1]) if t_df is not None and not t_df.empty else "missing"

    try:
        report = _cached_single_asset_report(
            ticker,
            horizon_name,
            universe_name,
            latest_bar_key,
            work_map,
            macro_df,
            scored_news,
            bundle,
        )
    except Exception as exc:
        st.error(f"Could not generate live prediction for {ticker}: {exc}")
        return

    cur = get_currency_symbol(report.ticker)
    st.caption(
        f"Closed Bar Date Used: **{report.as_of_timestamp}** | Forecast Horizon: **{report.horizon_display}** | "
        f"Gate Check: `{report.gate_reason}`"
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Gate / Status", report.verdict, f"Raw: {report.raw_verdict}")
    c2.metric("Est. Expected Move", f"{report.composite_expected_ret_pct:+.2f}%", f"Close: {format_pulse_value(report.ticker, report.last_close)}")
    c3.metric("Upside Probability", f"{report.prob_up_pct:.1f}%", f"Agree: {report.model_agreement_pct:.0f}%")
    c4.metric("GARCH Annual Vol", f"{report.garch_ann_vol_pct:.1f}%", f"21D RV: {report.realized_vol21_pct:.1f}%")
    c5.metric("Live Auto Allocation", f"{report.suggested_allocation_pct:+.1f}%", f"SL {report.stop_loss_price} / TP {report.take_profit_price}")

    # Plain-English Justification Cards ("Why?")
    st.markdown("### Why This Signal? (Objective Evidence & Justification)")
    j_cards = build_signal_justification(report, position_capital=scenario_capital)
    left_c, right_c = st.columns(2)
    for idx, card in enumerate(j_cards):
        target_col = left_c if idx % 2 == 0 else right_c
        target_col.markdown(
            f'<div class="qa-card"><div class="qa-card-title">{card["title"]}</div>'
            f'<div class="qa-card-body">{card["body"]}</div></div>',
            unsafe_allow_html=True,
        )

    # What-If Capital Risk / Reward Calculator
    scen = compute_capital_scenario(report, scenario_capital)
    st.markdown(f"### What-If Position Simulator (Hypothetical Position: {cur}{scen['capital']:,.0f})")
    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
    sc1.metric("Est. Expected P&L", f"{cur}{scen['expected_pnl']:+,.0f}", f"{report.composite_expected_ret_pct:+.2f}%")
    sc2.metric("1-Sigma Downside", f"{cur}{scen['range_low_pnl']:+,.0f}", f"{report.expected_range_low_pct:+.2f}%")
    sc3.metric("1-Sigma Upside", f"{cur}{scen['range_high_pnl']:+,.0f}", f"{report.expected_range_high_pct:+.2f}%")
    sc4.metric("ATR Stop Risk", f"{cur}{scen['stop_loss_risk']:+,.0f}", f"Stop @ {report.stop_loss_price}")
    sc5.metric("ATR Target Reward", f"{cur}{scen['take_profit_gain']:+,.0f}", f"Target @ {report.take_profit_price}")

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("#### 6-Model Family Breakdown")
        mb_df = pd.DataFrame(report.model_breakdown)
        st.dataframe(mb_df, width="stretch", hide_index=True)
        fig_mb = px.bar(
            mb_df,
            x="model",
            y="expected_ret_pct",
            color="signal",
            color_discrete_map={"BULLISH": "#10b981", "BEARISH": "#ef4444", "NEUTRAL": "#64748b"},
            title="Expected Return (%) by Model Family",
        )
        fig_mb.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_mb, width="stretch")

    with col_r:
        st.markdown("#### Top Local Feature Drivers (SHAP)")
        if report.top_drivers:
            drv_df = pd.DataFrame(report.top_drivers).copy()
            drv_df["label"] = drv_df["feature"].map(humanize_feature_name)
            st.dataframe(drv_df[["label", "value", "shap_impact", "direction"]], width="stretch", hide_index=True)
            fig_drv = px.bar(
                drv_df.sort_values("shap_impact"),
                x="shap_impact",
                y="label",
                orientation="h",
                color="direction",
                color_discrete_map={"Bullish": "#10b981", "Bearish": "#ef4444"},
                title="Live Feature Impact (Vol-Normalized Score)",
            )
            fig_drv.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_drv, width="stretch")
        else:
            st.info("SHAP local attribution unavailable for this model configuration.")

    if run_all_horizons:
        st.markdown("#### Multi-Horizon Comparison Matrix (Intraday + Next-Day + Swing + Long-Term)")
        with st.spinner("Evaluating all horizons..."):
            try:
                matrix_reports = predict_multi_horizon_matrix(ticker=ticker, universe_name=universe_name, settings=settings)
                matrix_rows = [
                    {
                        "Horizon": r.horizon_display,
                        "Gate Status": r.verdict,
                        "Model View": r.raw_verdict,
                        "Est. Move %": r.composite_expected_ret_pct,
                        "Prob Up %": r.prob_up_pct,
                        "Consensus %": r.model_agreement_pct,
                        "GARCH Vol %": r.garch_ann_vol_pct,
                        "Regime": r.regime_label,
                        "Live Alloc %": r.suggested_allocation_pct,
                        "OOS Sharpe": r.backtest_track_record.get("sharpe"),
                    }
                    for r in matrix_reports
                ]
                st.dataframe(pd.DataFrame(matrix_rows), width="stretch", hide_index=True)
            except Exception as exc:
                st.warning(f"All-horizon matrix incomplete: {exc}")


def _render_paper_desk(
    universe_name: str,
    horizon_name: str,
    ohlcv_map: dict,
    macro_df: pd.DataFrame,
    scored_news: pd.DataFrame,
    bundle: dict,
) -> None:
    st.subheader("Realistic Paper Trading Simulator & Prediction Accuracy Tracker")
    sim = PaperBrokerSimulator()
    ledger = PaperLedger()

    # Multi-user account selector & creation bar
    accounts = sim.list_accounts()
    acc_ids = [a["account_id"] for a in accounts]
    acc_labels = {a["account_id"]: f"{a['owner_name']} ({a['account_id']})" for a in accounts}

    a_col1, a_col2, a_col3, a_col4 = st.columns([1.4, 1.0, 0.8, 0.8])
    with a_col1:
        active_acc_id = st.selectbox("Active Paper Portfolio Account", acc_ids, format_func=lambda k: acc_labels.get(k, k))
    with a_col2:
        new_user = st.text_input("Invite / Add New Tester Name", value="", placeholder="e.g., Friend_1")
    with a_col3:
        start_cap = st.number_input("Starting Capital", min_value=10000.0, max_value=10000000.0, value=100000.0, step=10000.0)
    with a_col4:
        st.write("")
        if st.button("Create / Reset Account", width="stretch"):
            if new_user.strip():
                uid = new_user.strip().lower().replace(" ", "_")
                sim.ensure_account(uid, new_user.strip(), starting_cash=start_cap)
                st.success(f"Created account '{new_user.strip()}' with {start_cap:,.0f}.")
                st.rerun()
            else:
                sim.reset_account(active_acc_id, starting_cash=start_cap)
                st.warning(f"Reset '{active_acc_id}' back to {start_cap:,.0f}.")
                st.rerun()

    # Live price map for mark-to-market valuations
    price_map: Dict[str, float] = {}
    for t, df in ohlcv_map.items():
        if df is not None and not df.empty:
            price_map[t] = float(df["Close"].iloc[-1])

    snap = sim.get_portfolio_snapshot(active_acc_id, price_map=price_map)
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Total Equity", f"Rs {snap['total_equity']:,.2f}", f"{snap['net_return_pct']:+.2f}%")
    m2.metric("Available Cash", f"Rs {snap['cash_balance']:,.2f}")
    m3.metric("Holdings Value", f"Rs {snap['market_value']:,.2f}", f"Cost: Rs {snap['invested_cost']:,.0f}")
    m4.metric("Unrealized P&L", f"Rs {snap['unrealized_pnl']:+,.2f}")
    m5.metric("Realized P&L", f"Rs {snap['realized_pnl']:+,.2f}")
    m6.metric("Broker Fees Paid", f"Rs {snap['total_fees_paid']:,.2f}", "STT+Stamp+GST+DP")

    st.divider()
    left_tick, right_corr = st.columns([1.05, 1.35])

    with left_tick:
        st.markdown("#### Simulated Order Ticket (Real Indian Broker Fee Engine)")
        scan_df, _, _ = _cached_market_scan(
            horizon_name, universe_name, False, 0, ohlcv_map, macro_df, scored_news, bundle
        )
        order_tickers = UNIVERSES[universe_name]
        ord_ticker = st.selectbox("Instrument", order_tickers, index=0, key="sim_ord_ticker")
        ord_side = st.radio("Order Side", ["BUY", "SELL"], horizontal=True)
        default_ltp = price_map.get(ord_ticker, 100.0)
        is_crypto = ord_ticker.upper().endswith(("-USD", "-USDT", "-BTC"))
        qty_step = 0.01 if is_crypto else 1.0
        default_qty = 0.05 if is_crypto else max(1.0, float(int(10000.0 // max(default_ltp, 1.0))))
        ord_qty = st.number_input("Quantity", min_value=qty_step, value=default_qty, step=qty_step)
        ord_price = st.number_input("Execution Price (Latest Closed Bar)", min_value=0.01, value=round(default_ltp, 2))

        preview = sim.preview_order(active_acc_id, ord_ticker, ord_side, ord_qty, ord_price)
        if preview.valid:
            f = preview.fees
            st.info(
                f"**Order Value:** {preview.notional:,.2f} | **Total Charges:** {f['total_fee']:.2f} ({f['fee_bps']:.1f} bps)\n\n"
                f"STT: `{f['stt']:.2f}` | Stamp: `{f['stamp_duty']:.2f}` | NSE+SEBI: `{f['exchange_sebi']:.2f}` | "
                f"GST: `{f['gst']:.2f}` | DP Sell: `{f['dp_charge']:.2f}` | **Round-Trip Breakeven:** `{preview.breakeven_move_pct:.3f}%`"
            )
        else:
            st.warning(preview.reason)

        row_sig = scan_df[scan_df["ticker"] == ord_ticker] if not scan_df.empty else pd.DataFrame()
        sig_v = str(row_sig["raw_verdict"].iloc[0]) if not row_sig.empty else ""
        sig_ret = float(row_sig["composite_ret_pct"].iloc[0]) if not row_sig.empty else 0.0
        sl_val = float(row_sig["stop_loss"].iloc[0]) if not row_sig.empty else ord_price * 0.96
        tp_val = float(row_sig["take_profit"].iloc[0]) if not row_sig.empty else ord_price * 1.06

        if st.button("Execute Paper Order", type="primary", width="stretch", disabled=not preview.valid):
            ok, msg = sim.execute_order(
                account_id=active_acc_id,
                ticker=ord_ticker,
                side=ord_side,
                qty=ord_qty,
                fill_price=ord_price,
                stop_loss=sl_val,
                take_profit=tp_val,
                signal_verdict=sig_v,
                signal_ret_pct=sig_ret,
                notes=f"horizon={horizon_name}",
            )
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

        st.markdown("#### Open Virtual Holdings")
        if not snap["positions_df"].empty:
            st.dataframe(snap["positions_df"], width="stretch", hide_index=True)
        else:
            st.caption("No open positions in this account yet. Use the Order Ticket above to simulate delivery trades.")

    with right_corr:
        st.markdown("#### Daily Prediction vs. Market Variation Correlation")
        st.caption(
            "Tracks whether model predictions actually correlate with subsequent closed-bar market moves before committing capital."
        )
        b1, b2 = st.columns(2)
        with b1:
            if st.button("Log Closed-Bar Predictions (Universe)", width="stretch"):
                with st.spinner("Recording zero-leakage closed-bar forecasts..."):
                    events = log_paper_signals(
                        tickers=UNIVERSES[universe_name],
                        horizons=[horizon_name],
                        universe_name=universe_name,
                        use_cache=True,
                        ledger=ledger,
                    )
                st.success(f"Logged/verified {len(events)} closed-bar entries.")
        with b2:
            if st.button("Grade Matured Predictions vs. Market", width="stretch"):
                with st.spinner("Evaluating settled bars..."):
                    ev = ledger.evaluate_open_predictions()
                st.success(f"Checked: {ev.checked_open} | Newly Matured: {ev.newly_matured} | Open: {ev.still_open}")

        corr_stats = compute_prediction_market_correlation(ledger)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Matured Samples", corr_stats["n_matured"])
        c2.metric("Pearson r", f"{corr_stats['pearson_r']}" if corr_stats["pearson_r"] is not None else "Pending (<2)")
        c3.metric("Spearman IC", f"{corr_stats['spearman_ic']}" if corr_stats["spearman_ic"] is not None else "Pending (<2)")
        c4.metric(
            "Hit Rate vs Long Base",
            f"{corr_stats['hit_rate_pct']}%" if corr_stats["hit_rate_pct"] is not None else "Pending",
            f"Always Long: {corr_stats['always_long_pct']}%" if corr_stats["always_long_pct"] is not None else None,
        )

        matured_df = corr_stats["df"]
        if not matured_df.empty and len(matured_df) >= 2:
            fig_sc = px.scatter(
                matured_df,
                x="composite_expected_ret_pct",
                y="actual_asset_ret_pct",
                color="horizon_name",
                hover_data=["ticker", "as_of_timestamp", "raw_verdict"] if "raw_verdict" in matured_df.columns else ["ticker", "as_of_timestamp"],
                title="Predicted Move (%) vs. Actual Market Move (%)",
                labels={"composite_expected_ret_pct": "Model Predicted Return (%)", "actual_asset_ret_pct": "Actual Realized Return (%)"},
            )
            fig_sc.add_hline(y=0, line_dash="dot", line_color="#64748b")
            fig_sc.add_vline(x=0, line_dash="dot", line_color="#64748b")
            fig_sc.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_sc, width="stretch")
        else:
            st.info(
                "No matured forward comparisons yet. Click **Log Closed-Bar Predictions (Universe)** today, "
                "and once the horizon settles (`1` trading day for Tomorrow, `5` days for Swing), click **Grade Matured Predictions**."
            )

    if not snap["orders_df"].empty:
        st.markdown("#### Simulated Order Blotter & Fee Audit")
        st.dataframe(snap["orders_df"], width="stretch", hide_index=True)


def _render_backtest_tab(horizon_name: str, bundle: dict) -> None:
    oof_df = bundle.get("oof_predictions")
    if oof_df is None or oof_df.empty:
        st.warning("No walk-forward backtest predictions stored yet.")
        return

    hcfg = HORIZON_CONFIGS[horizon_name]
    lb_df, eq_df, dd_df, asset_df = _cached_backtest(horizon_name, oof_df, tuple(bundle["model_names"]))
    st.subheader(f"Strictly Out-of-Sample Walk-Forward Proof — {hcfg.display_name}")
    st.caption(
        "Evaluated on chronological canonical exposures (duplicate Gold/Silver/Index ETFs deduplicated during training & CV) "
        "using benchmark-relative Newey-West HAC alpha, Deflated Sharpe Ratio, and moving-block bootstrap 95% CIs."
    )
    st.dataframe(lb_df, width="stretch", hide_index=True)

    eq_long = eq_df.reset_index().melt(id_vars="Timestamp", var_name="Strategy", value_name="Equity")
    fig_eq = px.line(eq_long, x="Timestamp", y="Equity", color="Strategy", title="Out-of-Sample Cumulative Equity (Net of Costs)")
    fig_eq.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_eq, width="stretch")

    dd_long = dd_df.reset_index().melt(id_vars="Timestamp", var_name="Strategy", value_name="Drawdown %")
    fig_dd = px.line(dd_long, x="Timestamp", y="Drawdown %", color="Strategy", title="Out-of-Sample Drawdown (%)")
    fig_dd.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_dd, width="stretch")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Per-Asset OOS Summary")
        st.dataframe(asset_df, width="stretch", hide_index=True)
    with col_b:
        st.markdown("#### Top Global Features (Mean |SHAP|)")
        shap_df = bundle.get("shap_importance")
        if shap_df is not None and not shap_df.empty:
            top_s = shap_df.head(15).copy()
            top_s["label"] = top_s["feature"].map(humanize_feature_name)
            fig_shap = px.bar(
                top_s.iloc[::-1],
                x="mean_abs_shap",
                y="label",
                orientation="h",
                title="Top 15 Global Quantitative & Cross-Sectional Drivers",
            )
            fig_shap.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_shap, width="stretch")


def _render_guide_and_feeds(
    ticker: str,
    horizon_name: str,
    universe_name: str,
    ohlcv_map: dict,
    macro_df: pd.DataFrame,
    scored_news: pd.DataFrame,
) -> None:
    st.subheader("How to Use This Platform + Exact Indian Broker Cost Sheet + Feed Audit")

    g1, g2 = st.columns(2)
    with g1:
        st.markdown(
            """
            <div class="qa-card">
                <div class="qa-card-title">Step-by-Step Beginner Workflow</div>
                <div class="qa-card-body">
                    <strong>1. Check Market Radar &amp; Tomorrow:</strong> Inspect top raw upside &amp; downside risk movers and note the closed-bar date.<br>
                    <strong>2. Open Signal Deep Dive &amp; Why:</strong> Verify the 5-part justification (Quant Drivers, GARCH Risk, Regime, News, Fees).<br>
                    <strong>3. Paper Trade &amp; Track Correlation:</strong> Simulate delivery buys/sells in INR and log closed-bar predictions. Wait for the horizon to settle, then grade Pearson/Spearman correlation.<br>
                    <strong>4. Respect the Edge Gate:</strong> Only trust automated allocations after OOS Alpha &gt; 0, HAC t-stat &ge; 2.0, and DSR &ge; 0.95 pass.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with g2:
        st.markdown(
            f"""
            <div class="qa-card">
                <div class="qa-card-title">Exact Delivery Fee Model Used in Paper Simulator</div>
                <div class="qa-card-body">
                    • <strong>Equity Delivery Brokerage:</strong> {DEFAULT_FEE_SCHEDULE.india_brokerage_rate*100:.2f}% (Zero discount delivery standard)<br>
                    • <strong>STT:</strong> {DEFAULT_FEE_SCHEDULE.india_equity_stt_rate*100:.2f}% Buy &amp; Sell (Equities) / {DEFAULT_FEE_SCHEDULE.india_etf_sell_stt_rate*100:.3f}% Sell (ETFs)<br>
                    • <strong>Stamp Duty (Buy):</strong> {DEFAULT_FEE_SCHEDULE.india_stamp_duty_buy*100:.3f}% | <strong>NSE+SEBI:</strong> 0.00307% + 18% GST<br>
                    • <strong>CDSL DP Charge (Sell):</strong> Rs {DEFAULT_FEE_SCHEDULE.india_dp_sell_charge:.2f} flat per ISIN sell execution<br>
                    • <strong>Crypto Spot:</strong> {DEFAULT_FEE_SCHEDULE.crypto_fee_rate*100:.2f}% per side
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Multi-Source Data Feed Audit")
    audit_df = run_full_feed_audit(ohlcv_map, macro_df, scored_news, driver_tickers=set(GLOBAL_DRIVERS))
    st.dataframe(audit_df, width="stretch", hide_index=True)

    col_c1, col_c2 = st.columns([1.2, 1.8])
    with col_c1:
        corr_candidates = list(
            dict.fromkeys(
                ["^NSEI", "^NSEBANK", "GOLDBEES.NS", "SILVERBEES.NS", "GLD", "SLV", "USDINR=X", "BZ=F", "^INDIAVIX", "SPY", "^VIX"]
                + UNIVERSES[universe_name][:10]
            )
        )
        corr_tickers = [t for t in corr_candidates if t in ohlcv_map][:14]
        corr_df = pd.DataFrame({t: ohlcv_map[t]["Close"].pct_change(1) for t in corr_tickers}).tail(252).corr().round(2)
        if not corr_df.empty:
            fig_corr = px.imshow(corr_df, text_auto=True, color_continuous_scale="RdBu", zmin=-1, zmax=1, title="252-Bar Cross-Asset Correlation")
            fig_corr.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_corr, width="stretch")

    with col_c2:
        st.markdown("#### Live Financial News Sentiment Feed")
        if scored_news is not None and not scored_news.empty:
            st.dataframe(
                scored_news[["published_at", "ticker", "feed", "source", "sentiment_label", "sentiment_score", "title"]].head(25),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("No news headlines retrieved in current session.")


def main() -> None:
    configure_logging()
    ensure_directories()
    inject_quant_theme()

    st.title("Quant Alpha Terminal — India Research & Paper-Trading Desk")
    st.caption(
        f"{_nse_market_status_str()} • Strictly Causal ML Ensemble + Chronological Market HMM + Peer Cross-Sectional Ranks + Realistic INR Costs"
    )

    with st.sidebar:
        st.header("Desk Controls")
        universe_name = st.selectbox("Market Universe", list(UNIVERSES.keys()), index=0)
        horizon_name = st.selectbox(
            "Forecast Horizon",
            list(HORIZON_CONFIGS.keys()),
            index=0,
            format_func=lambda k: HORIZON_CONFIGS[k].display_name,
        )
        scenario_capital = st.number_input(
            "What-If Position Size (INR / USD)",
            min_value=1000.0,
            max_value=100000000.0,
            value=100000.0,
            step=10000.0,
        )
        st.divider()
        default_tickers = UNIVERSES[universe_name]
        selected_ticker = st.selectbox("Stock / ETF / Crypto Focus", default_tickers, index=0)
        custom_ticker = st.text_input("Or Custom Ticker (e.g., HAL.NS, BTC-USD)", value="").strip().upper()
        ticker = custom_ticker or selected_ticker

        if st.button("Refresh Market Quotes & News", width="stretch"):
            snap_file = DATA_CACHE_DIR / "snapshots" / f"{horizon_name}__{universe_name}.joblib"
            if snap_file.exists():
                snap_file.unlink()
            _cached_market_bundle.clear()
            _cached_market_scan.clear()
            _cached_single_asset_report.clear()
            st.rerun()

        with st.expander("Advanced & Retraining Controls", expanded=False):
            include_trending = st.checkbox("Auto-Discover Trending Tickers", value=True)
            max_trending = st.slider("Extra Trending Tickers", min_value=0, max_value=20, value=10)
            run_all_horizons = st.checkbox("Compute 4-Horizon Matrix in Deep Dive", value=False)
            tune_trials = st.slider("Optuna Trials (retraining)", min_value=5, max_value=30, value=12)
            retrain_clicked = st.button("Retrain Active Horizon Bundle", type="primary", width="stretch")

    settings = EngineSettings(n_optuna_trials=tune_trials)

    if retrain_clicked:
        with st.spinner(f"Training & tuning purged walk-forward models for {horizon_name}..."):
            ensure_horizon_bundle(
                horizon_name=horizon_name,
                universe_name=universe_name,
                tune=True,
                settings=settings,
                force_retrain=True,
            )
        st.cache_resource.clear()
        st.cache_data.clear()
        st.success(f"Completed purged walk-forward training for '{horizon_name}'.")

    with st.spinner("Loading cached market bundle..."):
        ohlcv_map, macro_df, scored_news, trending_df = _cached_market_bundle(
            universe_name, horizon_name, include_trending, max_trending
        )

    try:
        bundle = _cached_ensemble_bundle(horizon_name)
    except FileNotFoundError:
        with st.spinner(f"No saved bundle for '{horizon_name}'. Training initial bundle..."):
            bundle = ensure_horizon_bundle(
                horizon_name=horizon_name,
                universe_name=universe_name,
                tune=True,
                settings=settings,
            )
            st.cache_resource.clear()

    workspace = st.radio(
        "Workspace",
        [
            "Market Radar & Tomorrow",
            "Signal Deep Dive & Why",
            "Paper Trading Simulator",
            "Alpha Proof & Backtest",
            "Guide, Fees & Data Health",
        ],
        horizontal=True,
        label_visibility="collapsed",
    )

    if workspace == "Market Radar & Tomorrow":
        _render_market_radar(
            horizon_name, universe_name, include_trending, max_trending, ohlcv_map, macro_df, scored_news, trending_df, bundle
        )
    elif workspace == "Signal Deep Dive & Why":
        _render_deep_dive(
            ticker, horizon_name, universe_name, scenario_capital, run_all_horizons, ohlcv_map, macro_df, scored_news, bundle, settings
        )
    elif workspace == "Paper Trading Simulator":
        _render_paper_desk(universe_name, horizon_name, ohlcv_map, macro_df, scored_news, bundle)
    elif workspace == "Alpha Proof & Backtest":
        _render_backtest_tab(horizon_name, bundle)
    elif workspace == "Guide, Fees & Data Health":
        _render_guide_and_feeds(ticker, horizon_name, universe_name, ohlcv_map, macro_df, scored_news)


if __name__ == "__main__":
    main()
