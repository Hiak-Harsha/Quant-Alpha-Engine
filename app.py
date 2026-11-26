"""Interactive Streamlit dashboard for Quant Alpha Engine."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quant_alpha.backtest.engine import run_backtest_comparison
from quant_alpha.config import (
    GLOBAL_DRIVERS,
    HORIZON_CONFIGS,
    UNIVERSES,
    EngineSettings,
    configure_logging,
    ensure_directories,
)
from quant_alpha.data.macro_feed import load_all_macro_series
from quant_alpha.data.market_feed import fetch_universe_ohlcv
from quant_alpha.data.news_feed import fetch_all_news
from quant_alpha.data.trending_feed import discover_trending_stocks
from quant_alpha.data.validator import run_full_feed_audit
from quant_alpha.features.cross_asset import build_cross_asset_features
from quant_alpha.features.sentiment import score_news_dataframe
from quant_alpha.features.volatility import compute_volatility_features
from quant_alpha.models.ensemble import load_ensemble_artifacts
from quant_alpha.paper.ledger import PaperLedger, log_paper_signals
from quant_alpha.predictor import (
    ensure_horizon_bundle,
    predict_asset,
    predict_multi_horizon_matrix,
    scan_market_predictions,
)
from quant_alpha.ui.components import (
    build_metals_spotlight_df,
    build_screener_display_df,
    build_upside_downside_watchlists,
    compute_capital_scenario,
    format_pulse_value,
    summarize_gate_status,
)

st.set_page_config(
    page_title="Quant Alpha Engine | Multi-Horizon Trading & Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(ttl=900, show_spinner=False)
def _cached_market_bundle(
    universe_name: str,
    horizon_name: str,
    extra_ticker: str,
    include_trending: bool,
    max_trending: int,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hcfg = HORIZON_CONFIGS[horizon_name]
    base_tickers = list(UNIVERSES[universe_name])
    india_focus = universe_name.startswith("india_")
    macro_df = load_all_macro_series()
    news_tickers = list(dict.fromkeys(base_tickers[:35] + ([extra_ticker] if extra_ticker else [])))
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

    tickers = list(dict.fromkeys(scan_tickers + GLOBAL_DRIVERS + ([extra_ticker] if extra_ticker else [])))
    ohlcv_map, _ = fetch_universe_ohlcv(tickers, period=hcfg.period, interval=hcfg.interval, max_workers=10)
    return ohlcv_map, macro_df, news_df, trending_df


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
            return (
                lb_cached,
                pd.read_parquet(eq_p),
                pd.read_parquet(dd_p),
                as_cached,
            )

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
    return f"{now_ist.strftime('%Y-%m-%d %H:%M:%S IST')} | {status}"


def main() -> None:
    configure_logging()
    ensure_directories()
    st.title("Quant Alpha Engine — India Market, Gold/Silver ETFs & Live Monitor")
    st.caption(
        f"{_nse_market_status_str()} • Closed-bar causal predictions across LightGBM, XGBoost, ExtraTrees, "
        "Ridge/Logistic, Neural MLP, Classical Quant Signals, GARCH(1,1) Vol & Gaussian HMM Regimes."
    )

    with st.sidebar:
        st.header("Configuration")
        universe_name = st.selectbox("Asset Universe", list(UNIVERSES.keys()), index=0)
        horizon_name = st.selectbox(
            "Prediction Horizon",
            list(HORIZON_CONFIGS.keys()),
            index=0,
            format_func=lambda k: HORIZON_CONFIGS[k].display_name,
        )
        scenario_capital = st.number_input(
            "What-If Position Size (Scenario Estimator)",
            min_value=1000.0,
            max_value=100000000.0,
            value=100000.0,
            step=10000.0,
        )
        st.divider()
        st.subheader("Live Monitor Controls")
        auto_refresh = st.toggle("Enable Auto-Refresh Live Monitor", value=False)
        refresh_secs = st.selectbox("Refresh Interval", [60, 180, 300], index=1, format_func=lambda s: f"{s} sec")
        if st.button("Refresh Live Quotes & News Now", width="stretch"):
            _cached_market_bundle.clear()
            _cached_market_scan.clear()
            st.rerun()

        st.divider()
        include_trending = st.checkbox("Auto-Discover Trending & Buzzed Tickers", value=True)
        max_trending = st.slider("Extra Trending Stocks to Scan", min_value=0, max_value=20, value=10)
        default_tickers = UNIVERSES[universe_name]
        selected_ticker = st.selectbox("Single-Asset Deep Dive", default_tickers, index=0)
        custom_ticker = st.text_input("Or Custom Ticker (e.g., TATAPOWER.NS, GOLDBEES.NS)", value="").strip().upper()
        ticker = custom_ticker or selected_ticker

        run_all_horizons = st.checkbox("Compute All-Horizons Matrix (Intraday + Next-Day + Swing + Long-Term)", value=False)
        tune_trials = st.slider("Optuna Tuning Trials (when retraining)", min_value=5, max_value=30, value=12)
        retrain_clicked = st.button("Train / Retune Active Horizon", type="primary")

    if auto_refresh:
        if "_monitor_tick" not in st.session_state:
            st.session_state["_monitor_tick"] = 0

        @st.fragment(run_every=refresh_secs)
        def _live_monitor_heartbeat() -> None:
            st.session_state["_monitor_tick"] += 1
            if st.session_state["_monitor_tick"] > 1:
                _cached_market_bundle.clear()
                _cached_market_scan.clear()
                st.rerun()
            st.caption(f"Auto-refresh active every {refresh_secs}s • Last checked: {_nse_market_status_str()}")

        _live_monitor_heartbeat()

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
        st.cache_data.clear()
        st.success(f"Completed purged walk-forward training and saved artifacts for '{horizon_name}'.")

    with st.spinner("Loading stocks, Gold/Silver ETFs, macro, volatility & news feeds..."):
        ohlcv_map, macro_df, scored_news, trending_df = _cached_market_bundle(
            universe_name, horizon_name, ticker, include_trending, max_trending
        )

    try:
        bundle = load_ensemble_artifacts(horizon_name)
    except FileNotFoundError:
        with st.spinner(f"No saved bundle for '{horizon_name}'. Running initial walk-forward training..."):
            bundle = ensure_horizon_bundle(
                horizon_name=horizon_name,
                universe_name=universe_name,
                tune=True,
                settings=settings,
            )

    tabs = st.tabs(
        [
            "India & Metals Watchlist",
            "Single-Asset Deep Dive",
            "Walk-Forward Backtest & Leaderboard",
            "Cross-Asset Correlation & Volatility",
            "Feed Validation & News Radar",
            "Live Paper Tracker",
        ]
    )

    # ------------------------------------------------------------------ TAB 1: SCANNER & WATCHLISTS
    with tabs[0]:
        hcfg = HORIZON_CONFIGS[horizon_name]
        st.subheader(f"Market Watchlist & Outlook — {hcfg.display_name}")

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

        gate_info = summarize_gate_status(horizon_tr)
        if gate_info["passed"]:
            st.success(f"**{gate_info['headline']}** — {gate_info['detail']}")
        else:
            st.warning(
                f"**{gate_info['headline']}**\n\n"
                f"{gate_info['detail']}. "
                "The **Top Potential Upside** and **Top Potential Downside / Risk** tables below show the model's raw closed-bar forecasts "
                "for objective research & paper tracking while keeping live allocation locked at `0.0%`."
            )

        if scan_df.empty:
            st.warning("No predictions generated across scanned tickers.")
        else:
            raw_bull = int(scan_df["raw_verdict"].astype(str).str.contains("BULLISH").sum())
            raw_bear = int(scan_df["raw_verdict"].astype(str).str.contains("BEARISH").sum())
            raw_neut = int((scan_df["raw_verdict"] == "NEUTRAL / HOLD").sum())
            gross_alloc = float(scan_df["alloc_pct"].abs().sum())

            s1, s2, s3, s4, s5 = st.columns(5)
            s1.metric("Assets Scanned", len(scan_df), f"Skipped: {len(skipped_map)}")
            s2.metric("Raw Bullish / Bearish", f"{raw_bull} / {raw_bear}", "Ungated model view")
            s3.metric("Raw Neutral / Hold", raw_neut)
            s4.metric("Live Portfolio Gross", f"{gross_alloc:.1f}%", "Gate Active" if not gate_info["passed"] else "Live Cap <= 100%")
            s5.metric("Buzzed / News Active", int((scan_df["buzz_score"] > 0).sum()), f"Trending: {len(trending_df)}")

            gainers_df, decliners_df = build_upside_downside_watchlists(scan_df, top_n=12)

            st.markdown("### Potential Profit Candidates (Top Raw Model Upside)")
            if not gainers_df.empty:
                st.dataframe(gainers_df, width="stretch", hide_index=True)
            else:
                st.info("No positive expected-return candidates on this closed-bar scan.")

            st.markdown("### Potential Loss / Downside Risk Candidates (Avoid Cash Longs / Short Watch)")
            if not decliners_df.empty:
                st.dataframe(decliners_df, width="stretch", hide_index=True)
            else:
                st.info("No negative expected-return candidates on this closed-bar scan.")

            metals_df = build_metals_spotlight_df(scan_df)
            if not metals_df.empty:
                st.markdown("### Gold & Silver ETFs + Bullion Spotlight")
                st.dataframe(metals_df, width="stretch", hide_index=True)

            st.markdown("### Full Universe Screener & Paper-Trade Logging")
            f_col1, f_col2, f_col3, f_col4 = st.columns([1.1, 1.1, 1.1, 1.2])
            with f_col1:
                mkt_filter = st.multiselect(
                    "Market Segment",
                    sorted(scan_df["market"].unique().tolist()),
                    default=sorted(scan_df["market"].unique().tolist()),
                )
            with f_col2:
                search_q = st.text_input("Search Ticker", value="").strip().upper()
            with f_col3:
                sig_filter = st.selectbox(
                    "Model Signal Filter",
                    ["All Assets", "Raw Bullish Only", "Raw Bearish Only", "Passed Edge Gate Only", "Trending / Buzzed Only"],
                    index=0,
                )
            with f_col4:
                log_top_btn = st.button("Log Top 12 Actionable Picks to Paper Ledger", width="stretch")

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

            if log_top_btn:
                actionable_tickers = (
                    scan_df[~scan_df["verdict"].isin({"NEUTRAL / HOLD", "NO EDGE / GATED"})]["ticker"]
                    .head(12)
                    .tolist()
                )
                if actionable_tickers:
                    with st.spinner("Logging closed-bar signals for top actionable stocks..."):
                        events = log_paper_signals(
                            tickers=actionable_tickers,
                            horizons=[horizon_name],
                            universe_name=universe_name,
                            use_cache=True,
                        )
                    st.success(f"Logged/checked {len(events)} closed-bar signals in Live Paper Tracker.")
                else:
                    st.info("No non-gated actionable signals to log right now (Tab 6 lets you log research universe calls).")

            screener_df = build_screener_display_df(filtered)
            st.dataframe(screener_df, width="stretch", hide_index=True)

            plot_src = filtered[filtered["raw_verdict"] != "NEUTRAL / HOLD"].sort_values("composite_ret_pct", key=lambda s: s.abs(), ascending=False).head(16)
            if not plot_src.empty:
                fig_scan = px.bar(
                    plot_src.sort_values("composite_ret_pct"),
                    x="composite_ret_pct",
                    y="ticker",
                    orientation="h",
                    color="raw_verdict",
                    color_discrete_map={
                        "STRONG BULLISH": "#15803d",
                        "LEAN BULLISH": "#22c55e",
                        "LEAN BEARISH": "#f87171",
                        "STRONG BEARISH": "#b91c1c",
                    },
                    title=f"Raw Model Expected Return (%) — Top 16 Directional Moves ({hcfg.display_name})",
                )
                st.plotly_chart(fig_scan, width="stretch")

            with st.expander("Live Trending & Most-Talked-About Discovery Feed"):
                if not trending_df.empty:
                    st.dataframe(trending_df.head(30), width="stretch", hide_index=True)
                if skipped_map:
                    st.caption(f"Skipped tickers lacking history (<70 bars): {skipped_map}")

    # ------------------------------------------------------------------ TAB 2: SINGLE ASSET DEEP DIVE
    with tabs[1]:
        report = None
        try:
            report = predict_asset(
                ticker=ticker,
                horizon_name=horizon_name,
                universe_name=universe_name,
                settings=settings,
                ohlcv_map=ohlcv_map,
                macro_df=macro_df,
                scored_news=scored_news,
                bundle=bundle,
            )
        except Exception as exc:
            st.error(f"Could not generate live prediction for {ticker}: {exc}")

        if report is not None:
            st.caption(
                f"Closed Bar Used: **{report.as_of_timestamp}** | Horizon: **{report.horizon_display}** | "
                f"Gate Reason: **{report.gate_reason}**"
            )
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Gate Status", report.verdict, f"Raw Model: {report.raw_verdict}")
            c2.metric(
                "Est. Expected Move",
                f"{report.composite_expected_ret_pct:+.2f}%",
                f"Close: {format_pulse_value(report.ticker, report.last_close)}",
            )
            c3.metric("Upside Probability", f"{report.prob_up_pct:.1f}%", f"Consensus: {report.model_agreement_pct:.0f}%")
            c4.metric(
                "GARCH Annual Vol",
                f"{report.garch_ann_vol_pct:.1f}%",
                f"21-bar RV: {report.realized_vol21_pct:.1f}%",
            )
            c5.metric(
                "Live Allocation",
                f"{report.suggested_allocation_pct:+.1f}%",
                f"Stop {report.stop_loss_price} / TP {report.take_profit_price}",
            )

            scen = compute_capital_scenario(report, scenario_capital)
            st.subheader(f"Objective Risk & Return Scenario (Hypothetical Position: {scen['capital']:,.0f})")
            sc1, sc2, sc3, sc4, sc5 = st.columns(5)
            sc1.metric("Est. Expected P&L", f"{scen['expected_pnl']:+,.0f}", f"{report.composite_expected_ret_pct:+.2f}%")
            sc2.metric("1-Sigma Downside", f"{scen['range_low_pnl']:+,.0f}", f"{report.expected_range_low_pct:+.2f}%")
            sc3.metric("1-Sigma Upside", f"{scen['range_high_pnl']:+,.0f}", f"{report.expected_range_high_pct:+.2f}%")
            sc4.metric("ATR Stop-Loss Risk", f"{scen['stop_loss_risk']:+,.0f}", f"Stop @ {report.stop_loss_price}")
            sc5.metric("ATR Target Reward", f"{scen['take_profit_gain']:+,.0f}", f"Live Gated Capital: {scen['gated_capital']:,.0f}")

            st.info(
                f"**HMM Market Regime:** {report.regime_label} ({report.regime_probs}) | "
                f"**1-Sigma Return Band:** [{report.expected_range_low_pct:+.2f}%, {report.expected_range_high_pct:+.2f}%] | "
                f"**Round-Trip Cost Hurdle:** {report.cost_hurdle_pct:.2f}% | "
                f"**VIX:** {report.vix_level if report.vix_level is not None else 'N/A'}"
            )

            if report.backtest_track_record:
                tr = report.backtest_track_record
                st.caption(
                    f"Historical Out-of-Sample Walk-Forward Track Record ({horizon_name}): "
                    f"Sharpe = {tr.get('sharpe')} (95% Block-CI [{tr.get('sharpe_ci_low')}, {tr.get('sharpe_ci_high')}]), "
                    f"Alpha = {tr.get('alpha_ann_pct')}% (HAC t={tr.get('hac_t_stat')}, DSR={tr.get('dsr_prob')}), "
                    f"Spearman IC = {tr.get('spearman_ic')}, Max Drawdown = {tr.get('max_drawdown_pct')}% -> "
                    f"**{tr.get('edge_verdict')}**"
                )

            col_l, col_r = st.columns(2)
            with col_l:
                st.subheader("Multi-Model Prediction Breakdown")
                mb_df = pd.DataFrame(report.model_breakdown)
                st.dataframe(mb_df, width="stretch", hide_index=True)
                fig_mb = px.bar(
                    mb_df,
                    x="model",
                    y="expected_ret_pct",
                    color="signal",
                    color_discrete_map={"BULLISH": "#16a34a", "BEARISH": "#dc2626", "NEUTRAL": "#64748b"},
                    title="Expected Return (%) by Model Family",
                )
                st.plotly_chart(fig_mb, width="stretch")

            with col_r:
                st.subheader("Top Local Drivers (SHAP Attribution)")
                if report.top_drivers:
                    drv_df = pd.DataFrame(report.top_drivers)
                    st.dataframe(drv_df, width="stretch", hide_index=True)
                    fig_drv = px.bar(
                        drv_df.sort_values("shap_impact"),
                        x="shap_impact",
                        y="feature",
                        orientation="h",
                        color="direction",
                        color_discrete_map={"Bullish": "#16a34a", "Bearish": "#dc2626"},
                        title="Live Feature Impact (Vol-Normalized Score)",
                    )
                    st.plotly_chart(fig_drv, width="stretch")
                else:
                    st.write("SHAP local attribution unavailable for this model configuration.")

            if run_all_horizons:
                st.subheader("Multi-Horizon Consensus Matrix (Intraday + Next-Day + Swing + Long-Term)")
                with st.spinner("Evaluating all horizons..."):
                    try:
                        matrix_reports = predict_multi_horizon_matrix(
                            ticker=ticker, universe_name=universe_name, settings=settings
                        )
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

    # ------------------------------------------------------------------ TAB 3: BACKTEST
    with tabs[2]:
        oof_df = bundle.get("oof_predictions")
        if oof_df is None or oof_df.empty:
            st.warning("No walk-forward backtest predictions stored yet.")
        else:
            hcfg = HORIZON_CONFIGS[horizon_name]
            lb_df, eq_df, dd_df, asset_df = _cached_backtest(horizon_name, oof_df, tuple(bundle["model_names"]))

            st.subheader(f"Strictly Out-of-Sample Walk-Forward Leaderboard ({hcfg.display_name})")
            st.dataframe(lb_df, width="stretch", hide_index=True)

            eq_long = eq_df.reset_index().melt(id_vars="Timestamp", var_name="Strategy", value_name="Equity")
            fig_eq = px.line(eq_long, x="Timestamp", y="Equity", color="Strategy", title="Out-of-Sample Cumulative Equity (Net of Costs)")
            st.plotly_chart(fig_eq, width="stretch")

            dd_long = dd_df.reset_index().melt(id_vars="Timestamp", var_name="Strategy", value_name="Drawdown %")
            fig_dd = px.line(dd_long, x="Timestamp", y="Drawdown %", color="Strategy", title="Out-of-Sample Drawdown (%)")
            st.plotly_chart(fig_dd, width="stretch")

            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("Per-Asset Ensemble Out-of-Sample Performance")
                st.dataframe(asset_df, width="stretch", hide_index=True)
            with col_b:
                st.subheader("Top Global Features (Mean |SHAP|)")
                shap_df = bundle.get("shap_importance")
                if shap_df is not None and not shap_df.empty:
                    fig_shap = px.bar(
                        shap_df.head(15).iloc[::-1],
                        x="mean_abs_shap",
                        y="feature",
                        orientation="h",
                        title="Top 15 Quantitative & Cross-Asset Drivers",
                    )
                    st.plotly_chart(fig_shap, width="stretch")

    # ------------------------------------------------------------------ TAB 4: CORRELATIONS
    with tabs[3]:
        st.subheader("Cross-Asset Return Correlation Matrix")
        corr_candidates = list(
            dict.fromkeys(
                ["^NSEI", "^NSEBANK", "GOLDBEES.NS", "SILVERBEES.NS", "GLD", "SLV", "USDINR=X", "BZ=F", "^INDIAVIX", "SPY", "^VIX"]
                + UNIVERSES[universe_name][:12]
            )
        )
        corr_tickers = [t for t in corr_candidates if t in ohlcv_map][:18]
        ret_series = {t: ohlcv_map[t]["Close"].pct_change(1) for t in corr_tickers}
        corr_df = pd.DataFrame(ret_series).tail(252).corr().round(2)
        if not corr_df.empty:
            fig_corr = px.imshow(
                corr_df,
                text_auto=True,
                color_continuous_scale="RdBu",
                zmin=-1,
                zmax=1,
                title="Trailing 252-Bar Cross-Asset Return Correlations (India, Gold/Silver, FX & Global)",
            )
            st.plotly_chart(fig_corr, width="stretch")

        if ticker in ohlcv_map:
            hcfg = HORIZON_CONFIGS[horizon_name]
            vol_df = compute_volatility_features(ohlcv_map[ticker], bars_per_year=hcfg.bars_per_year).tail(300)
            cross_df = build_cross_asset_features(
                ticker,
                ohlcv_map[ticker],
                ohlcv_map,
                macro_df,
                is_intraday=hcfg.interval not in {"1d", "1wk", "1mo"},
                bars_per_year=hcfg.bars_per_year,
            ).tail(300)

            col_v1, col_v2 = st.columns(2)
            with col_v1:
                v_plot = vol_df[["rv_21", "gk_vol_21", "garch_vol_forecast"]].dropna(how="all") * 100.0
                fig_v = px.line(v_plot, title=f"{ticker} Annualized Volatility: Realized vs Garman-Klass vs GARCH(1,1) (%)")
                st.plotly_chart(fig_v, width="stretch")
            with col_v2:
                corr_cols = [
                    c for c in (
                        "corr21_nsei", "corr21_gold_fut", "corr21_silver_fut",
                        "corr21_usdinr", "corr21_spy", "corr21_vix"
                    ) if c in cross_df.columns
                ]
                if corr_cols:
                    fig_rc = px.line(cross_df[corr_cols], title=f"{ticker} Rolling 21-Bar Correlations (NIFTY, Gold, Silver, USD/INR, SPY)")
                    st.plotly_chart(fig_rc, width="stretch")

    # ------------------------------------------------------------------ TAB 5: FEEDS & NEWS
    with tabs[4]:
        st.subheader("Multi-Source Feed Validation Audit")
        audit_df = run_full_feed_audit(ohlcv_map, macro_df, scored_news, driver_tickers=set(GLOBAL_DRIVERS))
        st.dataframe(audit_df, width="stretch", hide_index=True)

        col_m, col_n = st.columns([1, 2])
        with col_m:
            st.subheader("Latest FRED Macro Readings")
            if macro_df is not None and not macro_df.empty:
                latest_macro = macro_df.ffill().tail(1).T.reset_index()
                latest_macro.columns = ["FRED Series", "Latest Value"]
                st.dataframe(latest_macro, width="stretch", hide_index=True)
            else:
                st.info("FRED macro endpoint unavailable; using market macro proxies (^TNX, ^IRX, HYG/IEF, UUP).")

        with col_n:
            st.subheader("Live News NLP Sentiment Feed (GDELT / RSS / SEC 8-K / Yahoo)")
            if scored_news is not None and not scored_news.empty:
                st.dataframe(
                    scored_news[["published_at", "ticker", "feed", "source", "sentiment_label", "sentiment_score", "title"]].head(35),
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.info("No news headlines retrieved in current session.")

    # ------------------------------------------------------------------ TAB 6: PAPER TRACKER
    with tabs[5]:
        st.subheader("Real-Time Forward Paper-Trading & Verification Ledger")
        st.caption(
            "Logs closed-bar predictions idempotently per (ticker, horizon, bar) and evaluates matured signals "
            "against actual settled future bars with conservative same-bar Stop-Loss priority and round-trip costs."
        )
        ledger = PaperLedger()

        p_col1, p_col2, p_col3 = st.columns([2, 1, 1])
        with p_col1:
            log_horizons = st.multiselect(
                "Horizons to Log",
                list(HORIZON_CONFIGS.keys()),
                default=["next_day", "swing"],
            )
        with p_col2:
            log_btn = st.button("Log Closed-Bar Signals (Universe)", width="stretch")
        with p_col3:
            eval_btn = st.button("Evaluate Matured Trades (Live)", width="stretch")

        if log_btn and log_horizons:
            with st.spinner("Generating & logging causal closed-bar signals..."):
                events = log_paper_signals(
                    tickers=UNIVERSES[universe_name],
                    horizons=log_horizons,
                    universe_name=universe_name,
                    use_cache=False,
                    ledger=ledger,
                )
            st.dataframe(pd.DataFrame(events), width="stretch", hide_index=True)

        if eval_btn:
            with st.spinner("Checking settled future bars against OPEN paper trades..."):
                ev_res = ledger.evaluate_open_predictions()
            st.success(
                f"Checked Open: {ev_res.checked_open} | Newly Matured: {ev_res.newly_matured} | Still Open: {ev_res.still_open}"
            )

        summary = ledger.get_summary()
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Total Logged", summary["total_logged"], f"Open: {summary['open_count']}")
        k2.metric("Matured Calls", summary["matured_count"], f"Active: {summary['active_matured_count']}")
        k3.metric("Live Hit Rate", f"{summary['hit_rate_pct']}%" if summary["hit_rate_pct"] is not None else "Pending")
        k4.metric("Cum Hold P&L", f"{summary['cum_alloc_hold_pnl_pct']:+.2f}%")
        k5.metric("Cum Risk-Mgmt P&L", f"{summary['cum_alloc_risk_pnl_pct']:+.2f}%", f"TP {summary['tp_count']} / SL {summary['sl_count']}")
        k6.metric("Live Spearman IC", f"{summary['live_spearman_ic']}" if summary["live_spearman_ic"] is not None else "Pending (<5)")

        open_mtm = ledger.get_open_mtm()
        if not open_mtm.empty:
            st.subheader("Open Forward Trades (Unrealized Mark-to-Market)")
            st.dataframe(open_mtm, width="stretch", hide_index=True)

        all_paper = ledger.load_all_predictions()
        matured_df = all_paper[all_paper["status"] == "MATURED"]
        if not matured_df.empty:
            st.subheader("Matured Paper Trades (Settled Ground Truth)")
            disp_cols = [
                "ticker", "horizon_name", "as_of_timestamp", "exit_timestamp", "verdict",
                "entry_price", "exit_price", "actual_asset_ret_pct", "net_strategy_ret_pct",
                "risk_managed_ret_pct", "allocated_hold_pnl_pct", "allocated_risk_pnl_pct",
                "barrier_outcome", "mfe_pct", "mae_pct",
            ]
            st.dataframe(matured_df[disp_cols], width="stretch", hide_index=True)

            lb_live = ledger.get_model_accuracy_leaderboard()
            if not lb_live.empty:
                st.subheader("Live Forward Model Accuracy Leaderboard")
                st.dataframe(lb_live, width="stretch", hide_index=True)
        elif summary["total_logged"] == 0:
            st.info("No paper predictions logged yet. Click 'Log Closed-Bar Signals (Universe)' or run `paper-log` in CLI.")


if __name__ == "__main__":
    main()
