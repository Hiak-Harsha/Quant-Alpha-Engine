"""Interactive Streamlit dashboard for Quant Alpha Engine."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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

st.set_page_config(
    page_title="Quant Alpha Engine | Multi-Horizon Trading & Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(ttl=1800, show_spinner=False)
def _cached_market_bundle(
    universe_name: str,
    horizon_name: str,
    extra_ticker: str,
    include_trending: bool,
    max_trending: int,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hcfg = HORIZON_CONFIGS[horizon_name]
    base_tickers = list(UNIVERSES[universe_name])
    macro_df = load_all_macro_series()
    news_tickers = list(dict.fromkeys(base_tickers[:30] + ([extra_ticker] if extra_ticker else [])))
    news_df = score_news_dataframe(fetch_all_news(news_tickers))
    trending_df = discover_trending_stocks(news_df=news_df, candidate_tickers=base_tickers)

    scan_tickers = list(base_tickers)
    if include_trending and not trending_df.empty:
        extras = [s for s in trending_df["symbol"].tolist() if s not in scan_tickers][:max_trending]
        scan_tickers.extend(extras)

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
) -> tuple[pd.DataFrame, dict]:
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
    return out.scan_df, out.skipped_tickers


@st.cache_data(ttl=1800, show_spinner=False)
def _cached_backtest(horizon_name: str, _oof_df: pd.DataFrame, model_names: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from quant_alpha.config import ARTIFACTS_DIR

    hdir = ARTIFACTS_DIR / horizon_name
    lb_p = hdir / "leaderboard.parquet"
    eq_p = hdir / "equity_curves.parquet"
    dd_p = hdir / "drawdown_curves.parquet"
    as_p = hdir / "per_asset_summary.parquet"
    if lb_p.exists() and eq_p.exists() and dd_p.exists() and as_p.exists():
        return (
            pd.read_parquet(lb_p),
            pd.read_parquet(eq_p),
            pd.read_parquet(dd_p),
            pd.read_parquet(as_p),
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


def main() -> None:
    ensure_directories()
    st.title("Quant Alpha Engine — Multi-Asset, Multi-Horizon Prediction System")
    st.caption(
        "Combines LightGBM, XGBoost, ExtraTrees, Regularized Ridge/Logistic, Neural MLP, Classical Quant Signals, "
        "Causal GARCH(1,1) Volatility, Gaussian HMM Regimes, Cross-Asset Correlations, FRED Macro & Live Financial News NLP."
    )

    with st.sidebar:
        st.header("Configuration")
        universe_name = st.selectbox("Asset Universe", list(UNIVERSES.keys()), index=0)
        horizon_name = st.selectbox(
            "Primary Trading Style / Horizon",
            list(HORIZON_CONFIGS.keys()),
            index=1,
            format_func=lambda k: HORIZON_CONFIGS[k].display_name,
        )
        include_trending = st.checkbox("Auto-Discover Live Trending & Most-Active Stocks", value=True)
        max_trending = st.slider("Extra Trending Stocks to Scan", min_value=0, max_value=25, value=15)
        default_tickers = UNIVERSES[universe_name]
        selected_ticker = st.selectbox("Single-Asset Deep Dive", default_tickers, index=0)
        custom_ticker = st.text_input("Or Custom Ticker (Yahoo symbol)", value="").strip().upper()
        ticker = custom_ticker or selected_ticker

        run_all_horizons = st.checkbox("Compute All-Horizons Consensus (Intraday + Swing + Long-Term)", value=False)
        tune_trials = st.slider("Optuna Tuning Trials (when retraining)", min_value=5, max_value=30, value=12)
        retrain_clicked = st.button("Train / Retune Active Horizon", type="primary")

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

    with st.spinner("Loading top stocks, live trending tickers, macro, volatility & news feeds..."):
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
            "Active Top-Stocks Scanner",
            "Single-Asset Deep Dive",
            "Walk-Forward Backtest & Leaderboard",
            "Cross-Asset Correlation & Volatility",
            "Feed Validation & News Radar",
            "Live Paper Tracker",
        ]
    )

    # ------------------------------------------------------------------ TAB 1: SCANNER
    with tabs[0]:
        st.subheader(f"Active Multi-Model Predictions: Top & Most Talked-About Stocks ({HORIZON_CONFIGS[horizon_name].display_name})")
        scan_df, skipped_map = _cached_market_scan(
            horizon_name,
            universe_name,
            include_trending,
            max_trending,
            ohlcv_map,
            macro_df,
            scored_news,
            bundle,
        )
        if scan_df.empty:
            st.warning("No predictions generated across scanned tickers.")
        else:
            bull_n = int(scan_df["verdict"].str.contains("BULLISH").sum())
            bear_n = int(scan_df["verdict"].str.contains("BEARISH").sum())
            neut_n = int((scan_df["verdict"] == "NEUTRAL / HOLD").sum())
            trend_n = int((scan_df["buzz_score"] > 0).sum())

            s1, s2, s3, s4, s5 = st.columns(5)
            s1.metric("Stocks Predicted", len(scan_df), f"Skipped: {len(skipped_map)}")
            s2.metric("Bullish Calls", bull_n)
            s3.metric("Bearish Calls", bear_n)
            s4.metric("Neutral / Below Hurdle", neut_n)
            s5.metric("Trending / Buzzed", trend_n, f"Sources: {len(trending_df)} total")

            f_col1, f_col2, f_col3 = st.columns([1.2, 1.2, 1.3])
            with f_col1:
                mkt_filter = st.multiselect(
                    "Filter Market",
                    sorted(scan_df["market"].unique().tolist()),
                    default=sorted(scan_df["market"].unique().tolist()),
                )
            with f_col2:
                sig_filter = st.selectbox(
                    "Signal Filter",
                    ["All Assets", "Actionable Only (Bullish/Bearish)", "Bullish Only", "Bearish Only", "Trending / Buzzed Only"],
                    index=0,
                )
            with f_col3:
                log_top_btn = st.button("Log Top 12 Actionable Picks to Paper Ledger", width="stretch")

            view_df = scan_df[scan_df["market"].isin(mkt_filter)].copy()
            if sig_filter == "Actionable Only (Bullish/Bearish)":
                view_df = view_df[view_df["verdict"] != "NEUTRAL / HOLD"]
            elif sig_filter == "Bullish Only":
                view_df = view_df[view_df["verdict"].str.contains("BULLISH")]
            elif sig_filter == "Bearish Only":
                view_df = view_df[view_df["verdict"].str.contains("BEARISH")]
            elif sig_filter == "Trending / Buzzed Only":
                view_df = view_df[view_df["buzz_score"] > 0]

            if log_top_btn:
                actionable_tickers = scan_df[scan_df["verdict"] != "NEUTRAL / HOLD"]["ticker"].head(12).tolist()
                if actionable_tickers:
                    with st.spinner("Logging closed-bar signals for top actionable stocks..."):
                        events = log_paper_signals(
                            tickers=actionable_tickers,
                            horizons=[horizon_name],
                            universe_name=universe_name,
                            use_cache=True,
                        )
                    st.success(f"Logged/checked {len(events)} closed-bar signals in Live Paper Tracker.")

            st.dataframe(view_df, width="stretch", hide_index=True)

            actionable_plot = view_df[view_df["verdict"] != "NEUTRAL / HOLD"].head(16)
            if not actionable_plot.empty:
                fig_scan = px.bar(
                    actionable_plot.sort_values("composite_ret_pct"),
                    x="composite_ret_pct",
                    y="ticker",
                    orientation="h",
                    color="verdict",
                    color_discrete_map={
                        "STRONG BULLISH": "#15803d",
                        "LEAN BULLISH": "#22c55e",
                        "LEAN BEARISH": "#f87171",
                        "STRONG BEARISH": "#b91c1c",
                    },
                    title="Top High-Conviction Composite Expected Return (%)",
                )
                st.plotly_chart(fig_scan, width="stretch")

            with st.expander("Live Trending & Most-Talked-About Discovery Feed (Yahoo Trending + Most Active + Gainers/Losers + News Buzz)"):
                if not trending_df.empty:
                    st.dataframe(trending_df.head(30), width="stretch", hide_index=True)
                if skipped_map:
                    st.caption(f"Skipped newly-listed/sparse tickers (<70 bars): {skipped_map}")

    # ------------------------------------------------------------------ TAB 2: SINGLE ASSET
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
            st.error(f"Prediction failed for {ticker}: {exc}")

        if report is not None:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Verdict", report.verdict, f"Close: {report.last_close}")
            c2.metric(
                "Composite Expected Return",
                f"{report.composite_expected_ret_pct:+.2f}%",
                f"Pure Quant: {report.quant_expected_ret_pct:+.2f}%",
            )
            c3.metric("Upside Probability", f"{report.prob_up_pct:.1f}%", f"Consensus: {report.model_agreement_pct:.0f}%")
            c4.metric(
                "GARCH Annual Vol",
                f"{report.garch_ann_vol_pct:.1f}%",
                f"21d RV: {report.realized_vol21_pct:.1f}%",
            )
            c5.metric(
                "Vol-Targeted Allocation",
                f"{report.suggested_allocation_pct:+.1f}%",
                f"Stop {report.stop_loss_price} / TP {report.take_profit_price}",
            )

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
                    f"Sharpe = {tr.get('sharpe')} (95% CI [{tr.get('sharpe_ci_low')}, {tr.get('sharpe_ci_high')}]), "
                    f"Spearman IC = {tr.get('spearman_ic')}, ROC-AUC = {tr.get('roc_auc')}, "
                    f"Max Drawdown = {tr.get('max_drawdown_pct')}%, p-value = {tr.get('p_value')} -> **{tr.get('edge_verdict')}**"
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
                st.subheader("Multi-Horizon Consensus Matrix (Intraday + Swing + Long-Term)")
                with st.spinner("Evaluating Intraday, Swing, and Long-Term horizons..."):
                    try:
                        matrix_reports = predict_multi_horizon_matrix(
                            ticker=ticker, universe_name=universe_name, settings=settings
                        )
                        matrix_rows = [
                            {
                                "Horizon": r.horizon_display,
                                "Verdict": r.verdict,
                                "Quant Ret %": r.quant_expected_ret_pct,
                                "Composite Ret %": r.composite_expected_ret_pct,
                                "Prob Up %": r.prob_up_pct,
                                "Model Consensus %": r.model_agreement_pct,
                                "GARCH Vol %": r.garch_ann_vol_pct,
                                "Regime": r.regime_label,
                                "Alloc %": r.suggested_allocation_pct,
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
        corr_tickers = [t for t in UNIVERSES[universe_name] + ["^VIX", "UUP", "HYG", "IEF"] if t in ohlcv_map]
        ret_series = {}
        for t in dict.fromkeys(corr_tickers):
            ret_series[t] = ohlcv_map[t]["Close"].pct_change(1)
        corr_df = pd.DataFrame(ret_series).tail(252).corr().round(2)
        if not corr_df.empty:
            fig_corr = px.imshow(
                corr_df,
                text_auto=True,
                color_continuous_scale="RdBu",
                zmin=-1,
                zmax=1,
                title="Trailing 252-Bar Cross-Asset Return Correlations",
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
                corr_cols = [c for c in ("corr21_spy", "corr21_tlt", "corr21_gld", "corr21_btc", "corr21_vix") if c in cross_df.columns]
                if corr_cols:
                    fig_rc = px.line(cross_df[corr_cols], title=f"{ticker} Rolling 21-Bar Correlations to Global Drivers")
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
                default=["swing", "intraday"],
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
