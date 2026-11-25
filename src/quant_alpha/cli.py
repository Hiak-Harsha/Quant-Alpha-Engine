"""Command-line interface for feed validation, walk-forward training, backtesting, and prediction."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import List

import pandas as pd

from quant_alpha.backtest.engine import run_backtest_comparison
from quant_alpha.config import (
    GLOBAL_DRIVERS,
    HORIZON_CONFIGS,
    REPORTS_DIR,
    UNIVERSES,
    EngineSettings,
    ensure_directories,
)
from quant_alpha.data.macro_feed import load_all_macro_series
from quant_alpha.data.market_feed import fetch_universe_ohlcv
from quant_alpha.data.news_feed import fetch_all_news
from quant_alpha.data.validator import run_full_feed_audit
from quant_alpha.features.pipeline import build_universe_dataset
from quant_alpha.features.sentiment import score_news_dataframe
from quant_alpha.models.ensemble import load_ensemble_artifacts, train_and_evaluate_walkforward
from quant_alpha.paper.ledger import PaperLedger, log_paper_signals
from quant_alpha.predictor import (
    predict_asset,
    predict_multi_horizon_matrix,
    scan_market_predictions,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("quant_alpha")


def cmd_validate_feeds(horizon: str, universe: str, refresh: bool) -> int:
    ensure_directories()
    hcfg = HORIZON_CONFIGS[horizon]
    tickers = UNIVERSES.get(universe, UNIVERSES["stepping_stone"])
    all_tickers = list(dict.fromkeys(tickers + GLOBAL_DRIVERS))
    ohlcv_map, errors = fetch_universe_ohlcv(
        all_tickers, period=hcfg.period, interval=hcfg.interval, use_cache=not refresh
    )
    macro_df = load_all_macro_series(use_cache=not refresh)
    news_df = fetch_all_news(tickers, use_cache=not refresh)
    audit = run_full_feed_audit(ohlcv_map, macro_df, news_df, driver_tickers=set(GLOBAL_DRIVERS))

    print("\n=== FEED VALIDATION AUDIT ===")
    print(audit.to_string(index=False))
    if errors:
        print("\nFeed Errors:", errors)
    audit.to_csv(REPORTS_DIR / f"feed_audit_{horizon}.csv", index=False)
    return 0


def cmd_train_backtest(
    horizon: str,
    universe: str,
    tune: bool,
    trials: int,
    refresh: bool,
) -> int:
    ensure_directories()
    hcfg = HORIZON_CONFIGS[horizon]
    settings = EngineSettings(n_optuna_trials=trials)
    tickers = UNIVERSES.get(universe, UNIVERSES["stepping_stone"])
    all_tickers = list(dict.fromkeys(tickers + GLOBAL_DRIVERS))

    logger.info("Fetching %s data (%s, %s) for %d tickers...", horizon, hcfg.interval, hcfg.period, len(all_tickers))
    ohlcv_map, errors = fetch_universe_ohlcv(
        all_tickers, period=hcfg.period, interval=hcfg.interval, use_cache=not refresh, settings=settings
    )
    if errors:
        logger.warning("Some feeds failed/skipped: %s", errors)

    macro_df = load_all_macro_series(use_cache=not refresh, settings=settings)
    dataset = build_universe_dataset(tickers, ohlcv_map, macro_df, hcfg)
    logger.info("Built panel dataset: %d rows, %d causal features across %d target assets.",
                len(dataset.panel_df), len(dataset.feature_cols), dataset.panel_df["ticker"].nunique())

    ens_res = train_and_evaluate_walkforward(
        dataset=dataset,
        horizon_cfg=hcfg,
        tune=tune,
        n_optuna_trials=trials,
        random_state=settings.random_state,
        save_artifacts=True,
    )
    bt = run_backtest_comparison(ens_res.oof_predictions, ens_res.model_names, hcfg, settings)

    print(f"\n=== WALK-FORWARD BACKTEST LEADERBOARD ({hcfg.display_name}) ===")
    cols = [
        "strategy", "spearman_ic", "hit_rate_pct", "roc_auc",
        "total_return_pct", "cagr_pct", "sharpe", "sortino",
        "max_drawdown_pct", "p_value", "edge_verdict",
    ]
    print(bt.leaderboard[cols].to_string(index=False))

    print("\n=== LEARNED ENSEMBLE WEIGHTS ===")
    for m, w in ens_res.ensemble_weights.items():
        print(f"  {m:18s}: {w*100:.1f}%")

    print("\n=== PER-ASSET OUT-OF-SAMPLE SUMMARY (ENSEMBLE) ===")
    print(bt.per_asset_summary.to_string(index=False))

    if not ens_res.shap_importance.empty:
        print("\n=== TOP 12 GLOBAL DRIVERS (SHAP) ===")
        print(ens_res.shap_importance.head(12).to_string(index=False))

    bt.leaderboard.to_csv(REPORTS_DIR / f"leaderboard_{horizon}.csv", index=False)
    bt.per_asset_summary.to_csv(REPORTS_DIR / f"per_asset_{horizon}.csv", index=False)
    return 0


def cmd_predict(ticker: str, horizon: str, universe: str, all_horizons: bool) -> int:
    ensure_directories()
    if all_horizons:
        reports = predict_multi_horizon_matrix(ticker=ticker, universe_name=universe)
    else:
        reports = [predict_asset(ticker=ticker, horizon_name=horizon, universe_name=universe)]

    for r in reports:
        print("\n" + "=" * 72)
        print(f"OBJECTIVE PREDICTION: {r.ticker} | {r.horizon_display} | As of {r.as_of_timestamp}")
        print("=" * 72)
        print(f"Last Close        : {r.last_close}")
        print(f"Verdict           : {r.verdict}")
        print(f"Quant Expected Ret: {r.quant_expected_ret_pct:+.2f}%  | News+Quant Composite: {r.composite_expected_ret_pct:+.2f}%")
        print(f"1-Sigma Range     : [{r.expected_range_low_pct:+.2f}%, {r.expected_range_high_pct:+.2f}%]  (Cost Hurdle: {r.cost_hurdle_pct:.2f}%)")
        print(f"Upside Probability: {r.prob_up_pct:.1f}%  | Model Consensus: {r.model_agreement_pct:.1f}% (Dispersion: {r.model_disagreement_std:.3f})")
        print(f"Volatility (GARCH): {r.garch_ann_vol_pct:.2f}% (21-bar RV: {r.realized_vol21_pct:.2f}% | VIX: {r.vix_level})")
        print(f"HMM Market Regime : {r.regime_label} -> {r.regime_probs}")
        print(f"Suggested Sizing  : {r.suggested_allocation_pct:+.1f}% | Stop: {r.stop_loss_price} | Target: {r.take_profit_price}")
        print(
            f"News Sentiment    : Composite={r.news_summary['composite_sentiment']:+.3f} "
            f"(Asset={r.news_summary['asset_sentiment']:+.3f} [{r.news_summary['asset_article_count']}], "
            f"Macro={r.news_summary['macro_sentiment']:+.3f} [{r.news_summary['macro_article_count']}])"
        )
        print("\nModel Breakdown:")
        print(pd.DataFrame(r.model_breakdown).to_string(index=False))
        if r.top_drivers:
            print("\nTop Local SHAP Drivers:")
            print(pd.DataFrame(r.top_drivers).to_string(index=False))
        if r.backtest_track_record:
            tr = r.backtest_track_record
            print(
                f"\nOOS Backtest Track Record ({r.horizon_name}): "
                f"Sharpe={tr.get('sharpe')}, IC={tr.get('spearman_ic')}, "
                f"AUC={tr.get('roc_auc')}, MaxDD={tr.get('max_drawdown_pct')}%, "
                f"p={tr.get('p_value')} ({tr.get('edge_verdict')})"
            )
    return 0


def cmd_paper_log(universe: str, horizons_csv: str, ticker: str | None, refresh: bool) -> int:
    ensure_directories()
    horizons = [h.strip() for h in horizons_csv.split(",") if h.strip() in HORIZON_CONFIGS]
    if not horizons:
        horizons = ["swing"]
    tickers = [ticker.strip().upper()] if ticker else UNIVERSES.get(universe, UNIVERSES["stepping_stone"])
    events = log_paper_signals(
        tickers=tickers,
        horizons=horizons,
        universe_name=universe,
        use_cache=not refresh,
    )
    print("\n=== REAL-TIME PAPER SIGNAL LOGGING (CLOSED-BAR CAUSAL) ===")
    print(pd.DataFrame(events).to_string(index=False))
    return 0


def cmd_paper_eval() -> int:
    ensure_directories()
    ledger = PaperLedger()
    res = ledger.evaluate_open_predictions()
    print(
        f"\n=== PAPER EVALUATION COMPLETE ===\n"
        f"Checked Open: {res.checked_open} | Newly Matured: {res.newly_matured} | Still Open: {res.still_open}"
    )
    if res.errors:
        print("Errors:", res.errors)
    return cmd_paper_report()


def cmd_paper_report() -> int:
    ensure_directories()
    ledger = PaperLedger()
    summary = ledger.get_summary()
    print("\n=== REAL-TIME FORWARD PAPER-TRADING SCORECARD ===")
    for k, v in summary.items():
        print(f"  {k:26s}: {v}")

    open_mtm = ledger.get_open_mtm()
    if not open_mtm.empty:
        print("\n=== OPEN POSITIONS (LIVE MARK-TO-MARKET) ===")
        print(open_mtm.to_string(index=False))

    all_df = ledger.load_all_predictions()
    matured = all_df[all_df["status"] == "MATURED"]
    if not matured.empty:
        cols = [
            "ticker", "horizon_name", "as_of_timestamp", "exit_timestamp", "verdict",
            "entry_price", "exit_price", "actual_asset_ret_pct",
            "net_strategy_ret_pct", "risk_managed_ret_pct", "barrier_outcome",
            "mfe_pct", "mae_pct",
        ]
        print("\n=== RECENT MATURED TRADES ===")
        print(matured[cols].head(20).to_string(index=False))

        lb = ledger.get_model_accuracy_leaderboard()
        if not lb.empty:
            print("\n=== LIVE FORWARD MODEL LEADERBOARD ===")
            print(lb.to_string(index=False))
    return 0


def cmd_scan(horizon: str, universe: str, include_trending: bool, max_trending: int, refresh: bool) -> int:
    ensure_directories()
    out = scan_market_predictions(
        horizon_name=horizon,
        universe_name=universe,
        include_trending=include_trending,
        max_trending_extras=max_trending,
        use_cache=not refresh,
    )
    print(f"\n=== ACTIVE TOP & TRENDING STOCK PREDICTIONS ({horizon.upper()} | {len(out.scan_df)} ASSETS) ===")
    if not out.scan_df.empty:
        cols = [
            "ticker", "market", "buzz_score", "news_mentions", "close",
            "verdict", "composite_ret_pct", "prob_up_pct", "consensus_pct",
            "garch_vol_pct", "alloc_pct", "stop_loss", "take_profit",
        ]
        print(out.scan_df[cols].to_string(index=False))
        out.scan_df.to_csv(REPORTS_DIR / f"market_scan_{horizon}.csv", index=False)
    if out.skipped_tickers:
        print(f"\nSkipped ({len(out.skipped_tickers)} tickers lacking history):", out.skipped_tickers)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quant Alpha Engine CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_val = sub.add_parser("validate-feeds", help="Fetch and validate all OHLCV, macro, and news feeds.")
    p_val.add_argument("--horizon", choices=list(HORIZON_CONFIGS.keys()), default="swing")
    p_val.add_argument("--universe", choices=list(UNIVERSES.keys()), default="top_talked_leaders")
    p_val.add_argument("--refresh", action="store_true")

    p_tr = sub.add_parser("train-backtest", help="Run purged walk-forward training, Optuna tuning, and backtest.")
    p_tr.add_argument("--horizon", choices=list(HORIZON_CONFIGS.keys()), default="swing")
    p_tr.add_argument("--universe", choices=list(UNIVERSES.keys()), default="top_talked_leaders")
    p_tr.add_argument("--no-tune", action="store_true")
    p_tr.add_argument("--trials", type=int, default=12)
    p_tr.add_argument("--refresh", action="store_true")

    p_pr = sub.add_parser("predict", help="Generate multi-model objective prediction for a ticker.")
    p_pr.add_argument("--ticker", type=str, default="NVDA")
    p_pr.add_argument("--horizon", choices=list(HORIZON_CONFIGS.keys()), default="swing")
    p_pr.add_argument("--universe", choices=list(UNIVERSES.keys()), default="top_talked_leaders")
    p_pr.add_argument("--all-horizons", action="store_true")

    p_sc = sub.add_parser("scan", help="Scan & rank active predictions across top & trending stocks.")
    p_sc.add_argument("--horizon", choices=list(HORIZON_CONFIGS.keys()), default="swing")
    p_sc.add_argument("--universe", choices=list(UNIVERSES.keys()), default="top_talked_leaders")
    p_sc.add_argument("--no-trending", action="store_true")
    p_sc.add_argument("--max-trending", type=int, default=15)
    p_sc.add_argument("--refresh", action="store_true")

    p_plog = sub.add_parser("paper-log", help="Log live closed-bar predictions into the forward paper ledger.")
    p_plog.add_argument("--universe", choices=list(UNIVERSES.keys()), default="top_talked_leaders")
    p_plog.add_argument("--horizons", type=str, default="swing,intraday")
    p_plog.add_argument("--ticker", type=str, default=None)
    p_plog.add_argument("--refresh", action="store_true")

    sub.add_parser("paper-eval", help="Grade open paper predictions once horizon bars settle and print report.")
    sub.add_parser("paper-report", help="Print live forward paper-trading scorecard and model accuracy.")

    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "validate-feeds":
        return cmd_validate_feeds(args.horizon, args.universe, args.refresh)
    if args.command == "train-backtest":
        return cmd_train_backtest(args.horizon, args.universe, not args.no_tune, args.trials, args.refresh)
    if args.command == "predict":
        return cmd_predict(args.ticker, args.horizon, args.universe, args.all_horizons)
    if args.command == "scan":
        return cmd_scan(args.horizon, args.universe, not args.no_trending, args.max_trending, args.refresh)
    if args.command == "paper-log":
        return cmd_paper_log(args.universe, args.horizons, args.ticker, args.refresh)
    if args.command == "paper-eval":
        return cmd_paper_eval()
    if args.command == "paper-report":
        return cmd_paper_report()
    return 1


if __name__ == "__main__":
    sys.exit(main())
