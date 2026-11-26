"""Fast, cache-backed FastAPI REST API + SPA static server for Quant Alpha Terminal."""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from quant_alpha.backtest.engine import run_backtest_comparison
from quant_alpha.config import (
    ARTIFACTS_DIR,
    DATA_CACHE_DIR,
    GLOBAL_DRIVERS,
    HORIZON_CONFIGS,
    PROJECT_ROOT,
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
from quant_alpha.portfolio.construct import is_cash_long_only_asset
from quant_alpha.predictor import (
    load_scan_snapshot,
    predict_asset,
    predict_multi_horizon_matrix,
    scan_market_predictions,
)
from quant_alpha.ui.components import (
    build_metals_spotlight_df,
    build_signal_justification,
    build_upside_downside_watchlists,
    compute_capital_scenario,
    format_pulse_value,
    humanize_feature_name,
    summarize_gate_status,
)

WEB_DIST_DIR = PROJECT_ROOT / "web" / "dist"


def _clean_value(val: Any) -> Any:
    """Convert numpy/pandas scalars and NaN/Inf cleanly into JSON-safe Python primitives."""
    if val is None:
        return None
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (float, np.floating)):
        f = float(val)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(val, pd.Timestamp):
        return str(val)[:19]
    if isinstance(val, dict):
        return {str(k): _clean_value(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_clean_value(v) for v in val]
    return val


def _df_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    return [{str(k): _clean_value(v) for k, v in row.items()} for row in df.to_dict(orient="records")]


class _RuntimeState:
    """In-memory pin cache so repeated UI clicks execute in <20ms."""

    def __init__(self) -> None:
        self.bundles: Dict[str, Dict[str, Any]] = {}
        self.ohlcv_cache: Dict[str, Dict[str, pd.DataFrame]] = {}
        self.macro_df: Optional[pd.DataFrame] = None
        self.scored_news: Optional[pd.DataFrame] = None
        self.trending_df: Optional[pd.DataFrame] = None
        self.asset_reports: Dict[str, Any] = {}
        self.backtest_cache: Dict[str, Dict[str, Any]] = {}

    def get_bundle(self, horizon: str) -> Dict[str, Any]:
        if horizon not in HORIZON_CONFIGS:
            raise HTTPException(status_code=404, detail=f"Unknown horizon '{horizon}'.")
        if horizon not in self.bundles:
            try:
                self.bundles[horizon] = load_ensemble_artifacts(horizon)
            except FileNotFoundError as exc:
                raise HTTPException(
                    status_code=503,
                    detail=f"Model bundle for '{horizon}' has not been trained yet. Run CLI train-backtest first.",
                ) from exc
        return self.bundles[horizon]

    def get_market_inputs(
        self,
        horizon: str,
        universe: str,
        use_cache: bool = True,
    ) -> tuple[Dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        key = f"{horizon}:{universe}"
        if not use_cache:
            self.ohlcv_cache.pop(key, None)
            self.macro_df = None
            self.scored_news = None
            self.trending_df = None
            self.asset_reports.clear()

        hcfg = HORIZON_CONFIGS[horizon]
        base_tickers = list(UNIVERSES.get(universe, UNIVERSES["india_market_gold_silver"]))
        if key not in self.ohlcv_cache:
            fetch_list = list(dict.fromkeys(base_tickers + GLOBAL_DRIVERS))
            ohlcv_map, _ = fetch_universe_ohlcv(
                fetch_list,
                period=hcfg.period,
                interval=hcfg.interval,
                use_cache=use_cache,
                max_workers=10,
            )
            self.ohlcv_cache[key] = ohlcv_map

        if self.macro_df is None:
            self.macro_df = load_all_macro_series(use_cache=use_cache)
        if self.scored_news is None:
            self.scored_news = score_news_dataframe(fetch_all_news(base_tickers[:35], use_cache=use_cache))
        if self.trending_df is None:
            self.trending_df = discover_trending_stocks(
                news_df=self.scored_news,
                candidate_tickers=base_tickers,
                use_cache=use_cache,
                india_focus=universe.startswith("india_"),
            )
        return self.ohlcv_cache[key], self.macro_df, self.scored_news, self.trending_df


def _nse_market_status() -> Dict[str, Any]:
    now_ist = pd.Timestamp.now(tz="Asia/Kolkata")
    is_weekday = now_ist.day_of_week < 5
    in_hours = (now_ist.hour > 9 or (now_ist.hour == 9 and now_ist.minute >= 15)) and (
        now_ist.hour < 15 or (now_ist.hour == 15 and now_ist.minute <= 30)
    )
    is_open = bool(is_weekday and in_hours)
    return {
        "ist_time": now_ist.strftime("%Y-%m-%d %H:%M:%S IST"),
        "is_open": is_open,
        "status_label": "NSE OPEN (09:15-15:30 IST)" if is_open else "NSE CLOSED / AFTER-HOURS",
    }


class OrderRequest(BaseModel):
    account_id: str = "default_student"
    ticker: str
    side: str
    qty: float = Field(gt=0.0)
    price: float = Field(gt=0.0)
    horizon: str = "next_day"
    stop_loss: float = 0.0
    take_profit: float = 0.0
    signal_verdict: str = ""
    signal_ret_pct: float = 0.0


class AccountRequest(BaseModel):
    account_id: Optional[str] = None
    owner_name: str
    starting_cash: float = Field(default=100000.0, ge=1000.0)


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    configure_logging()
    ensure_directories()

    api = FastAPI(title="Quant Alpha Terminal API", version="1.0.0")
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    state = _RuntimeState()

    @api.get("/api/health")
    def health() -> Dict[str, Any]:
        trained = [h for h in HORIZON_CONFIGS if (ARTIFACTS_DIR / h / "bundle.joblib").exists()]
        return {
            "status": "ok",
            "market_session": _nse_market_status(),
            "trained_horizons": trained,
        }

    @api.get("/api/config")
    def get_config() -> Dict[str, Any]:
        return {
            "universes": {k: list(v) for k, v in UNIVERSES.items()},
            "horizons": {
                k: {
                    "name": v.name,
                    "display_name": v.display_name,
                    "interval": v.interval,
                    "forecast_horizon": v.forecast_horizon,
                    "description": v.description,
                }
                for k, v in HORIZON_CONFIGS.items()
            },
            "fee_schedule": asdict(DEFAULT_FEE_SCHEDULE),
            "market_session": _nse_market_status(),
        }

    @api.get("/api/scan")
    def get_scan(
        horizon: str = Query("next_day"),
        universe: str = Query("india_market_gold_silver"),
        refresh: bool = Query(False),
    ) -> Dict[str, Any]:
        if horizon not in HORIZON_CONFIGS:
            raise HTTPException(status_code=404, detail=f"Unknown horizon '{horizon}'.")
        if universe not in UNIVERSES:
            raise HTTPException(status_code=404, detail=f"Unknown universe '{universe}'.")

        bundle = state.get_bundle(horizon)
        ohlcv_map, macro_df, scored_news, trending_df = state.get_market_inputs(
            horizon, universe, use_cache=not refresh
        )

        if refresh:
            snap_file = DATA_CACHE_DIR / "snapshots" / f"{horizon}__{universe}.joblib"
            if snap_file.exists():
                snap_file.unlink()

        ttl = 0.5 if horizon == "intraday" else 6.0
        snap = None if refresh else load_scan_snapshot(horizon, universe, ttl_hours=ttl)
        if snap is not None:
            scan_df, skipped_map, horizon_tr = snap
        else:
            out = scan_market_predictions(
                horizon_name=horizon,
                universe_name=universe,
                include_trending=True,
                max_trending_extras=10,
                use_cache=not refresh,
                ohlcv_map=ohlcv_map,
                macro_df=macro_df,
                scored_news=scored_news,
                bundle=bundle,
            )
            scan_df, skipped_map, horizon_tr = out.scan_df, out.skipped_tickers, out.horizon_track_record

        pulse = []
        for pt in ("^NSEI", "^NSEBANK", "^INDIAVIX", "USDINR=X", "GOLDBEES.NS", "SILVERBEES.NS", "BTC-USD"):
            pdf = ohlcv_map.get(pt)
            if pdf is not None and len(pdf) >= 2:
                last_v = float(pdf["Close"].iloc[-1])
                chg_v = float(pdf["Close"].pct_change(1).iloc[-1] * 100.0)
                pulse.append(
                    {
                        "ticker": pt,
                        "value": _clean_value(last_v),
                        "formatted": format_pulse_value(pt, last_v),
                        "chg_pct": round(chg_v, 2),
                    }
                )

        gainers_df, decliners_df = build_upside_downside_watchlists(scan_df, top_n=12)
        metals_df = build_metals_spotlight_df(scan_df)
        gate_info = summarize_gate_status(horizon_tr)

        return _clean_value(
            {
                "horizon": horizon,
                "universe": universe,
                "market_session": _nse_market_status(),
                "gate_info": gate_info,
                "pulse": pulse,
                "gainers": _df_records(gainers_df),
                "decliners": _df_records(decliners_df),
                "metals": _df_records(metals_df),
                "scan_rows": _df_records(scan_df),
                "trending": _df_records(trending_df.head(25)) if not trending_df.empty else [],
                "skipped_count": len(skipped_map),
                "skipped": skipped_map,
            }
        )

    @api.get("/api/asset/{ticker}")
    def get_asset_detail(
        ticker: str,
        horizon: str = Query("next_day"),
        universe: str = Query("india_market_gold_silver"),
        capital: float = Query(100000.0),
    ) -> Dict[str, Any]:
        t_clean = ticker.strip().upper()
        bundle = state.get_bundle(horizon)
        ohlcv_map, macro_df, scored_news, _ = state.get_market_inputs(horizon, universe, use_cache=True)

        work_map = dict(ohlcv_map)
        hcfg = HORIZON_CONFIGS[horizon]
        if t_clean not in work_map:
            extra = fetch_ohlcv(t_clean, period=hcfg.period, interval=hcfg.interval)
            if extra.empty:
                raise HTTPException(status_code=404, detail=f"No price data available for '{t_clean}'.")
            work_map[t_clean] = extra

        t_df = work_map[t_clean]
        bar_key = f"{t_clean}:{horizon}:{universe}:{t_df.index[-1]}"
        if bar_key not in state.asset_reports:
            try:
                state.asset_reports[bar_key] = predict_asset(
                    ticker=t_clean,
                    horizon_name=horizon,
                    universe_name=universe,
                    ohlcv_map=work_map,
                    macro_df=macro_df,
                    scored_news=scored_news,
                    bundle=bundle,
                    compute_shap=True,
                )
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        report = state.asset_reports[bar_key]
        cards = build_signal_justification(report, position_capital=capital)
        scenario = compute_capital_scenario(report, capital)
        drivers = [
            {**d, "label": humanize_feature_name(str(d["feature"]))}
            for d in report.top_drivers
        ]

        hist = t_df.tail(90).copy()
        history = [
            {
                "date": str(idx)[:10] if hcfg.interval in {"1d", "1wk", "1mo"} else str(idx)[:16],
                "close": round(float(row["Close"]), 4),
                "high": round(float(row["High"]), 4),
                "low": round(float(row["Low"]), 4),
            }
            for idx, row in hist.iterrows()
        ]

        return _clean_value(
            {
                "report": asdict(report),
                "cash_long_only": is_cash_long_only_asset(t_clean),
                "justification": cards,
                "scenario": scenario,
                "drivers": drivers,
                "history": history,
            }
        )

    @api.get("/api/asset/{ticker}/horizons")
    def get_asset_horizons(
        ticker: str,
        universe: str = Query("india_market_gold_silver"),
    ) -> Dict[str, Any]:
        t_clean = ticker.strip().upper()
        try:
            reports = predict_multi_horizon_matrix(ticker=t_clean, universe_name=universe)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _clean_value(
            {
                "ticker": t_clean,
                "rows": [
                    {
                        "horizon_name": r.horizon_name,
                        "horizon": r.horizon_display,
                        "verdict": r.verdict,
                        "raw_verdict": r.raw_verdict,
                        "composite_ret_pct": r.composite_expected_ret_pct,
                        "prob_up_pct": r.prob_up_pct,
                        "consensus_pct": r.model_agreement_pct,
                        "garch_vol_pct": r.garch_ann_vol_pct,
                        "regime": r.regime_label,
                        "alloc_pct": r.suggested_allocation_pct,
                        "oos_sharpe": r.backtest_track_record.get("sharpe"),
                    }
                    for r in reports
                ],
            }
        )

    @api.get("/api/backtest")
    def get_backtest(horizon: str = Query("next_day")) -> Dict[str, Any]:
        bundle = state.get_bundle(horizon)
        if horizon in state.backtest_cache:
            return state.backtest_cache[horizon]

        hdir = ARTIFACTS_DIR / horizon
        lb_p = hdir / "leaderboard.parquet"
        eq_p = hdir / "equity_curves.parquet"
        dd_p = hdir / "drawdown_curves.parquet"
        as_p = hdir / "per_asset_summary.parquet"

        if lb_p.exists() and eq_p.exists() and dd_p.exists() and as_p.exists():
            lb_df = pd.read_parquet(lb_p)
            eq_df = pd.read_parquet(eq_p)
            dd_df = pd.read_parquet(dd_p)
            as_df = pd.read_parquet(as_p)
        else:
            oof_df = bundle.get("oof_predictions")
            if oof_df is None or oof_df.empty:
                raise HTTPException(status_code=404, detail=f"No OOF backtest predictions for '{horizon}'.")
            bt = run_backtest_comparison(oof_df, list(bundle["model_names"]), HORIZON_CONFIGS[horizon], EngineSettings(bootstrap_samples=150))
            lb_df, eq_df, dd_df, as_df = bt.leaderboard, bt.equity_curves, bt.drawdown_curves, bt.per_asset_summary

        # Sample equity/drawdown series cleanly (~250 points max) for smooth interactive charts
        step = max(1, len(eq_df) // 250)
        eq_sub = eq_df.iloc[::step].copy()
        dd_sub = dd_df.iloc[::step].copy()
        eq_sub["date"] = eq_sub.index.astype(str).str[:10]
        dd_sub["date"] = dd_sub.index.astype(str).str[:10]

        shap_df = bundle.get("shap_importance", pd.DataFrame())
        shap_rows = []
        if shap_df is not None and not shap_df.empty:
            for _, row in shap_df.head(15).iterrows():
                shap_rows.append(
                    {
                        "feature": str(row["feature"]),
                        "label": humanize_feature_name(str(row["feature"])),
                        "mean_abs_shap": round(float(row["mean_abs_shap"]), 5),
                    }
                )

        payload = _clean_value(
            {
                "horizon": horizon,
                "display_name": HORIZON_CONFIGS[horizon].display_name,
                "leaderboard": _df_records(lb_df),
                "equity_curves": _df_records(eq_sub),
                "drawdown_curves": _df_records(dd_sub),
                "per_asset": _df_records(as_df),
                "global_shap": shap_rows,
                "ensemble_weights": bundle.get("ensemble_weights", {}),
            }
        )
        state.backtest_cache[horizon] = payload
        return payload

    @api.get("/api/feeds/audit")
    def get_feeds_audit(
        horizon: str = Query("next_day"),
        universe: str = Query("india_market_gold_silver"),
    ) -> Dict[str, Any]:
        ohlcv_map, macro_df, scored_news, _ = state.get_market_inputs(horizon, universe, use_cache=True)
        audit_df = run_full_feed_audit(ohlcv_map, macro_df, scored_news, driver_tickers=set(GLOBAL_DRIVERS))
        return _clean_value(
            {
                "audit": _df_records(audit_df),
                "news": _df_records(
                    scored_news[["published_at", "ticker", "feed", "source", "sentiment_label", "sentiment_score", "title"]].head(30)
                )
                if scored_news is not None and not scored_news.empty
                else [],
            }
        )

    # ---------------- Paper Trading Simulator & Daily Correlation Endpoints ----------------

    @api.get("/api/paper/accounts")
    def list_paper_accounts() -> Dict[str, Any]:
        sim = PaperBrokerSimulator()
        return {"accounts": _clean_value(sim.list_accounts())}

    @api.post("/api/paper/accounts")
    def create_paper_account(req: AccountRequest) -> Dict[str, Any]:
        sim = PaperBrokerSimulator()
        acc_id = (req.account_id or req.owner_name.lower().replace(" ", "_")).strip()
        if not acc_id:
            raise HTTPException(status_code=400, detail="Invalid account name.")
        sim.ensure_account(acc_id, req.owner_name, starting_cash=req.starting_cash)
        return {"account": _clean_value(sim.get_account(acc_id)), "accounts": _clean_value(sim.list_accounts())}

    @api.post("/api/paper/accounts/{account_id}/reset")
    def reset_paper_account(account_id: str, starting_cash: float = Query(100000.0, ge=1000.0)) -> Dict[str, Any]:
        sim = PaperBrokerSimulator()
        sim.reset_account(account_id, starting_cash=starting_cash)
        return {"account": _clean_value(sim.get_account(account_id))}

    @api.get("/api/paper/portfolio")
    def get_paper_portfolio(
        account_id: str = Query("default_student"),
        horizon: str = Query("next_day"),
        universe: str = Query("india_market_gold_silver"),
    ) -> Dict[str, Any]:
        sim = PaperBrokerSimulator()
        ohlcv_map, _, _, _ = state.get_market_inputs(horizon, universe, use_cache=True)
        price_map = {
            t: float(df["Close"].iloc[-1])
            for t, df in ohlcv_map.items()
            if df is not None and not df.empty
        }
        try:
            snap = sim.get_portfolio_snapshot(account_id, price_map=price_map)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return _clean_value(
            {
                "account": snap["account"],
                "cash_balance": snap["cash_balance"],
                "invested_cost": snap["invested_cost"],
                "market_value": snap["market_value"],
                "total_equity": snap["total_equity"],
                "unrealized_pnl": snap["unrealized_pnl"],
                "realized_pnl": snap["realized_pnl"],
                "total_fees_paid": snap["total_fees_paid"],
                "net_return_pct": snap["net_return_pct"],
                "positions": _df_records(snap["positions_df"]),
                "orders": _df_records(snap["orders_df"]),
            }
        )

    @api.post("/api/paper/preview")
    def preview_paper_order(req: OrderRequest) -> Dict[str, Any]:
        sim = PaperBrokerSimulator()
        try:
            prev = sim.preview_order(req.account_id, req.ticker, req.side, req.qty, req.price)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return _clean_value(asdict(prev))

    @api.post("/api/paper/order")
    def execute_paper_order(req: OrderRequest) -> Dict[str, Any]:
        sim = PaperBrokerSimulator()
        ok, msg = sim.execute_order(
            account_id=req.account_id,
            ticker=req.ticker,
            side=req.side,
            qty=req.qty,
            fill_price=req.price,
            stop_loss=req.stop_loss,
            take_profit=req.take_profit,
            signal_verdict=req.signal_verdict,
            signal_ret_pct=req.signal_ret_pct,
            notes=f"horizon={req.horizon}",
        )
        if not ok:
            raise HTTPException(status_code=400, detail=msg)
        return {"ok": True, "message": msg}

    @api.post("/api/paper/log-signals")
    def api_log_signals(
        horizon: str = Query("next_day"),
        universe: str = Query("india_market_gold_silver"),
    ) -> Dict[str, Any]:
        ledger = PaperLedger()
        events = log_paper_signals(
            tickers=UNIVERSES.get(universe, UNIVERSES["india_market_gold_silver"]),
            horizons=[horizon],
            universe_name=universe,
            use_cache=True,
            ledger=ledger,
        )
        return {"count": len(events), "events": _clean_value(events)}

    @api.post("/api/paper/evaluate")
    def api_evaluate_signals() -> Dict[str, Any]:
        ledger = PaperLedger()
        res = ledger.evaluate_open_predictions()
        return _clean_value(asdict(res))

    @api.get("/api/paper/correlation")
    def api_paper_correlation() -> Dict[str, Any]:
        ledger = PaperLedger()
        stats = compute_prediction_market_correlation(ledger)
        matured_df = stats["df"]
        points = (
            _df_records(
                matured_df[
                    [
                        c
                        for c in (
                            "ticker",
                            "horizon_name",
                            "as_of_timestamp",
                            "verdict",
                            "composite_expected_ret_pct",
                            "actual_asset_ret_pct",
                        )
                        if c in matured_df.columns
                    ]
                ]
            )
            if not matured_df.empty
            else []
        )
        summary = ledger.get_summary()
        return _clean_value(
            {
                "n_matured": stats["n_matured"],
                "pearson_r": stats["pearson_r"],
                "spearman_ic": stats["spearman_ic"],
                "hit_rate_pct": stats["hit_rate_pct"],
                "always_long_pct": stats["always_long_pct"],
                "points": points,
                "ledger_summary": summary,
            }
        )

    # Serve production React SPA if built in web/dist
    if WEB_DIST_DIR.exists():
        assets_dir = WEB_DIST_DIR / "assets"
        if assets_dir.exists():
            api.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

        @api.get("/{full_path:path}")
        def serve_spa(full_path: str) -> FileResponse:
            target = (WEB_DIST_DIR / full_path).resolve()
            if full_path and target.is_file() and str(target).startswith(str(WEB_DIST_DIR.resolve())):
                return FileResponse(target)
            return FileResponse(WEB_DIST_DIR / "index.html")

    return api


app = create_app()
