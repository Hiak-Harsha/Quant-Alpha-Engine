"""Objective multi-model, volatility-aware, regime-, portfolio-, and gate-aware live prediction engine."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from quant_alpha.backtest.engine import run_backtest_comparison
from quant_alpha.backtest.metrics import evaluate_asset_pass_gate, evaluate_pass_gate_from_metrics
from quant_alpha.config import (
    ARTIFACTS_DIR,
    GLOBAL_DRIVERS,
    HORIZON_CONFIGS,
    UNIVERSES,
    EngineSettings,
    HorizonConfig,
    is_crypto_ticker,
)
from quant_alpha.data.macro_feed import load_all_macro_series
from quant_alpha.data.market_feed import fetch_ohlcv, fetch_universe_ohlcv
from quant_alpha.data.news_feed import fetch_all_news
from quant_alpha.data.trending_feed import discover_trending_stocks
from quant_alpha.features.pipeline import build_asset_feature_frame, build_universe_dataset
from quant_alpha.features.regime import REGIME_LABELS
from quant_alpha.features.sentiment import compute_news_sentiment_summary, score_news_dataframe
from quant_alpha.models.ensemble import load_ensemble_artifacts, train_and_evaluate_walkforward
from quant_alpha.models.zoo import MODEL_DISPLAY_NAMES
from quant_alpha.portfolio.construct import (
    apply_portfolio_constraints,
    compute_signal_raw_weights,
    is_cash_long_only_asset,
)

logger = logging.getLogger(__name__)


@dataclass
class LivePredictionReport:
    """Comprehensive objective prediction output for one ticker and horizon."""

    ticker: str
    horizon_name: str
    horizon_display: str
    as_of_timestamp: str
    last_close: float
    verdict: str
    quant_expected_ret_pct: float
    composite_expected_ret_pct: float
    expected_range_low_pct: float
    expected_range_high_pct: float
    prob_up_pct: float
    model_agreement_pct: float
    model_disagreement_std: float
    cost_hurdle_pct: float
    suggested_allocation_pct: float
    stop_loss_price: float
    take_profit_price: float
    garch_ann_vol_pct: float
    realized_vol21_pct: float
    vix_level: Optional[float]
    regime_label: str
    regime_probs: Dict[str, float]
    news_summary: Dict[str, Any]
    model_breakdown: List[Dict[str, Any]]
    top_drivers: List[Dict[str, Any]]
    cross_asset_snapshot: Dict[str, Optional[float]]
    backtest_track_record: Dict[str, Any] = field(default_factory=dict)
    in_training_universe: bool = True
    gate_passed: bool = True
    gate_reason: str = ""
    raw_verdict: str = ""
    raw_target_weight: float = 0.0


def _local_shap_drivers(lgb_model: Any, X_row: pd.DataFrame, top_n: int = 8) -> List[Dict[str, Any]]:
    """Compute top positive/negative feature contributions for a single live prediction row."""
    if lgb_model is None or X_row.empty:
        return []
    try:
        import shap

        explainer = shap.TreeExplainer(lgb_model)
        vals = explainer.shap_values(X_row)
        if isinstance(vals, list):
            vals = vals[0]
        arr = np.asarray(vals).ravel()
        row_vals = X_row.iloc[0].to_numpy(dtype=float)
        cols = list(X_row.columns)
        order = np.argsort(np.abs(arr))[::-1][:top_n]
        drivers = []
        for idx in order:
            v = float(row_vals[idx]) if np.isfinite(row_vals[idx]) else None
            drivers.append(
                {
                    "feature": cols[idx],
                    "value": round(v, 4) if v is not None else None,
                    "shap_impact": round(float(arr[idx]), 4),
                    "direction": "Bullish" if arr[idx] > 0 else "Bearish",
                }
            )
        return drivers
    except Exception:
        return []


def ensure_horizon_bundle(
    horizon_name: str,
    universe_name: str = "india_market_gold_silver",
    tune: bool = True,
    settings: Optional[EngineSettings] = None,
    force_retrain: bool = False,
) -> Dict[str, Any]:
    """Load saved models for a horizon, or train/backtest them if missing or requested."""
    settings = settings or EngineSettings()
    if not force_retrain:
        try:
            return load_ensemble_artifacts(horizon_name)
        except FileNotFoundError:
            pass

    horizon_cfg = HORIZON_CONFIGS[horizon_name]
    target_tickers = UNIVERSES.get(universe_name, UNIVERSES["india_market_gold_silver"])
    all_fetch = list(dict.fromkeys(target_tickers + GLOBAL_DRIVERS))

    ohlcv_map, _ = fetch_universe_ohlcv(
        all_fetch,
        period=horizon_cfg.period,
        interval=horizon_cfg.interval,
        settings=settings,
    )
    macro_df = load_all_macro_series(settings=settings)
    dataset = build_universe_dataset(target_tickers, ohlcv_map, macro_df, horizon_cfg)

    train_and_evaluate_walkforward(
        dataset=dataset,
        horizon_cfg=horizon_cfg,
        tune=tune,
        n_optuna_trials=settings.n_optuna_trials,
        random_state=settings.random_state,
        save_artifacts=True,
    )
    return load_ensemble_artifacts(horizon_name)


def _load_or_compute_track_record(
    horizon_name: str,
    bundle: Dict[str, Any],
    horizon_cfg: HorizonConfig,
    model_names: List[str],
    settings: EngineSettings,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], bool, str]:
    """Load Phase 0 benchmark-relative OOS leaderboard & per-asset summary, recomputing stale caches."""
    hdir = ARTIFACTS_DIR / horizon_name
    lb_cache = hdir / "leaderboard.parquet"
    as_cache = hdir / "per_asset_summary.parquet"

    lb_df: Optional[pd.DataFrame] = None
    asset_df: Optional[pd.DataFrame] = None
    if lb_cache.exists() and as_cache.exists():
        try:
            lb_cand = pd.read_parquet(lb_cache)
            as_cand = pd.read_parquet(as_cache)
            if (
                {"alpha_ann_pct", "hac_t_stat", "dsr_prob", "gate_passed"}.issubset(lb_cand.columns)
                and {"hit_edge_pct", "gate_passed"}.issubset(as_cand.columns)
            ):
                lb_df, asset_df = lb_cand, as_cand
        except Exception:
            lb_df, asset_df = None, None

    if lb_df is None or asset_df is None:
        oof_df = bundle.get("oof_predictions")
        if oof_df is not None and not oof_df.empty:
            bt_settings = EngineSettings(
                bootstrap_samples=min(150, settings.bootstrap_samples),
                bootstrap_block_len=settings.bootstrap_block_len,
                random_state=settings.random_state,
                max_leverage=settings.max_leverage,
                min_signal_threshold=settings.min_signal_threshold,
                portfolio=settings.portfolio,
                pass_gate=settings.pass_gate,
                verdicts=settings.verdicts,
            )
            bt = run_backtest_comparison(oof_df, model_names, horizon_cfg, bt_settings)
            lb_df = bt.leaderboard
            asset_df = bt.per_asset_summary
            try:
                hdir.mkdir(parents=True, exist_ok=True)
                lb_df.to_parquet(lb_cache, index=False)
                bt.equity_curves.to_parquet(hdir / "equity_curves.parquet")
                bt.drawdown_curves.to_parquet(hdir / "drawdown_curves.parquet")
                asset_df.to_parquet(as_cache, index=False)
            except Exception:
                pass

    track_record: Dict[str, Any] = {}
    per_asset_map: Dict[str, Dict[str, Any]] = {}
    if lb_df is not None and not lb_df.empty:
        ens_row = lb_df[lb_df["strategy"] == "Meta-Ensemble (Blended)"]
        if not ens_row.empty:
            track_record = ens_row.iloc[0].to_dict()
    if asset_df is not None and not asset_df.empty:
        per_asset_map = asset_df.set_index("ticker").to_dict(orient="index")

    horizon_passed, horizon_reason = evaluate_pass_gate_from_metrics(track_record, settings.pass_gate)
    return track_record, per_asset_map, horizon_passed, horizon_reason


def predict_asset(
    ticker: str,
    horizon_name: str = "swing",
    universe_name: str = "india_market_gold_silver",
    settings: Optional[EngineSettings] = None,
    ohlcv_map: Optional[Dict[str, pd.DataFrame]] = None,
    macro_df: Optional[pd.DataFrame] = None,
    scored_news: Optional[pd.DataFrame] = None,
    bundle: Optional[Dict[str, Any]] = None,
    use_cache: bool = True,
    closed_bars_only: bool = True,
    compute_shap: bool = True,
    _track_cache: Optional[Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], bool, str]] = None,
) -> LivePredictionReport:
    """Generate an objective multi-model, volatility-, regime-, gate-, and constraint-aware forecast."""
    settings = settings or EngineSettings()
    vcfg = settings.verdicts
    horizon_cfg: HorizonConfig = HORIZON_CONFIGS[horizon_name]
    bundle = bundle or ensure_horizon_bundle(horizon_name, universe_name=universe_name, settings=settings)

    if ohlcv_map is None:
        fetch_list = list(dict.fromkeys([ticker] + GLOBAL_DRIVERS))
        ohlcv_map, _ = fetch_universe_ohlcv(
            fetch_list,
            period=horizon_cfg.period,
            interval=horizon_cfg.interval,
            use_cache=use_cache,
            settings=settings,
        )
    elif ticker not in ohlcv_map:
        df_t = fetch_ohlcv(
            ticker,
            period=horizon_cfg.period,
            interval=horizon_cfg.interval,
            use_cache=use_cache,
            settings=settings,
        )
        if not df_t.empty:
            ohlcv_map[ticker] = df_t

    target_df = ohlcv_map.get(ticker)
    if closed_bars_only and target_df is not None and not target_df.empty:
        from quant_alpha.paper.ledger import filter_closed_bars

        target_df = filter_closed_bars(target_df, horizon_cfg.interval, ticker)

    if target_df is None or len(target_df) < 70:
        raise ValueError(f"Insufficient price history for '{ticker}' on {horizon_name} ({horizon_cfg.interval}).")

    if len(target_df) > 360:
        target_df = target_df.tail(360)
    inference_ohlcv_map = {
        k: (v.tail(450) if len(v) > 450 else v)
        for k, v in ohlcv_map.items()
    }
    inference_ohlcv_map[ticker] = target_df

    if macro_df is None:
        macro_df = load_all_macro_series(use_cache=use_cache, settings=settings)

    if scored_news is None:
        raw_news = fetch_all_news([ticker] + UNIVERSES.get(universe_name, [])[:30], use_cache=use_cache, settings=settings)
        scored_news = score_news_dataframe(raw_news)

    asset_frame = build_asset_feature_frame(
        ticker=ticker,
        ohlcv_df=target_df,
        ohlcv_map=inference_ohlcv_map,
        macro_df=macro_df,
        horizon_cfg=horizon_cfg,
    )
    ticker_to_id: Dict[str, Any] = bundle.get("ticker_to_id", {})
    in_universe = ticker in ticker_to_id or ticker in UNIVERSES.get(universe_name, [])

    regime_detector = bundle["final_regime_detector"]
    regime_df = regime_detector.predict_proba(asset_frame)
    full_frame = pd.concat([asset_frame, regime_df], axis=1)

    feature_cols: List[str] = bundle["feature_cols"]
    for col in feature_cols:
        if col not in full_frame.columns:
            full_frame[col] = np.nan

    latest_row = full_frame.dropna(subset=["close_price", "rv_21"]).tail(1)
    if latest_row.empty:
        raise ValueError(f"Unable to construct valid latest feature row for '{ticker}'.")

    X_live = latest_row[feature_cols].copy()
    last_close = float(latest_row["close_price"].iloc[0])
    est_ann_vol = float(latest_row["est_ann_vol"].iloc[0])
    rv21 = float(latest_row["rv_21"].iloc[0])
    horizon_vol = float(latest_row["horizon_vol"].iloc[0])
    atr_pct = float(latest_row["atr_14_pct"].iloc[0]) if pd.notna(latest_row["atr_14_pct"].iloc[0]) else 0.015

    model_names: List[str] = bundle["model_names"]
    weights: Dict[str, float] = bundle["ensemble_weights"]
    final_models: Dict[str, Any] = bundle["final_models"]

    breakdown: List[Dict[str, Any]] = []
    pred_vols = []
    prob_ups = []
    w_list = []

    for m in model_names:
        reg_m, clf_m = final_models[m]
        p_vol = float(np.clip(reg_m.predict(X_live)[0], -3.0, 3.0))
        p_up = float(clf_m.predict_proba(X_live)[0, 1])
        p_ret = p_vol * horizon_vol * 100.0
        w = float(weights.get(m, 1.0 / len(model_names)))
        pred_vols.append(p_vol)
        prob_ups.append(p_up)
        w_list.append(w)
        breakdown.append(
            {
                "model": MODEL_DISPLAY_NAMES.get(m, m),
                "model_key": m,
                "ensemble_weight": round(w * 100.0, 1),
                "pred_vol_norm": round(p_vol, 3),
                "expected_ret_pct": round(p_ret, 2),
                "prob_up_pct": round(p_up * 100.0, 1),
                "signal": (
                    "BULLISH"
                    if p_vol > settings.min_signal_threshold
                    else ("BEARISH" if p_vol < -settings.min_signal_threshold else "NEUTRAL")
                ),
            }
        )

    w_arr = np.array(w_list) / max(np.sum(w_list), 1e-8)
    pv_arr = np.array(pred_vols)
    pu_arr = np.array(prob_ups)

    quant_pred_vol = float(pv_arr @ w_arr)
    quant_prob_up = float(pu_arr @ w_arr)
    disagreement_std = float(np.std(pv_arr))
    ens_sign = np.sign(quant_pred_vol) if abs(quant_pred_vol) > 1e-4 else 1.0
    agreement_pct = float((np.sign(pv_arr) == ens_sign).mean() * 100.0)

    news_summary = compute_news_sentiment_summary(scored_news, ticker)
    total_articles = news_summary["asset_article_count"] + news_summary["macro_article_count"]
    sent_score = float(news_summary["composite_sentiment"])
    if total_articles >= vcfg.news_min_articles:
        news_vol_nudge = sent_score * vcfg.news_nudge_scale
        composite_pred_vol = (1.0 - vcfg.news_blend_weight) * quant_pred_vol + vcfg.news_blend_weight * news_vol_nudge
    else:
        composite_pred_vol = quant_pred_vol

    quant_ret_pct = quant_pred_vol * horizon_vol * 100.0
    comp_ret_pct = composite_pred_vol * horizon_vol * 100.0
    sigma_pct = horizon_vol * 100.0
    range_low = comp_ret_pct - sigma_pct
    range_high = comp_ret_pct + sigma_pct

    hurdle_pct = 2.0 * (horizon_cfg.tx_cost_bps + horizon_cfg.slippage_bps) / 100.0

    if abs(comp_ret_pct) < hurdle_pct or agreement_pct < vcfg.min_agreement_pct:
        raw_verdict = "NEUTRAL / HOLD"
    elif composite_pred_vol >= vcfg.strong_vol_norm_threshold and agreement_pct >= vcfg.strong_agreement_pct:
        raw_verdict = "STRONG BULLISH"
    elif composite_pred_vol > 0.0:
        raw_verdict = "LEAN BULLISH"
    elif composite_pred_vol <= -vcfg.strong_vol_norm_threshold and agreement_pct >= vcfg.strong_agreement_pct:
        raw_verdict = "STRONG BEARISH"
    else:
        raw_verdict = "LEAN BEARISH"

    reg_state = int(round(float(latest_row["regime_state"].iloc[0])))
    regime_probs = {
        "Low-Vol Bull": round(float(latest_row["prob_regime_bull"].iloc[0]) * 100.0, 1),
        "Normal / Chop": round(float(latest_row["prob_regime_chop"].iloc[0]) * 100.0, 1),
        "High-Vol Crisis": round(float(latest_row["prob_regime_crisis"].iloc[0]) * 100.0, 1),
    }

    if _track_cache is not None:
        track_record, per_asset_map, horizon_passed, horizon_reason = _track_cache
    else:
        track_record, per_asset_map, horizon_passed, horizon_reason = _load_or_compute_track_record(
            horizon_name, bundle, horizon_cfg, model_names, settings
        )

    asset_oos = per_asset_map.get(ticker)
    gate_passed, gate_reason = evaluate_asset_pass_gate(
        asset_oos,
        horizon_passed=horizon_passed,
        horizon_reason=horizon_reason,
        pass_gate=settings.pass_gate,
    )

    if not gate_passed:
        verdict = "NO EDGE / GATED"
        raw_weight = 0.0
        alloc_pct = 0.0
        stop_price = last_close * (1.0 - vcfg.atr_stop_mult * atr_pct)
        tp_price = last_close * (1.0 + vcfg.neutral_tp_mult * atr_pct)
    elif raw_verdict == "NEUTRAL / HOLD":
        verdict = raw_verdict
        raw_weight = 0.0
        alloc_pct = 0.0
        stop_price = last_close * (1.0 - vcfg.atr_stop_mult * atr_pct)
        tp_price = last_close * (1.0 + vcfg.neutral_tp_mult * atr_pct)
    else:
        verdict = raw_verdict
        raw_weight = float(
            compute_signal_raw_weights(
                pred_vol=np.array([composite_pred_vol]),
                prob_up=np.array([quant_prob_up]),
                est_ann_vol=np.array([est_ann_vol]),
                crisis_prob=np.array([float(latest_row["prob_regime_crisis"].iloc[0])]),
                horizon_cfg=horizon_cfg,
                settings=settings,
            )[0]
        )
        # Cash Indian equities/ETFs cannot be shorted; convert bearish cash signals to flat/avoid
        if is_cash_long_only_asset(ticker, settings.portfolio) and raw_weight < 0.0:
            raw_weight = 0.0
            alloc_pct = 0.0
        else:
            constrained = apply_portfolio_constraints(pd.Series({ticker: raw_weight}), settings.portfolio)
            alloc_pct = round(float(constrained.iloc[0]) * 100.0, 1)

        direction = 1.0 if composite_pred_vol > 0 else -1.0
        stop_price = last_close * (1.0 - direction * vcfg.atr_stop_mult * atr_pct)
        tp_price = last_close * (1.0 + direction * vcfg.atr_tp_mult * atr_pct)

    lgb_reg = final_models["lightgbm"][0] if "lightgbm" in final_models else None
    top_drivers = _local_shap_drivers(lgb_reg, X_live, top_n=8) if compute_shap else []

    cross_snap: Dict[str, Optional[float]] = {}
    for col in (
        "corr21_spy",
        "corr21_vix",
        "corr21_tlt",
        "corr21_gld",
        "corr21_btc",
        "beta_market_63",
        "vix_to_rv21",
        "yield_curve_10y_3m",
        "sig_macro_risk_on",
        "fred_t10y2y",
        "fred_bamlh0a0hym2",
    ):
        if col in latest_row.columns and pd.notna(latest_row[col].iloc[0]):
            cross_snap[col] = round(float(latest_row[col].iloc[0]), 4)

    vix_val = (
        round(float(latest_row["vix_level"].iloc[0]), 2)
        if "vix_level" in latest_row.columns and pd.notna(latest_row["vix_level"].iloc[0])
        else None
    )

    return LivePredictionReport(
        ticker=ticker,
        horizon_name=horizon_name,
        horizon_display=horizon_cfg.display_name,
        as_of_timestamp=str(latest_row.index[0])[:19],
        last_close=round(last_close, 4),
        verdict=verdict,
        quant_expected_ret_pct=round(quant_ret_pct, 2),
        composite_expected_ret_pct=round(comp_ret_pct, 2),
        expected_range_low_pct=round(range_low, 2),
        expected_range_high_pct=round(range_high, 2),
        prob_up_pct=round(quant_prob_up * 100.0, 1),
        model_agreement_pct=round(agreement_pct, 1),
        model_disagreement_std=round(disagreement_std, 3),
        cost_hurdle_pct=round(hurdle_pct, 3),
        suggested_allocation_pct=alloc_pct,
        stop_loss_price=round(stop_price, 4),
        take_profit_price=round(tp_price, 4),
        garch_ann_vol_pct=round(est_ann_vol * 100.0, 2),
        realized_vol21_pct=round(rv21 * 100.0, 2),
        vix_level=vix_val,
        regime_label=REGIME_LABELS.get(reg_state, "Unknown"),
        regime_probs=regime_probs,
        news_summary=news_summary,
        model_breakdown=breakdown,
        top_drivers=top_drivers,
        cross_asset_snapshot=cross_snap,
        backtest_track_record=track_record,
        in_training_universe=in_universe,
        gate_passed=gate_passed,
        gate_reason=gate_reason,
        raw_verdict=raw_verdict,
        raw_target_weight=raw_weight,
    )


def predict_multi_horizon_matrix(
    ticker: str,
    universe_name: str = "india_market_gold_silver",
    settings: Optional[EngineSettings] = None,
    closed_bars_only: bool = True,
) -> List[LivePredictionReport]:
    """Generate side-by-side Intraday, Swing, and Long-Term objective predictions."""
    settings = settings or EngineSettings()
    macro_df = load_all_macro_series(settings=settings)
    raw_news = fetch_all_news([ticker] + UNIVERSES.get(universe_name, [])[:25], settings=settings)
    scored_news = score_news_dataframe(raw_news)

    reports: List[LivePredictionReport] = []
    for h_name in ("intraday", "swing", "longterm"):
        reports.append(
            predict_asset(
                ticker=ticker,
                horizon_name=h_name,
                universe_name=universe_name,
                settings=settings,
                macro_df=macro_df,
                scored_news=scored_news,
                closed_bars_only=closed_bars_only,
            )
        )
    return reports


@dataclass
class MarketScanOutput:
    """Ranked multi-stock prediction scan combining core universe and live trending tickers."""

    scan_df: pd.DataFrame
    trending_df: pd.DataFrame
    reports: Dict[str, LivePredictionReport]
    skipped_tickers: Dict[str, str]


_METALS_TICKERS = {
    "GOLDBEES.NS", "HDFCGOLD.NS", "GOLDCASE.NS",
    "SILVERBEES.NS", "SILVER.NS", "SILVERIETF.NS", "HDFCSILVER.NS",
    "GLD", "SLV", "GC=F", "SI=F",
}
_INDIA_INDEX_ETFS = {
    "^NSEI", "^NSEBANK", "^CNXIT", "NIFTYBEES.NS", "BANKBEES.NS", "JUNIORBEES.NS", "INDA",
}
_MACRO_FX_TICKERS = {"USDINR=X", "BZ=F", "USO", "TLT", "UUP", "HYG", "IEF"}


def _classify_market_bucket(ticker: str) -> str:
    up = ticker.upper()
    if up in _METALS_TICKERS:
        return "Gold / Silver ETF"
    if up in _INDIA_INDEX_ETFS:
        return "India Index / ETF"
    if up.endswith((".NS", ".BO")):
        return "India Stock (.NS)"
    if is_crypto_ticker(up) or up in _MACRO_FX_TICKERS:
        return "Macro / FX / Crypto"
    return "US Stock / ETF"


def scan_market_predictions(
    horizon_name: str = "swing",
    universe_name: str = "india_market_gold_silver",
    include_trending: bool = True,
    max_trending_extras: int = 15,
    use_cache: bool = True,
    closed_bars_only: bool = True,
    settings: Optional[EngineSettings] = None,
    ohlcv_map: Optional[Dict[str, pd.DataFrame]] = None,
    macro_df: Optional[pd.DataFrame] = None,
    scored_news: Optional[pd.DataFrame] = None,
    bundle: Optional[Dict[str, Any]] = None,
) -> MarketScanOutput:
    """Actively scan and rank predictions using portfolio constraints, long-only cash rules, and PassGate."""
    settings = settings or EngineSettings()
    horizon_cfg = HORIZON_CONFIGS[horizon_name]
    bundle = bundle or ensure_horizon_bundle(horizon_name, universe_name=universe_name, settings=settings)

    base_tickers = list(UNIVERSES.get(universe_name, UNIVERSES["india_market_gold_silver"]))
    india_focus = universe_name.startswith("india_")
    if macro_df is None:
        macro_df = load_all_macro_series(use_cache=use_cache, settings=settings)

    if scored_news is None:
        news_tickers = list(dict.fromkeys(base_tickers[:35]))
        scored_news = score_news_dataframe(fetch_all_news(news_tickers, use_cache=use_cache, settings=settings))

    trending_df = discover_trending_stocks(
        news_df=scored_news,
        candidate_tickers=base_tickers,
        use_cache=use_cache,
        settings=settings,
        india_focus=india_focus,
    )

    scan_tickers = list(base_tickers)
    if include_trending and not trending_df.empty:
        cands = [s for s in trending_df["symbol"].tolist() if s not in scan_tickers]
        if india_focus:
            cands = [
                s for s in cands
                if s.endswith((".NS", ".BO")) or s in _METALS_TICKERS or s in _INDIA_INDEX_ETFS or s == "USDINR=X"
            ]
        scan_tickers.extend(cands[:max_trending_extras])

    if ohlcv_map is None:
        all_fetch = list(dict.fromkeys(scan_tickers + GLOBAL_DRIVERS))
        ohlcv_map, _ = fetch_universe_ohlcv(
            all_fetch,
            period=horizon_cfg.period,
            interval=horizon_cfg.interval,
            use_cache=use_cache,
            max_workers=10,
            settings=settings,
        )
    else:
        missing = [t for t in scan_tickers if t not in ohlcv_map]
        if missing:
            extra_map, _ = fetch_universe_ohlcv(
                missing,
                period=horizon_cfg.period,
                interval=horizon_cfg.interval,
                use_cache=use_cache,
                max_workers=8,
                settings=settings,
            )
            ohlcv_map.update(extra_map)

    buzz_lookup = (
        trending_df.set_index("symbol").to_dict(orient="index")
        if not trending_df.empty
        else {}
    )

    track_cache = _load_or_compute_track_record(
        horizon_name, bundle, horizon_cfg, bundle["model_names"], settings
    )

    rows: List[Dict[str, Any]] = []
    reports: Dict[str, LivePredictionReport] = {}
    skipped: Dict[str, str] = {}

    for t in scan_tickers:
        try:
            rep = predict_asset(
                ticker=t,
                horizon_name=horizon_name,
                universe_name=universe_name,
                settings=settings,
                ohlcv_map=ohlcv_map,
                macro_df=macro_df,
                scored_news=scored_news,
                bundle=bundle,
                use_cache=use_cache,
                closed_bars_only=closed_bars_only,
                compute_shap=False,
                _track_cache=track_cache,
            )
            reports[t] = rep
        except Exception as exc:
            skipped[t] = str(exc)

    if not reports:
        return MarketScanOutput(pd.DataFrame(), trending_df, reports, skipped)

    # Joint portfolio construction across all scanned assets (bounded by cluster, sector, gross/net & cash rules)
    raw_series = pd.Series(
        {
            t: (0.0 if (not rep.gate_passed or rep.verdict in {"NEUTRAL / HOLD", "NO EDGE / GATED"}) else rep.raw_target_weight)
            for t, rep in reports.items()
        },
        dtype=float,
    )
    portfolio_weights = apply_portfolio_constraints(raw_series, settings.portfolio)

    for t, rep in reports.items():
        alloc_pct = round(float(portfolio_weights.get(t, 0.0)) * 100.0, 1)
        rep.suggested_allocation_pct = alloc_pct
        binfo = buzz_lookup.get(t, {})
        t_df = ohlcv_map.get(t)
        chg_1_pct = (
            round(float(t_df["Close"].pct_change(1).iloc[-1] * 100.0), 2)
            if t_df is not None and len(t_df) >= 2 and pd.notna(t_df["Close"].pct_change(1).iloc[-1])
            else 0.0
        )
        is_actionable = 1 if rep.verdict not in {"NEUTRAL / HOLD", "NO EDGE / GATED"} else 0
        conviction_rank = abs(rep.composite_expected_ret_pct) * (rep.model_agreement_pct / 100.0)
        rows.append(
            {
                "ticker": t,
                "market": _classify_market_bucket(t),
                "buzz_score": float(binfo.get("buzz_score", 0.0)),
                "news_mentions": int(binfo.get("news_mentions", 0)),
                "trending_tags": str(binfo.get("sources", "")),
                "close": rep.last_close,
                "chg_1_pct": chg_1_pct,
                "verdict": rep.verdict,
                "raw_verdict": rep.raw_verdict,
                "gate_passed": rep.gate_passed,
                "composite_ret_pct": rep.composite_expected_ret_pct,
                "quant_ret_pct": rep.quant_expected_ret_pct,
                "prob_up_pct": rep.prob_up_pct,
                "consensus_pct": rep.model_agreement_pct,
                "garch_vol_pct": rep.garch_ann_vol_pct,
                "regime": rep.regime_label,
                "news_sent": float((rep.news_summary or {}).get("composite_sentiment", 0.0)),
                "alloc_pct": alloc_pct,
                "stop_loss": rep.stop_loss_price,
                "take_profit": rep.take_profit_price,
                "in_universe": rep.in_training_universe,
                "_sort_act": is_actionable,
                "_sort_conv": round(conviction_rank, 4),
            }
        )

    scan_df = (
        pd.DataFrame(rows)
        .sort_values(["_sort_act", "_sort_conv", "buzz_score"], ascending=False)
        .drop(columns=["_sort_act", "_sort_conv"])
        .reset_index(drop=True)
    )
    return MarketScanOutput(scan_df, trending_df, reports, skipped)
