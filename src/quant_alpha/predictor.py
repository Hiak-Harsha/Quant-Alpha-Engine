"""Objective multi-model, volatility-aware, regime- and news-adjusted live prediction engine."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from quant_alpha.backtest.engine import run_backtest_comparison
from quant_alpha.config import (
    ARTIFACTS_DIR,
    GLOBAL_DRIVERS,
    HORIZON_CONFIGS,
    UNIVERSES,
    EngineSettings,
    HorizonConfig,
)
from quant_alpha.data.macro_feed import load_all_macro_series
from quant_alpha.data.market_feed import fetch_ohlcv, fetch_universe_ohlcv
from quant_alpha.data.news_feed import fetch_all_news
from quant_alpha.features.pipeline import build_asset_feature_frame, build_universe_dataset
from quant_alpha.features.regime import REGIME_LABELS
from quant_alpha.features.sentiment import compute_news_sentiment_summary, score_news_dataframe
from quant_alpha.models.ensemble import load_ensemble_artifacts, train_and_evaluate_walkforward
from quant_alpha.models.zoo import MODEL_DISPLAY_NAMES

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
    universe_name: str = "stepping_stone",
    tune: bool = True,
    settings: Optional[EngineSettings] = None,
    force_retrain: bool = False,
) -> Dict[str, Any]:
    """Load a trained horizon bundle or train it cleanly if absent."""
    settings = settings or EngineSettings()
    if not force_retrain:
        try:
            return load_ensemble_artifacts(horizon_name)
        except FileNotFoundError:
            pass

    horizon_cfg = HORIZON_CONFIGS[horizon_name]
    tickers = UNIVERSES.get(universe_name, UNIVERSES["stepping_stone"])
    all_tickers = list(dict.fromkeys(tickers + GLOBAL_DRIVERS))
    ohlcv_map, _ = fetch_universe_ohlcv(
        all_tickers,
        period=horizon_cfg.period,
        interval=horizon_cfg.interval,
        settings=settings,
    )
    macro_df = load_all_macro_series(settings=settings)
    dataset = build_universe_dataset(tickers, ohlcv_map, macro_df, horizon_cfg)
    train_and_evaluate_walkforward(
        dataset=dataset,
        horizon_cfg=horizon_cfg,
        tune=tune,
        n_optuna_trials=settings.n_optuna_trials,
        random_state=settings.random_state,
        save_artifacts=True,
    )
    return load_ensemble_artifacts(horizon_name)


def predict_asset(
    ticker: str,
    horizon_name: str = "swing",
    universe_name: str = "stepping_stone",
    settings: Optional[EngineSettings] = None,
    ohlcv_map: Optional[Dict[str, pd.DataFrame]] = None,
    macro_df: Optional[pd.DataFrame] = None,
    scored_news: Optional[pd.DataFrame] = None,
    bundle: Optional[Dict[str, Any]] = None,
    use_cache: bool = True,
    closed_bars_only: bool = False,
) -> LivePredictionReport:
    """Generate an objective multi-model, volatility-, regime-, and news-aware forecast."""
    settings = settings or EngineSettings()
    horizon_cfg: HorizonConfig = HORIZON_CONFIGS[horizon_name]
    bundle = bundle or ensure_horizon_bundle(horizon_name, universe_name=universe_name, settings=settings)

    # Fetch live/cached data if not passed in
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

    if macro_df is None:
        macro_df = load_all_macro_series(use_cache=use_cache, settings=settings)

    if scored_news is None:
        raw_news = fetch_all_news([ticker] + UNIVERSES.get(universe_name, []), use_cache=use_cache, settings=settings)
        scored_news = score_news_dataframe(raw_news)

    # Build full causal feature frame for target ticker
    asset_frame = build_asset_feature_frame(
        ticker=ticker,
        ohlcv_df=target_df,
        ohlcv_map=ohlcv_map,
        macro_df=macro_df,
        horizon_cfg=horizon_cfg,
    )
    ticker_to_id: Dict[str, float] = bundle.get("ticker_to_id", {})
    in_universe = ticker in ticker_to_id
    asset_frame["asset_id"] = float(ticker_to_id.get(ticker, np.median(list(ticker_to_id.values())) if ticker_to_id else 0.0))

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

    # Run every model in the zoo
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
                "signal": "BULLISH" if p_vol > 0.05 else ("BEARISH" if p_vol < -0.05 else "NEUTRAL"),
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

    # News & Macro Sentiment Overlay (transparent separation from pure historical quant score)
    news_summary = compute_news_sentiment_summary(scored_news, ticker)
    total_articles = news_summary["asset_article_count"] + news_summary["macro_article_count"]
    sent_score = float(news_summary["composite_sentiment"])
    if total_articles >= 2:
        news_vol_nudge = sent_score * 0.30
        composite_pred_vol = 0.85 * quant_pred_vol + 0.15 * news_vol_nudge
    else:
        composite_pred_vol = quant_pred_vol

    quant_ret_pct = quant_pred_vol * horizon_vol * 100.0
    comp_ret_pct = composite_pred_vol * horizon_vol * 100.0
    sigma_pct = horizon_vol * 100.0
    range_low = comp_ret_pct - sigma_pct
    range_high = comp_ret_pct + sigma_pct

    # Cost hurdle = round-trip transaction cost + slippage
    hurdle_pct = 2.0 * (horizon_cfg.tx_cost_bps + horizon_cfg.slippage_bps) / 100.0

    if abs(comp_ret_pct) < hurdle_pct or agreement_pct < 55.0:
        verdict = "NEUTRAL / HOLD"
    elif composite_pred_vol >= 0.30 and agreement_pct >= 66.0:
        verdict = "STRONG BULLISH"
    elif composite_pred_vol > 0.0:
        verdict = "LEAN BULLISH"
    elif composite_pred_vol <= -0.30 and agreement_pct >= 66.0:
        verdict = "STRONG BEARISH"
    else:
        verdict = "LEAN BEARISH"

    # Regime & Volatility-Targeted Position Sizing
    reg_state = int(round(float(latest_row["regime_state"].iloc[0])))
    regime_probs = {
        "Low-Vol Bull": round(float(latest_row["prob_regime_bull"].iloc[0]) * 100.0, 1),
        "Normal / Chop": round(float(latest_row["prob_regime_chop"].iloc[0]) * 100.0, 1),
        "High-Vol Crisis": round(float(latest_row["prob_regime_crisis"].iloc[0]) * 100.0, 1),
    }

    if verdict == "NEUTRAL / HOLD":
        alloc_pct = 0.0
        stop_price = last_close * (1.0 - 1.5 * atr_pct)
        tp_price = last_close * (1.0 + 2.0 * atr_pct)
    else:
        conf_mult = 0.65 + 0.70 * abs(quant_prob_up - 0.5)
        vol_scale = float(np.clip(horizon_cfg.target_vol_annual / max(est_ann_vol, 0.05), 0.35, settings.max_leverage))
        crisis_damp = float(np.clip(1.0 - 0.50 * float(latest_row["prob_regime_crisis"].iloc[0]), 0.40, 1.0))
        raw_alloc = float(np.tanh(composite_pred_vol * 3.0) * conf_mult * vol_scale * crisis_damp)
        alloc_pct = round(float(np.clip(raw_alloc, -settings.max_leverage, settings.max_leverage)) * 100.0, 1)
        direction = 1.0 if composite_pred_vol > 0 else -1.0
        stop_price = last_close * (1.0 - direction * 1.5 * atr_pct)
        tp_price = last_close * (1.0 + direction * 2.5 * atr_pct)

    # Local SHAP drivers for transparency
    lgb_reg = final_models["lightgbm"][0] if "lightgbm" in final_models else None
    top_drivers = _local_shap_drivers(lgb_reg, X_live, top_n=8)

    # Cross-asset correlation & macro snapshot
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

    # Historical walk-forward track record for this horizon (use cached summary if available for speed)
    track_record: Dict[str, Any] = {}
    lb_cache = ARTIFACTS_DIR / horizon_name / "leaderboard.parquet"
    if lb_cache.exists():
        try:
            lb_df = pd.read_parquet(lb_cache)
            ens_row = lb_df[lb_df["strategy"] == "Meta-Ensemble (Blended)"]
            if not ens_row.empty:
                track_record = ens_row.iloc[0].to_dict()
        except Exception:
            pass
    if not track_record:
        oof_df = bundle.get("oof_predictions")
        if oof_df is not None and not oof_df.empty:
            bt = run_backtest_comparison(oof_df, model_names, horizon_cfg, EngineSettings(bootstrap_samples=100))
            try:
                lb_cache.parent.mkdir(parents=True, exist_ok=True)
                bt.leaderboard.to_parquet(lb_cache, index=False)
            except Exception:
                pass
            ens_row = bt.leaderboard[bt.leaderboard["strategy"] == "Meta-Ensemble (Blended)"]
            if not ens_row.empty:
                track_record = ens_row.iloc[0].to_dict()

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
    )


def predict_multi_horizon_matrix(
    ticker: str,
    universe_name: str = "stepping_stone",
    settings: Optional[EngineSettings] = None,
) -> List[LivePredictionReport]:
    """Generate side-by-side Intraday, Swing, and Long-Term objective predictions."""
    settings = settings or EngineSettings()
    macro_df = load_all_macro_series(settings=settings)
    raw_news = fetch_all_news([ticker] + UNIVERSES.get(universe_name, []), settings=settings)
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
            )
        )
    return reports
