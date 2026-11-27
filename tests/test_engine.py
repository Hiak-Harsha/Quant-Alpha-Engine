"""Unit and anti-leakage verification tests for Quant Alpha Engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_alpha.backtest.engine import run_backtest_comparison
from quant_alpha.config import HORIZON_CONFIGS, EngineSettings
from quant_alpha.data.validator import validate_ohlcv_df
from quant_alpha.features.pipeline import build_universe_dataset
from quant_alpha.features.sentiment import score_headline
from quant_alpha.features.technical import compute_technical_features
from quant_alpha.features.volatility import compute_volatility_features
from quant_alpha.models.ensemble import train_and_evaluate_walkforward
from quant_alpha.models.validation import PurgedWalkForwardSplit
from quant_alpha.paper.ledger import PaperLedger, filter_closed_bars
from quant_alpha.predictor import LivePredictionReport


def _synthetic_ohlcv(n: int = 420, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-03", periods=n, freq="B")
    rets = rng.normal(0.0004, 0.012, size=n)
    close = 100.0 * np.exp(np.cumsum(rets))
    open_p = close * (1.0 + rng.normal(0.0, 0.003, size=n))
    high = np.maximum(open_p, close) * (1.0 + np.abs(rng.normal(0.004, 0.003, size=n)))
    low = np.minimum(open_p, close) * (1.0 - np.abs(rng.normal(0.004, 0.003, size=n)))
    vol = rng.integers(500_000, 3_000_000, size=n).astype(float)
    return pd.DataFrame(
        {"Open": open_p, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=dates,
    )


def test_technical_features_strictly_causal() -> None:
    df = _synthetic_ohlcv(250, seed=11)
    f1 = compute_technical_features(df)

    # Perturb the final 5 bars massively; earlier features up to index -6 must stay bit-for-bit identical
    df_perturbed = df.copy()
    df_perturbed.iloc[-5:, df_perturbed.columns.get_loc("Close")] *= 1.8
    df_perturbed.iloc[-5:, df_perturbed.columns.get_loc("High")] *= 1.9
    f2 = compute_technical_features(df_perturbed)

    cutoff = len(df) - 6
    pd.testing.assert_frame_equal(f1.iloc[:cutoff], f2.iloc[:cutoff])


def test_volatility_features_and_garch_causal() -> None:
    df = _synthetic_ohlcv(260, seed=21)
    v1 = compute_volatility_features(df, bars_per_year=252)
    assert "garch_vol_forecast" in v1.columns
    assert "parkinson_vol_21" in v1.columns
    assert "gk_vol_21" in v1.columns
    assert (v1["rv_21"].dropna() > 0).all()

    df_mod = df.copy()
    df_mod.iloc[-3:, df_mod.columns.get_loc("Close")] *= 0.5
    v2 = compute_volatility_features(df_mod, bars_per_year=252)
    # Earlier rows before perturbation must not change
    pd.testing.assert_frame_equal(v1.iloc[:-4], v2.iloc[:-4])


def test_purged_walkforward_has_zero_overlap_and_gap() -> None:
    dates = pd.Series(np.repeat(pd.date_range("2022-01-03", periods=150, freq="B"), 2))
    splitter = PurgedWalkForwardSplit(n_splits=4, purge_bars=10, embargo_bars=5, min_train_frac=0.4)
    folds = list(splitter.split(dates))
    assert len(folds) >= 3
    for tr_idx, te_idx in folds:
        max_tr = dates.iloc[tr_idx].max()
        min_te = dates.iloc[te_idx].min()
        assert max_tr < min_te
        unique_all = np.sort(dates.unique())
        pos_tr = int(np.searchsorted(unique_all, max_tr))
        pos_te = int(np.searchsorted(unique_all, min_te))
        assert (pos_te - pos_tr) >= 10


def test_financial_sentiment_scoring() -> None:
    bull = score_headline("Nvidia beats estimates and raises guidance on strong AI rally")
    bear = score_headline("Company slashes guidance amid recession warning and layoffs")
    assert bull > 0.25
    assert bear < -0.25


def test_validator_catches_healthy_and_empty() -> None:
    healthy = validate_ohlcv_df("SPY", _synthetic_ohlcv(200))
    empty = validate_ohlcv_df("BAD", pd.DataFrame())
    assert healthy.status == "OK"
    assert empty.status == "FAIL"


def test_end_to_end_pipeline_and_backtest_no_save() -> None:
    ohlcv_map = {
        "SPY": _synthetic_ohlcv(340, seed=1),
        "QQQ": _synthetic_ohlcv(340, seed=2),
        "GLD": _synthetic_ohlcv(340, seed=3),
        "^VIX": _synthetic_ohlcv(340, seed=4) * 0.2,
    }
    hcfg = HORIZON_CONFIGS["swing"]
    dataset = build_universe_dataset(["SPY", "QQQ", "GLD"], ohlcv_map, macro_df=None, horizon_cfg=hcfg)
    assert not dataset.panel_df.empty
    assert not dataset.latest_df.empty

    res = train_and_evaluate_walkforward(
        dataset=dataset,
        horizon_cfg=hcfg,
        model_names=["linear_ridge", "lightgbm", "classical_quant"],
        tune=False,
        save_artifacts=False,
    )
    assert "pred_vol_ensemble" in res.oof_predictions.columns
    assert np.isclose(sum(res.ensemble_weights.values()), 1.0, atol=0.02)

    bt = run_backtest_comparison(res.oof_predictions, res.model_names, hcfg, EngineSettings(bootstrap_samples=50))
    assert not bt.leaderboard.empty
    assert "Meta-Ensemble (Blended)" in bt.leaderboard["strategy"].values
    assert "Buy & Hold (Equal Weight)" in bt.leaderboard["strategy"].values


def _make_dummy_report(
    ticker: str,
    horizon_name: str,
    as_of: str,
    close: float,
    verdict: str,
    alloc_pct: float,
    sl: float,
    tp: float,
    pred_dir_sign: float,
) -> LivePredictionReport:
    return LivePredictionReport(
        ticker=ticker,
        horizon_name=horizon_name,
        horizon_display=HORIZON_CONFIGS[horizon_name].display_name,
        as_of_timestamp=as_of,
        last_close=close,
        verdict=verdict,
        quant_expected_ret_pct=1.2 * pred_dir_sign,
        composite_expected_ret_pct=1.2 * pred_dir_sign,
        expected_range_low_pct=-2.0,
        expected_range_high_pct=3.0,
        prob_up_pct=62.0 if pred_dir_sign > 0 else 38.0,
        model_agreement_pct=83.3,
        model_disagreement_std=0.15,
        cost_hurdle_pct=0.10,
        suggested_allocation_pct=alloc_pct,
        stop_loss_price=sl,
        take_profit_price=tp,
        garch_ann_vol_pct=18.0,
        realized_vol21_pct=17.5,
        vix_level=20.0,
        regime_label="Normal / Transition",
        regime_probs={"Low-Vol Bull": 20.0, "Normal / Chop": 70.0, "High-Vol Crisis": 10.0},
        news_summary={"composite_sentiment": 0.1},
        model_breakdown=[
            {
                "model": "LightGBM (Leaf-wise GBDT)",
                "model_key": "lightgbm",
                "ensemble_weight": 50.0,
                "pred_vol_norm": 0.35 * pred_dir_sign,
                "expected_ret_pct": 1.2 * pred_dir_sign,
                "prob_up_pct": 62.0 if pred_dir_sign > 0 else 38.0,
                "signal": "BULLISH" if pred_dir_sign > 0 else "BEARISH",
            }
        ],
        top_drivers=[],
        cross_asset_snapshot={},
    )


def test_filter_closed_bars_prevents_mid_candle_leakage() -> None:
    # Intraday 1H bars: at 18:35 UTC, 17:30 bar is closed (17:30+1h=18:30 <= 18:35), 18:30 bar is NOT closed
    now_utc = pd.Timestamp("2026-11-25 18:35:00")
    hr_idx = pd.to_datetime(["2026-11-25 16:30:00", "2026-11-25 17:30:00", "2026-11-25 18:30:00"])
    hr_df = pd.DataFrame({"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.5, "Volume": 1000.0}, index=hr_idx)
    closed_hr = filter_closed_bars(hr_df, "1h", "SPY", now_utc=now_utc)
    assert list(closed_hr.index) == list(hr_idx[:2])

    # Daily bars at 18:35 UTC on 2026-11-25: India (^NSEI) today is closed, US (SPY) and Crypto (BTC-USD) today are NOT closed
    d_idx = pd.to_datetime(["2026-11-24 00:00:00", "2026-11-25 00:00:00"])
    d_df = pd.DataFrame({"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.5, "Volume": 1000.0}, index=d_idx)
    assert len(filter_closed_bars(d_df, "1d", "^NSEI", now_utc=now_utc)) == 2
    assert len(filter_closed_bars(d_df, "1d", "SPY", now_utc=now_utc)) == 1
    assert len(filter_closed_bars(d_df, "1d", "BTC-USD", now_utc=now_utc)) == 1


def test_paper_ledger_idempotency_bar_maturity_and_conservative_barriers(tmp_path) -> None:
    ledger = PaperLedger(db_path=tmp_path / "paper_test.db")
    rep_long = _make_dummy_report("SPY", "swing", "2026-11-25 00:00:00", 100.0, "LEAN BULLISH", 20.0, 97.0, 105.0, 1.0)
    logged1, _ = ledger.log_prediction(rep_long)
    logged2, _ = ledger.log_prediction(rep_long)
    assert logged1 is True
    assert logged2 is False

    # Only 4 future closed bars -> must remain OPEN (no premature settlement)
    bars_4 = pd.DataFrame(
        {
            "Open": [100, 101, 102, 103],
            "High": [102, 103, 104, 104],
            "Low": [99, 100, 101, 102],
            "Close": [101, 102, 103, 103.5],
            "Volume": [1e6] * 4,
        },
        index=pd.to_datetime(["2026-11-27", "2026-11-30", "2026-12-01", "2026-12-02"]),
    )
    res4 = ledger.evaluate_open_predictions(
        ohlcv_fetcher=lambda t, p, i: bars_4,
        now_utc=pd.Timestamp("2026-12-03 22:00:00"),
    )
    assert res4.newly_matured == 0
    assert res4.still_open == 1

    # 5th future closed bar completes horizon; bar 2 touches BOTH SL (<=97) and TP (>=105) -> conservative STOP_LOSS priority!
    bars_5 = pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0, 103.0, 103.5],
            "High": [102.0, 106.0, 104.0, 104.5, 104.5],
            "Low": [99.0, 96.5, 101.0, 102.0, 103.0],
            "Close": [101.0, 102.0, 103.0, 103.5, 104.0],
            "Volume": [1e6] * 5,
        },
        index=pd.to_datetime(["2026-11-27", "2026-11-30", "2026-12-01", "2026-12-02", "2026-12-03"]),
    )
    res5 = ledger.evaluate_open_predictions(
        ohlcv_fetcher=lambda t, p, i: bars_5,
        now_utc=pd.Timestamp("2026-12-04 22:00:00"),
    )
    assert res5.newly_matured == 1
    assert res5.still_open == 0

    df_all = ledger.load_all_predictions()
    row = df_all.iloc[0]
    assert row["status"] == "MATURED"
    assert row["barrier_outcome"] == "STOP_LOSS"
    assert row["direction_correct"] == 1
    assert np.isclose(row["actual_asset_ret_pct"], 4.0)
    assert np.isclose(row["net_strategy_ret_pct"], 3.90)
    assert np.isclose(row["risk_managed_ret_pct"], -3.10)
    assert np.isclose(row["allocated_hold_pnl_pct"], 0.78)
    assert np.isclose(row["allocated_risk_pnl_pct"], -0.62)


def test_paper_ledger_short_take_profit_and_flat(tmp_path) -> None:
    ledger = PaperLedger(db_path=tmp_path / "paper_short.db")
    rep_short = _make_dummy_report("QQQ", "swing", "2026-11-25 00:00:00", 100.0, "STRONG BEARISH", -25.0, 103.0, 94.0, -1.0)
    rep_flat = _make_dummy_report("GLD", "swing", "2026-11-25 00:00:00", 200.0, "NEUTRAL / HOLD", 0.0, 195.0, 208.0, 0.0)
    ledger.log_prediction(rep_short)
    ledger.log_prediction(rep_flat)

    bars_short = pd.DataFrame(
        {
            "Open": [99.0, 97.0, 95.0, 94.0, 96.0],
            "High": [100.0, 98.0, 96.0, 96.0, 97.0],
            "Low": [97.5, 93.5, 94.0, 93.0, 95.0],
            "Close": [98.0, 95.0, 95.0, 95.5, 96.0],
            "Volume": [1e6] * 5,
        },
        index=pd.to_datetime(["2026-11-27", "2026-11-30", "2026-12-01", "2026-12-02", "2026-12-03"]),
    )
    res = ledger.evaluate_open_predictions(
        ohlcv_fetcher=lambda t, p, i: bars_short,
        now_utc=pd.Timestamp("2026-12-04 22:00:00"),
    )
    assert res.newly_matured == 2
    df = ledger.load_all_predictions().set_index("ticker")

    # Short QQQ: dropped 4% at maturity, hit TP 94.0 on bar 2 (+6.0% gross - 0.10% cost = +5.90%)
    q = df.loc["QQQ"]
    assert q["barrier_outcome"] == "TAKE_PROFIT"
    assert q["direction_correct"] == 1
    assert np.isclose(q["risk_managed_ret_pct"], 5.90)
    assert np.isclose(q["allocated_risk_pnl_pct"], 0.25 * 5.90)

    # Flat GLD: exact zero strategy P&L
    g = df.loc["GLD"]
    assert g["barrier_outcome"] == "FLAT"
    assert np.isclose(g["allocated_hold_pnl_pct"], 0.0)
    assert np.isclose(g["allocated_risk_pnl_pct"], 0.0)

    summary = ledger.get_summary()
    assert summary["matured_count"] == 2
    assert summary["active_matured_count"] == 1
    assert summary["hit_rate_pct"] == 100.0
    assert summary["tp_count"] == 1
    assert not ledger.get_model_accuracy_leaderboard().empty


def test_trending_symbol_filter_and_universes() -> None:
    from quant_alpha.config import UNIVERSES
    from quant_alpha.data.trending_feed import _is_tradable_symbol
    from quant_alpha.features.cross_asset import build_cross_asset_features

    assert len(UNIVERSES["india_market_gold_silver"]) >= 45
    assert "GOLDBEES.NS" in UNIVERSES["india_gold_silver"]
    assert "SILVERBEES.NS" in UNIVERSES["india_gold_silver"]
    assert len(UNIVERSES["top_talked_leaders"]) >= 40
    assert _is_tradable_symbol("NVDA") is True
    assert _is_tradable_symbol("RELIANCE.NS") is True
    assert _is_tradable_symbol("^NSEI") is True
    assert _is_tradable_symbol("USDINR=X") is True
    assert _is_tradable_symbol("GC=F") is True
    assert _is_tradable_symbol("BTC-USD") is True
    assert _is_tradable_symbol("ES=F") is False
    assert _is_tradable_symbol("^GSPC") is False

    # Verify Indian Gold ETF receives lagged global Gold/Silver & USDINR drivers with zero look-ahead
    gold_df = _synthetic_ohlcv(160, seed=31)
    ohlcv_map = {
        "GOLDBEES.NS": gold_df,
        "GC=F": _synthetic_ohlcv(160, seed=32),
        "SI=F": _synthetic_ohlcv(160, seed=33),
        "USDINR=X": _synthetic_ohlcv(160, seed=34),
        "^NSEI": _synthetic_ohlcv(160, seed=35),
    }
    cf1 = build_cross_asset_features("GOLDBEES.NS", gold_df, ohlcv_map)
    assert "drv_usdinr_ret5" in cf1.columns
    assert "corr21_gold_fut" in cf1.columns

    # Perturb final bar of USDINR=X; today's GOLDBEES.NS features must remain identical because USDINR=X is lagged by 1 day
    ohlcv_mod = dict(ohlcv_map)
    u_mod = ohlcv_map["USDINR=X"].copy()
    u_mod.iloc[-1, u_mod.columns.get_loc("Close")] *= 1.5
    ohlcv_mod["USDINR=X"] = u_mod
    cf2 = build_cross_asset_features("GOLDBEES.NS", gold_df, ohlcv_mod)
    pd.testing.assert_frame_equal(cf1, cf2)


def test_planted_feature_leak_and_daily_feature_hygiene() -> None:
    from quant_alpha.config import HORIZON_CONFIGS
    from quant_alpha.features.pipeline import (
        assert_causal_features,
        build_asset_feature_frame,
        build_universe_dataset,
    )
    from quant_alpha.features.technical import compute_technical_features

    df = _synthetic_ohlcv(180, seed=55)
    daily_tech = compute_technical_features(df, is_intraday=False)
    assert "hour_sin" not in daily_tech.columns
    assert "hour_cos" not in daily_tech.columns

    ohlcv_map = {"RELIANCE.NS": df, "TCS.NS": _synthetic_ohlcv(180, seed=56), "SPY": _synthetic_ohlcv(180, seed=57)}
    hcfg = HORIZON_CONFIGS["swing"]
    ds = build_universe_dataset(["RELIANCE.NS", "TCS.NS"], ohlcv_map, macro_df=None, horizon_cfg=hcfg)
    assert "asset_id" not in ds.feature_cols

    # Causal pipeline must pass assert_causal_features cleanly
    assert_causal_features(
        lambda d: build_asset_feature_frame("RELIANCE.NS", d, {"RELIANCE.NS": d}, None, hcfg),
        df,
        cutoff_idx=-10,
    )

    # Planted look-ahead leak (peeking at t+1 close) must be caught and raise AssertionError
    def leaky_builder(d: pd.DataFrame) -> pd.DataFrame:
        out = compute_technical_features(d, is_intraday=False)
        out["planted_future_ret"] = d["Close"].shift(-1) / d["Close"] - 1.0
        return out

    with pytest.raises(AssertionError, match="planted_future_ret"):
        assert_causal_features(leaky_builder, df, cutoff_idx=-10)


def test_portfolio_construction_constraints_and_cash_long_only() -> None:
    from quant_alpha.config import PortfolioConfig
    from quant_alpha.portfolio.construct import apply_portfolio_constraints, construct_portfolio_weights_df

    cfg = PortfolioConfig(
        max_gross_exposure=1.0,
        max_net_exposure=1.0,
        max_single_asset_weight=0.15,
        max_cluster_weight=0.25,
        max_sector_weight=0.35,
        long_only_cash_india=True,
    )
    raw = pd.Series(
        {
            "RELIANCE.NS": -0.60,     # Indian cash short attempt -> must clamp to 0.0
            "GOLDBEES.NS": 0.40,
            "HDFCGOLD.NS": 0.40,
            "GOLDCASE.NS": 0.40,
            "GLD": -0.30,             # Non-Indian ETF may short
            "HDFCBANK.NS": 0.50,
            "ICICIBANK.NS": 0.50,
            "SBIN.NS": 0.50,
            "TCS.NS": 0.50,
            "INFY.NS": 0.50,
            "SPY": 0.40,
        }
    )
    w = apply_portfolio_constraints(raw, cfg)
    assert w["RELIANCE.NS"] == 0.0
    assert w["GLD"] <= 0.0
    assert float(w.abs().max()) <= cfg.max_single_asset_weight + 1e-9
    gold_cluster_gross = float(w[["GOLDBEES.NS", "HDFCGOLD.NS", "GOLDCASE.NS", "GLD"]].abs().sum())
    assert gold_cluster_gross <= cfg.max_cluster_weight + 1e-9
    banks_sector_gross = float(w[["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS"]].abs().sum())
    assert banks_sector_gross <= cfg.max_sector_weight + 1e-9
    assert float(w.abs().sum()) <= cfg.max_gross_exposure + 1e-9
    assert float(abs(w.sum())) <= cfg.max_net_exposure + 1e-9

    wide_df = pd.DataFrame([raw, raw * 0.5], index=pd.date_range("2026-01-01", periods=2, freq="B"))
    w_wide = construct_portfolio_weights_df(wide_df, cfg)
    assert (w_wide["RELIANCE.NS"] >= 0.0).all()
    assert (w_wide.abs().sum(axis=1) <= cfg.max_gross_exposure + 1e-9).all()


def test_backtester_accounting_benchmark_metrics_and_gating() -> None:
    from quant_alpha.backtest.engine import _simulate_model_portfolio, run_backtest_comparison
    from quant_alpha.backtest.metrics import evaluate_asset_pass_gate, evaluate_pass_gate_from_metrics
    from quant_alpha.config import HORIZON_CONFIGS, EngineSettings, PassGateConfig, PortfolioConfig

    dates = pd.bdate_range("2026-01-01", periods=4)
    # Deterministic 1-asset test where weight at t=0..2 earns exact known returns at t=1..3
    toy_oof = pd.DataFrame(
        {
            "Timestamp": dates,
            "ticker": ["SPY"] * 4,
            "bar_ret_1": [0.00, 0.02, -0.01, 0.03],
            "est_ann_vol": [0.15] * 4,
            "horizon_vol": [0.02] * 4,
            "prob_regime_crisis": [0.0] * 4,
            "target_fwd_ret": [0.02, -0.01, 0.03, 0.01],
            "target_up": [1.0, 0.0, 1.0, 1.0],
            "pred_vol_ridge": [0.5, 0.5, 0.5, 0.0],
            "prob_up_ridge": [0.8, 0.8, 0.8, 0.5],
            "pred_vol_ensemble": [0.5, 0.5, 0.5, 0.0],
            "prob_up_ensemble": [0.8, 0.8, 0.8, 0.5],
        }
    )
    hcfg = HORIZON_CONFIGS["swing"]
    settings = EngineSettings(
        bootstrap_samples=20,
        portfolio=PortfolioConfig(max_single_asset_weight=1.0, max_cluster_weight=1.0, max_sector_weight=1.0, max_gross_exposure=1.0),
    )
    net_ret, turnover = _simulate_model_portfolio(toy_oof, "pred_vol_ensemble", "prob_up_ensemble", hcfg, settings, settings.portfolio)
    # At t=0: weight > 0 decided; at t=1 earns +0.02 * w0 - cost(dw)
    assert turnover.iloc[0] > 0.0
    assert np.isclose(net_ret.iloc[0], -turnover.iloc[0] * ((hcfg.tx_cost_bps + hcfg.slippage_bps) / 10000.0))

    # Full run_backtest_comparison check on synthetic 30-bar OOF data
    rng = np.random.default_rng(7)
    n_bars = 35
    d_idx = pd.bdate_range("2025-06-01", periods=n_bars)
    rows = []
    for t in ("SPY", "RELIANCE.NS"):
        rets = rng.normal(0.001, 0.01, size=n_bars)
        for i in range(n_bars):
            rows.append(
                {
                    "Timestamp": d_idx[i],
                    "ticker": t,
                    "bar_ret_1": rets[i],
                    "est_ann_vol": 0.15,
                    "horizon_vol": 0.02,
                    "prob_regime_crisis": 0.1,
                    "target_fwd_ret": rets[i],
                    "target_up": float(rets[i] > 0),
                    "pred_vol_ridge": 0.25 if i % 2 == 0 else -0.25,
                    "prob_up_ridge": 0.6 if i % 2 == 0 else 0.4,
                    "pred_vol_ensemble": 0.25 if i % 2 == 0 else -0.25,
                    "prob_up_ensemble": 0.6 if i % 2 == 0 else 0.4,
                }
            )
    bt = run_backtest_comparison(pd.DataFrame(rows), ["ridge"], hcfg, settings)
    lb = bt.leaderboard.set_index("strategy")
    assert lb.loc["Buy & Hold (Equal Weight)", "edge_verdict"] == "Benchmark (Not Rated)"
    assert "Buy & Hold (Vol-Matched)" in lb.index
    assert {"alpha_ann_pct", "beta_to_bh", "hac_t_stat", "dsr_prob", "always_long_hit_pct", "hit_edge_pct", "gate_passed"}.issubset(lb.columns)

    # PassGate tests
    gate_cfg = PassGateConfig(enable_pass_gate=True, min_alpha_ann_pct=0.0, min_hac_tstat=2.0, min_dsr_prob=0.95)
    ok_fail, _ = evaluate_pass_gate_from_metrics({"alpha_ann_pct": -1.5, "hac_t_stat": 0.5, "dsr_prob": 0.40}, gate_cfg)
    assert ok_fail is False
    ok_pass, _ = evaluate_pass_gate_from_metrics({"alpha_ann_pct": 4.5, "hac_t_stat": 2.4, "dsr_prob": 0.97}, gate_cfg)
    assert ok_pass is True
    asset_ok, _ = evaluate_asset_pass_gate({"sharpe": 1.2, "hit_edge_pct": 2.0}, horizon_passed=False, horizon_reason="failed", pass_gate=gate_cfg)
    assert asset_ok is False


def test_live_functions_default_closed_bars_only() -> None:
    import inspect

    from quant_alpha.predictor import predict_asset, scan_market_predictions

    assert inspect.signature(predict_asset).parameters["closed_bars_only"].default is True
    assert inspect.signature(scan_market_predictions).parameters["closed_bars_only"].default is True


def test_next_day_horizon_and_modular_ui_watchlists() -> None:
    from quant_alpha.config import HORIZON_CONFIGS
    from quant_alpha.ui.components import (
        build_metals_spotlight_df,
        build_upside_downside_watchlists,
        compute_capital_scenario,
        format_pulse_value,
        summarize_gate_status,
    )

    assert "next_day" in HORIZON_CONFIGS
    assert HORIZON_CONFIGS["next_day"].forecast_horizon == 1

    assert format_pulse_value("^NSEI", 25884.9) == "25,885"
    assert format_pulse_value("GOLDBEES.NS", 100.96) == "Rs 100.96"

    gate_fail = summarize_gate_status({"gate_passed": False, "alpha_ann_pct": 1.2, "hac_t_stat": 0.9, "dsr_prob": 0.5, "sharpe": 1.1})
    assert gate_fail["passed"] is False
    assert "0.0%" in gate_fail["headline"]

    sample_scan = pd.DataFrame(
        [
            {
                "ticker": "RELIANCE.NS",
                "market": "India Stock (.NS)",
                "as_of": "2026-11-26",
                "close": 1380.0,
                "chg_1_pct": 0.5,
                "verdict": "NO EDGE / GATED",
                "raw_verdict": "STRONG BULLISH",
                "cash_long_only": True,
                "composite_ret_pct": 1.4,
                "range_low_pct": -0.5,
                "range_high_pct": 3.3,
                "prob_up_pct": 66.0,
                "consensus_pct": 83.3,
                "garch_vol_pct": 22.0,
                "stop_loss": 1340.0,
                "take_profit": 1440.0,
                "alloc_pct": 0.0,
            },
            {
                "ticker": "TCS.NS",
                "market": "India Stock (.NS)",
                "as_of": "2026-11-26",
                "close": 2500.0,
                "chg_1_pct": -0.8,
                "verdict": "NO EDGE / GATED",
                "raw_verdict": "LEAN BEARISH",
                "cash_long_only": True,
                "composite_ret_pct": -1.1,
                "range_low_pct": -2.8,
                "range_high_pct": 0.6,
                "prob_up_pct": 38.0,
                "consensus_pct": 66.7,
                "garch_vol_pct": 24.0,
                "stop_loss": 2550.0,
                "take_profit": 2420.0,
                "alloc_pct": 0.0,
            },
            {
                "ticker": "GOLDBEES.NS",
                "market": "Gold / Silver ETF",
                "as_of": "2026-11-26",
                "close": 100.96,
                "chg_1_pct": 0.85,
                "verdict": "NO EDGE / GATED",
                "raw_verdict": "NEUTRAL / HOLD",
                "cash_long_only": True,
                "composite_ret_pct": -0.4,
                "range_low_pct": -2.1,
                "range_high_pct": 1.3,
                "prob_up_pct": 49.0,
                "consensus_pct": 33.3,
                "garch_vol_pct": 31.0,
                "stop_loss": 97.5,
                "take_profit": 105.0,
                "alloc_pct": 0.0,
            },
        ]
    )
    gainers, decliners = build_upside_downside_watchlists(sample_scan, top_n=5)
    assert gainers["Ticker"].tolist() == ["RELIANCE.NS"]
    assert decliners["Ticker"].tolist() == ["TCS.NS", "GOLDBEES.NS"]
    assert "Cash Long-Only" in decliners.loc[0, "Execution Note"]
    metals = build_metals_spotlight_df(sample_scan)
    assert metals["Ticker"].tolist() == ["GOLDBEES.NS"]

    rep = LivePredictionReport(
        ticker="RELIANCE.NS",
        horizon_name="next_day",
        horizon_display="Next Day",
        as_of_timestamp="2026-11-26",
        last_close=1000.0,
        verdict="NO EDGE / GATED",
        quant_expected_ret_pct=1.5,
        composite_expected_ret_pct=1.5,
        expected_range_low_pct=-1.0,
        expected_range_high_pct=4.0,
        prob_up_pct=65.0,
        model_agreement_pct=83.3,
        model_disagreement_std=0.1,
        cost_hurdle_pct=0.1,
        suggested_allocation_pct=0.0,
        stop_loss_price=980.0,
        take_profit_price=1040.0,
        garch_ann_vol_pct=20.0,
        realized_vol21_pct=19.0,
        vix_level=16.8,
        regime_label="Low-Vol Bull",
        regime_probs={},
        news_summary={},
        model_breakdown=[],
        top_drivers=[],
        cross_asset_snapshot={},
    )
    scen = compute_capital_scenario(rep, 100000.0)
    assert scen["expected_pnl"] == 1500.0
    assert scen["stop_loss_risk"] == -2000.0
    assert scen["take_profit_gain"] == 4000.0
    assert scen["gated_capital"] == 0.0


def test_ist_date_normalization_indian_costs_and_paper_simulator(tmp_path) -> None:
    from quant_alpha.data.market_feed import _normalize_ohlcv_df
    from quant_alpha.paper.costs import compute_order_fees, estimate_round_trip_breakeven_pct
    from quant_alpha.paper.simulator import PaperBrokerSimulator, compute_prediction_market_correlation

    # Verify IST 2026-11-26 midnight bar preserves 2026-11-26 (not UTC-shifted to 2026-11-25)
    ist_idx = pd.DatetimeIndex(["2026-11-25 00:00:00", "2026-11-26 00:00:00"]).tz_localize("Asia/Kolkata")
    raw = pd.DataFrame(
        {"Open": [100.0, 101.0], "High": [102.0, 103.0], "Low": [99.0, 100.0], "Close": [101.0, 102.0], "Volume": [1000, 1200]},
        index=ist_idx,
    )
    norm = _normalize_ohlcv_df(raw, interval="1d")
    assert list(norm.index.strftime("%Y-%m-%d")) == ["2026-11-25", "2026-11-26"]

    # Verify Indian equity & ETF delivery costs
    buy_fees = compute_order_fees("RELIANCE.NS", "BUY", 10000.0)
    sell_fees = compute_order_fees("RELIANCE.NS", "SELL", 10000.0)
    assert buy_fees["stt"] == 10.0
    assert buy_fees["stamp_duty"] == 1.5
    assert buy_fees["dp_charge"] == 0.0
    assert sell_fees["dp_charge"] == 15.34
    assert estimate_round_trip_breakeven_pct("GOLDBEES.NS", 10000.0) > 0.15

    # Verify PaperBrokerSimulator cash delivery enforcement & realized P&L
    db_file = tmp_path / "sim_test.db"
    sim = PaperBrokerSimulator(db_path=db_file)
    bad_frac = sim.preview_order("default_student", "RELIANCE.NS", "BUY", 1.5, 1000.0)
    assert bad_frac.valid is False
    ok_buy, _ = sim.execute_order("default_student", "RELIANCE.NS", "BUY", 10, 1000.0, signal_verdict="STRONG BULLISH")
    assert ok_buy is True
    bad_short = sim.preview_order("default_student", "RELIANCE.NS", "SELL", 15, 1050.0)
    assert bad_short.valid is False

    ok_sell, _ = sim.execute_order("default_student", "RELIANCE.NS", "SELL", 10, 1100.0)
    assert ok_sell is True
    snap = sim.get_portfolio_snapshot("default_student", {"RELIANCE.NS": 1100.0})
    assert snap["realized_pnl"] > 900.0  # +1000 gross minus real Indian buy+sell fees
    assert snap["total_fees_paid"] > 30.0
    assert abs(snap["total_equity"] - (100000.0 + snap["realized_pnl"])) < 0.5

    # Fractional crypto order allowed
    ok_btc, _ = sim.execute_order("default_student", "BTC-USD", "BUY", 0.05, 90000.0)
    assert ok_btc is True

    # Prediction vs Market Correlation helper runs cleanly
    ledger = PaperLedger(db_path=db_file)
    corr = compute_prediction_market_correlation(ledger)
    assert corr["n_matured"] == 0


def test_causal_market_hmm_cross_sectional_dedup_and_justification() -> None:
    from quant_alpha.config import HORIZON_CONFIGS
    from quant_alpha.features.cross_sectional import add_cross_sectional_features, select_canonical_tickers
    from quant_alpha.features.pipeline import build_universe_dataset
    from quant_alpha.features.regime import MarketRegimeDetector
    from quant_alpha.ui.components import build_signal_justification

    # Deduplication keeps 1 Gold, 1 Silver, 1 Nifty, plus individual equities
    canonical = select_canonical_tickers(
        ["GOLDBEES.NS", "HDFCGOLD.NS", "GC=F", "GLD", "SILVERBEES.NS", "SLV", "^NSEI", "NIFTYBEES.NS", "RELIANCE.NS"]
    )
    assert set(canonical) == {"GOLDBEES.NS", "SILVERBEES.NS", "^NSEI", "RELIANCE.NS"}

    # Chronological HMM forward filter has zero look-ahead when future bars change
    df1 = _synthetic_ohlcv(220, seed=71)
    feat1 = compute_volatility_features(df1)
    feat1["ret_5"] = df1["Close"].pct_change(5).fillna(0.0)
    feat2 = feat1.copy()
    feat2.iloc[-5:] *= 4.0

    det = MarketRegimeDetector(random_state=42).fit(feat1.iloc[:150], timestamps=pd.Series(df1.index[:150]))
    p1 = det.predict_proba(feat1, timestamps=pd.Series(df1.index))
    p2 = det.predict_proba(feat2, timestamps=pd.Series(df1.index))
    pd.testing.assert_frame_equal(p1.iloc[:-5], p2.iloc[:-5], atol=1e-10)

    # Cross-sectional features populate cleanly
    d_a = _synthetic_ohlcv(160, seed=81)
    d_b = _synthetic_ohlcv(160, seed=82)
    ds = build_universe_dataset(
        ["RELIANCE.NS", "TCS.NS", "GOLDBEES.NS", "HDFCGOLD.NS"],
        {"RELIANCE.NS": d_a, "TCS.NS": d_b, "GOLDBEES.NS": d_a, "HDFCGOLD.NS": d_a, "SPY": d_b},
        macro_df=None,
        horizon_cfg=HORIZON_CONFIGS["next_day"],
    )
    assert "cs_rank_ret21" in ds.feature_cols
    assert "HDFCGOLD.NS" not in set(ds.panel_df["ticker"].unique())

    cs_only = add_cross_sectional_features(
        pd.DataFrame({"Timestamp": [pd.Timestamp("2026-11-26")] * 2, "ticker": ["A", "B"], "ret_1": [0.02, -0.01], "ret_5": [0.05, 0.01], "ret_21": [0.1, -0.02], "rv_21": [0.2, 0.3], "rsi_14": [65.0, 40.0], "amihud_21": [1e-6, 2e-6], "idio_ret_21": [0.04, -0.02]})
    )
    assert float(cs_only.loc[0, "cs_rank_ret1"]) > float(cs_only.loc[1, "cs_rank_ret1"])

    rep = LivePredictionReport(
        ticker="RELIANCE.NS",
        horizon_name="next_day",
        horizon_display="Next Day",
        as_of_timestamp="2026-11-26",
        last_close=1396.2,
        verdict="NO EDGE / GATED",
        quant_expected_ret_pct=0.25,
        composite_expected_ret_pct=0.22,
        expected_range_low_pct=-1.1,
        expected_range_high_pct=1.54,
        prob_up_pct=56.0,
        model_agreement_pct=83.3,
        model_disagreement_std=0.1,
        cost_hurdle_pct=0.1,
        suggested_allocation_pct=0.0,
        stop_loss_price=1350.0,
        take_profit_price=1450.0,
        garch_ann_vol_pct=24.0,
        realized_vol21_pct=22.0,
        vix_level=16.8,
        regime_label="Low-Vol Bull / Calm",
        regime_probs={"Low-Vol Bull": 72.0},
        news_summary={"asset_article_count": 3, "macro_article_count": 2, "composite_sentiment": 0.2},
        model_breakdown=[],
        top_drivers=[{"feature": "cs_rank_ret21", "value": 0.85, "shap_impact": 0.12, "direction": "Bullish"}],
        cross_asset_snapshot={"beta_market_63": 1.05},
        gate_passed=False,
        gate_reason="Horizon gated",
        raw_verdict="LEAN BULLISH",
    )
    cards = build_signal_justification(rep, position_capital=100000.0)
    assert len(cards) == 5
    assert "21D Peer Momentum Rank" in cards[0]["body"]


def test_fastapi_endpoints_and_paper_flow() -> None:
    from fastapi.testclient import TestClient

    from quant_alpha.api.app import create_app

    client = TestClient(create_app())
    r_health = client.get("/api/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "ok"

    r_cfg = client.get("/api/config")
    assert r_cfg.status_code == 200
    cfg = r_cfg.json()
    assert "next_day" in cfg["horizons"]
    assert "india_market_gold_silver" in cfg["universes"]

    r_prev = client.post(
        "/api/paper/preview",
        json={"account_id": "default_student", "ticker": "GOLDBEES.NS", "side": "BUY", "qty": 10, "price": 100.0},
    )
    assert r_prev.status_code == 200
    prev = r_prev.json()
    assert prev["valid"] is True
    assert prev["breakeven_move_pct"] > 0.0

    r_corr = client.get("/api/paper/correlation")
    assert r_corr.status_code == 200
    assert "n_matured" in r_corr.json()


def test_daily_nan_backfill_from_intraday_and_fast_info() -> None:
    from quant_alpha.data.market_feed import _normalize_ohlcv_df, _repair_trailing_daily_nan

    ist_daily = pd.DatetimeIndex(["2026-11-25 00:00:00", "2026-11-26 00:00:00"]).tz_localize("Asia/Kolkata")
    raw_daily = pd.DataFrame(
        {
            "Open": [140.0, float("nan")],
            "High": [142.0, float("nan")],
            "Low": [138.0, float("nan")],
            "Close": [138.40, float("nan")],
            "Volume": [100000.0, 0.0],
        },
        index=ist_daily,
    )
    norm = _normalize_ohlcv_df(raw_daily, interval="1d")
    assert len(norm) == 1

    ist_h1 = pd.DatetimeIndex(["2026-11-26 09:15:00", "2026-11-26 15:15:00"]).tz_localize("Asia/Kolkata")
    h1_df = pd.DataFrame(
        {
            "Open": [139.5, 141.6],
            "High": [141.0, 142.0],
            "Low": [139.0, 141.0],
            "Close": [140.8, 141.27],
            "Volume": [50000.0, 60000.0],
        },
        index=ist_h1,
    )

    class _FakeTicker:
        class _FastInfo:
            last_price = 141.48

        fast_info = _FastInfo()

        def history(self, **_kwargs: object) -> pd.DataFrame:
            return h1_df

    repaired = _repair_trailing_daily_nan(_FakeTicker(), raw_daily, norm)
    assert len(repaired) == 2
    assert repaired.index[-1].strftime("%Y-%m-%d") == "2026-11-26"
    assert np.isclose(float(repaired["Close"].iloc[-1]), 141.48)
    assert np.isclose(float(repaired["Open"].iloc[-1]), 139.5)
    assert np.isclose(float(repaired["Volume"].iloc[-1]), 110000.0)

    # Also handle internal skipped date once next day's live bar (2026-11-27) opens
    ist_daily_gap = pd.DatetimeIndex(["2026-11-25 00:00:00", "2026-11-27 00:00:00"]).tz_localize("Asia/Kolkata")
    raw_gap = pd.DataFrame(
        {
            "Open": [140.0, 145.95],
            "High": [142.0, 146.04],
            "Low": [138.0, 144.61],
            "Close": [138.40, 144.68],
            "Volume": [100000.0, 3500000.0],
        },
        index=ist_daily_gap,
    )
    norm_gap = _normalize_ohlcv_df(raw_gap, interval="1d")

    class _FakeOpenTicker(_FakeTicker):
        class _FastInfo:
            last_price = 144.68

        fast_info = _FastInfo()

    repaired_gap = _repair_trailing_daily_nan(_FakeOpenTicker(), raw_gap, norm_gap)
    assert [d.strftime("%Y-%m-%d") for d in repaired_gap.index] == ["2026-11-25", "2026-11-26", "2026-11-27"]
    # Internal missing bar must use settled 1h close (141.27), NOT today's live last_price (144.68)
    assert np.isclose(float(repaired_gap.loc[pd.Timestamp("2026-11-26"), "Close"]), 141.27)
    assert np.isclose(float(repaired_gap.loc[pd.Timestamp("2026-11-27"), "Close"]), 144.68)






