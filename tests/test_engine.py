"""Unit and anti-leakage verification tests for Quant Alpha Engine."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_alpha.backtest.engine import run_backtest_comparison
from quant_alpha.config import HORIZON_CONFIGS, EngineSettings
from quant_alpha.data.validator import validate_ohlcv_df
from quant_alpha.features.pipeline import build_universe_dataset
from quant_alpha.features.regime import MarketRegimeDetector
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
