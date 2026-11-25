"""Strictly causal technical, momentum, mean-reversion, microstructure, and classical signal features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _wilder_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def compute_technical_features(df: pd.DataFrame, is_intraday: bool = False) -> pd.DataFrame:
    """Compute strictly causal technical, microstructure, and classical quant signal features."""
    if df is None or df.empty:
        return pd.DataFrame()

    open_p = df["Open"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    volume = df["Volume"].astype(float).fillna(0.0)

    feat = pd.DataFrame(index=df.index)

    # Multi-horizon returns & momentum acceleration
    for w in (1, 2, 3, 5, 10, 21, 63):
        feat[f"ret_{w}"] = close.pct_change(w)
        feat[f"log_ret_{w}"] = np.log(close / close.shift(w).replace(0.0, np.nan))

    feat["mom_accel_5_21"] = feat["ret_5"] - (feat["ret_21"] / 4.2)
    feat["mom_accel_10_63"] = feat["ret_10"] - (feat["ret_63"] / 6.3)

    # Trend & moving average distance
    for w in (10, 20, 50, 200):
        sma = close.rolling(w, min_periods=max(5, w // 3)).mean()
        feat[f"dist_sma_{w}"] = (close / sma.replace(0.0, np.nan)) - 1.0

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    feat["macd_norm"] = macd / close.replace(0.0, np.nan)
    feat["macd_hist_norm"] = (macd - macd_signal) / close.replace(0.0, np.nan)
    feat["macd_hist_slope"] = feat["macd_hist_norm"].diff(3)

    # Oscillators & Bollinger squeeze
    feat["rsi_14"] = _wilder_rsi(close, 14)
    feat["rsi_5"] = _wilder_rsi(close, 5)

    bb_mid = close.rolling(20, min_periods=10).mean()
    bb_std = close.rolling(20, min_periods=10).std()
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    bb_width = (bb_upper - bb_lower).replace(0.0, np.nan)
    feat["bb_pct_b"] = (close - bb_lower) / bb_width
    feat["bb_bandwidth"] = bb_width / bb_mid.replace(0.0, np.nan)

    # Mean-reversion z-scores
    for w in (5, 20, 60):
        m = close.rolling(w, min_periods=max(3, w // 2)).mean()
        s = close.rolling(w, min_periods=max(3, w // 2)).std().replace(0.0, np.nan)
        feat[f"price_zscore_{w}"] = (close - m) / s

    # True Range / ATR / Trend Strength proxy
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr14 = tr.ewm(alpha=1.0 / 14.0, min_periods=7, adjust=False).mean()
    feat["atr_14_pct"] = atr14 / close.replace(0.0, np.nan)

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_di = 100.0 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / 14, adjust=False).mean() / atr14.replace(0.0, np.nan)
    minus_di = 100.0 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / 14, adjust=False).mean() / atr14.replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    feat["adx_14"] = dx.ewm(alpha=1 / 14, adjust=False).mean()
    feat["di_spread"] = plus_di - minus_di

    # Microstructure & candlestick range ratios
    feat["gap_ret"] = (open_p / prev_close.replace(0.0, np.nan)) - 1.0
    feat["intraday_ret"] = (close / open_p.replace(0.0, np.nan)) - 1.0
    feat["hl_range_pct"] = (high - low) / close.replace(0.0, np.nan)
    hl_span = (high - low).replace(0.0, np.nan)
    feat["close_location_val"] = ((close - low) / hl_span).fillna(0.5)

    # Volume & liquidity shocks (handle indices with 0 volume gracefully)
    has_volume = (volume > 0).mean() > 0.2
    if has_volume:
        vol_ma20 = volume.rolling(20, min_periods=5).mean().replace(0.0, np.nan)
        vol_std20 = volume.rolling(20, min_periods=5).std().replace(0.0, np.nan)
        feat["vol_ratio_20"] = volume / vol_ma20
        feat["vol_zscore_20"] = (volume - vol_ma20) / vol_std20
        dollar_vol = (close * volume).replace(0.0, np.nan)
        amihud = feat["ret_1"].abs() / (dollar_vol / 1e6)
        feat["amihud_illiq_20"] = np.log1p(amihud.rolling(20, min_periods=5).mean().clip(lower=0.0))
        obv_dir = np.sign(feat["ret_1"].fillna(0.0)) * volume
        obv = obv_dir.cumsum()
        feat["obv_zscore_20"] = (obv - obv.rolling(20).mean()) / obv.rolling(20).std().replace(0.0, np.nan)
    else:
        feat["vol_ratio_20"] = 1.0
        feat["vol_zscore_20"] = 0.0
        feat["amihud_illiq_20"] = 0.0
        feat["obv_zscore_20"] = 0.0

    # Statistical distribution & variance-ratio (trend vs mean reversion)
    r1 = feat["ret_1"]
    var1 = r1.rolling(21, min_periods=10).var()
    var5 = feat["ret_5"].rolling(21, min_periods=10).var()
    feat["variance_ratio_5"] = var5 / (5.0 * var1.replace(0.0, np.nan))
    feat["ret_autocorr_20"] = r1.rolling(20, min_periods=12).apply(
        lambda x: pd.Series(x).autocorr(lag=1), raw=False
    )
    feat["ret_skew_21"] = r1.rolling(21, min_periods=12).skew()
    feat["ret_kurt_21"] = r1.rolling(21, min_periods=12).kurt()

    # Distance from 52w / rolling 252-bar high & low
    roll_high = high.rolling(252, min_periods=30).max().replace(0.0, np.nan)
    roll_low = low.rolling(252, min_periods=30).min().replace(0.0, np.nan)
    feat["dist_252_high"] = (close / roll_high) - 1.0
    feat["dist_252_low"] = (close / roll_low) - 1.0

    # Cyclical calendar features
    idx = pd.to_datetime(df.index)
    feat["dow_sin"] = np.sin(2.0 * np.pi * idx.day_of_week / 7.0)
    feat["dow_cos"] = np.cos(2.0 * np.pi * idx.day_of_week / 7.0)
    feat["month_sin"] = np.sin(2.0 * np.pi * idx.month / 12.0)
    feat["month_cos"] = np.cos(2.0 * np.pi * idx.month / 12.0)
    if is_intraday:
        feat["hour_sin"] = np.sin(2.0 * np.pi * idx.hour / 24.0)
        feat["hour_cos"] = np.cos(2.0 * np.pi * idx.hour / 24.0)
    else:
        feat["hour_sin"] = 0.0
        feat["hour_cos"] = 1.0

    # Classical Quant Signal Scores (interpretable rule-based components in [-1, +1])
    trend_raw = np.tanh(feat["dist_sma_50"].fillna(0.0) * 12.0 + feat["dist_sma_200"].fillna(0.0) * 6.0)
    mr_raw = np.tanh(-feat["price_zscore_20"].fillna(0.0) * 0.6)
    mom_raw = np.tanh(feat["ret_21"].fillna(0.0) * 8.0 + feat["ret_63"].fillna(0.0) * 4.0)

    feat["sig_classical_trend"] = trend_raw
    feat["sig_classical_mean_rev"] = mr_raw
    feat["sig_classical_momentum"] = mom_raw

    return feat.replace([np.inf, -np.inf], np.nan)
