"""Unified panel feature and label construction across diversified asset universes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from quant_alpha.config import HorizonConfig
from quant_alpha.features.cross_asset import build_cross_asset_features
from quant_alpha.features.cross_sectional import (
    add_cross_sectional_features,
    select_canonical_tickers,
)
from quant_alpha.features.technical import compute_technical_features
from quant_alpha.features.volatility import compute_volatility_features


@dataclass
class PreparedDataset:
    """Container holding strictly aligned historical labeled panel and latest inference rows."""

    panel_df: pd.DataFrame
    latest_df: pd.DataFrame
    feature_cols: List[str]
    ticker_to_id: Dict[str, float]
    canonical_tickers: List[str]
    target_ret_col: str = "target_fwd_ret"
    target_vol_ret_col: str = "target_vol_norm_ret"
    target_dir_col: str = "target_up"


def build_asset_feature_frame(
    ticker: str,
    ohlcv_df: pd.DataFrame,
    ohlcv_map: Dict[str, pd.DataFrame],
    macro_df: Optional[pd.DataFrame],
    horizon_cfg: HorizonConfig,
) -> pd.DataFrame:
    """Build complete feature and target DataFrame for a single asset."""
    is_intraday = horizon_cfg.interval not in {"1d", "1wk", "1mo"}
    tech = compute_technical_features(ohlcv_df, is_intraday=is_intraday)
    vol = compute_volatility_features(ohlcv_df, bars_per_year=horizon_cfg.bars_per_year)
    cross = build_cross_asset_features(
        target_ticker=ticker,
        target_df=ohlcv_df,
        ohlcv_map=ohlcv_map,
        macro_df=macro_df,
        is_intraday=is_intraday,
        bars_per_year=horizon_cfg.bars_per_year,
    )

    combined = pd.concat([tech, vol, cross], axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]

    close = ohlcv_df["Close"].astype(float)
    h = horizon_cfg.forecast_horizon
    fwd_ret = (close.shift(-h) / close) - 1.0

    # Volatility-normalized return target makes cross-asset pooling scale-consistent
    est_vol = vol["garch_vol_forecast"].fillna(vol["rv_21"]).fillna(0.18).clip(lower=0.04)
    horizon_vol = est_vol * np.sqrt(h / float(horizon_cfg.bars_per_year))
    vol_norm_ret = (fwd_ret / horizon_vol.replace(0.0, np.nan)).clip(-6.0, 6.0)

    combined["ticker"] = ticker
    combined["close_price"] = close
    combined["bar_ret_1"] = close.pct_change(1)
    combined["est_ann_vol"] = est_vol
    combined["horizon_vol"] = horizon_vol
    combined["target_fwd_ret"] = fwd_ret
    combined["target_vol_norm_ret"] = vol_norm_ret
    combined["target_up"] = np.where(fwd_ret.notna(), (fwd_ret > 0.0).astype(float), np.nan)
    return combined


def build_universe_dataset(
    target_tickers: List[str],
    ohlcv_map: Dict[str, pd.DataFrame],
    macro_df: Optional[pd.DataFrame],
    horizon_cfg: HorizonConfig,
) -> PreparedDataset:
    """Build unified cross-asset panel dataset sorted strictly by timestamp for purged CV."""
    frames: List[pd.DataFrame] = []
    latest_rows: List[pd.DataFrame] = []

    ticker_to_id = {t: idx for idx, t in enumerate(sorted(target_tickers))}

    for ticker in target_tickers:
        df = ohlcv_map.get(ticker)
        if df is None or len(df) < 120:
            continue
        asset_frame = build_asset_feature_frame(
            ticker=ticker,
            ohlcv_df=df,
            ohlcv_map=ohlcv_map,
            macro_df=macro_df,
            horizon_cfg=horizon_cfg,
        )
        asset_frame["asset_id"] = float(ticker_to_id[ticker])
        idx_name = asset_frame.index.name or "index"
        asset_frame = asset_frame.reset_index().rename(columns={idx_name: "Timestamp"})

        # Capture the most recent observable bar (even though forward target is NaN) for live prediction
        latest_valid = asset_frame.dropna(subset=["close_price", "rv_21"]).tail(1)
        if not latest_valid.empty:
            latest_rows.append(latest_valid)

        # Labeled historical training/backtesting rows require a real future target & warm-up history
        labeled = asset_frame.iloc[65:].dropna(subset=["target_fwd_ret", "target_vol_norm_ret", "rv_21"]).copy()
        if not labeled.empty:
            frames.append(labeled)

    if not frames:
        raise ValueError("No valid asset rows generated. Check market feeds and history length.")

    panel_df = pd.concat(frames, ignore_index=True)
    panel_df["Timestamp"] = pd.to_datetime(panel_df["Timestamp"])
    panel_df = panel_df.sort_values(["Timestamp", "ticker"]).reset_index(drop=True)
    panel_df = add_cross_sectional_features(panel_df)

    latest_df = pd.concat(latest_rows, ignore_index=True) if latest_rows else pd.DataFrame()
    if not latest_df.empty:
        latest_df["Timestamp"] = pd.to_datetime(latest_df["Timestamp"])
        latest_df = add_cross_sectional_features(latest_df)

    present_tickers = sorted(panel_df["ticker"].unique().tolist())
    canonical = select_canonical_tickers(present_tickers)
    if len(canonical) < len(present_tickers):
        panel_df = panel_df[panel_df["ticker"].isin(canonical)].sort_values(["Timestamp", "ticker"]).reset_index(drop=True)

    meta_cols = {
        "Timestamp",
        "ticker",
        "asset_id",
        "close_price",
        "bar_ret_1",
        "est_ann_vol",
        "horizon_vol",
        "target_fwd_ret",
        "target_vol_norm_ret",
        "target_up",
    }
    candidate_cols = [c for c in panel_df.columns if c not in meta_cols]
    # Keep features with reasonable non-missing coverage (<45% NaN across panel)
    valid_features = [
        c for c in candidate_cols
        if panel_df[c].isna().mean() < 0.45 and pd.api.types.is_numeric_dtype(panel_df[c])
    ]

    return PreparedDataset(
        panel_df=panel_df,
        latest_df=latest_df,
        feature_cols=valid_features,
        ticker_to_id={k: float(v) for k, v in ticker_to_id.items()},
        canonical_tickers=canonical,
    )


def assert_causal_features(
    feature_builder,
    ohlcv_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    cutoff_idx: int = -15,
    atol: float = 1e-10,
) -> None:
    """Assert that perturbing future bars after cutoff_idx causes zero change at or before cutoff_idx."""
    base_df = feature_builder(ohlcv_df)
    perturbed = ohlcv_df.copy()
    cut_pos = len(perturbed) + cutoff_idx if cutoff_idx < 0 else cutoff_idx
    if cut_pos <= 5 or cut_pos >= len(perturbed) - 1:
        raise ValueError("cutoff_idx must leave future rows after cut_pos to perturb.")

    for col in ("Open", "High", "Low", "Close"):
        if col in perturbed.columns:
            perturbed.iloc[cut_pos + 1 :, perturbed.columns.get_loc(col)] *= 1.35
    if "Volume" in perturbed.columns:
        perturbed.iloc[cut_pos + 1 :, perturbed.columns.get_loc("Volume")] *= 2.5

    mod_df = feature_builder(perturbed)
    cols = feature_cols or [c for c in base_df.columns if not str(c).startswith("target_") and pd.api.types.is_numeric_dtype(base_df[c])]
    leaks: List[str] = []
    for col in cols:
        a = base_df[col].iloc[: cut_pos + 1].to_numpy(dtype=float)
        b = mod_df[col].iloc[: cut_pos + 1].to_numpy(dtype=float)
        both_nan = np.isnan(a) & np.isnan(b)
        diff_mask = ~both_nan & ~np.isclose(a, b, atol=atol, rtol=1e-7, equal_nan=False)
        if np.any(diff_mask):
            leaks.append(str(col))

    if leaks:
        raise AssertionError(f"Causal look-ahead leak detected in features: {leaks}")

