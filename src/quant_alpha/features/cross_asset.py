"""Causal cross-asset correlation, rolling beta, macro spread, and global driver features."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from quant_alpha.config import is_asia_india_ticker, is_crypto_ticker


def _align_driver_series(
    driver_df: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    lag_bars: int = 0,
    is_intraday: bool = False,
) -> pd.DataFrame:
    """Align a driver OHLCV DataFrame strictly causally onto target_index."""
    if driver_df is None or driver_df.empty:
        return pd.DataFrame(index=target_index)

    src = driver_df.copy().sort_index()
    if lag_bars > 0:
        src = src.shift(lag_bars)

    if is_intraday:
        left = pd.DataFrame({"Timestamp": pd.to_datetime(target_index)}).sort_values("Timestamp")
        right = src.reset_index().rename(columns={src.index.name or "index": "Timestamp"})
        right["Timestamp"] = pd.to_datetime(right["Timestamp"])
        right = right.sort_values("Timestamp")
        merged = pd.merge_asof(
            left,
            right,
            on="Timestamp",
            direction="backward",
            tolerance=pd.Timedelta("7D"),
        )
        merged = merged.set_index("Timestamp").reindex(target_index)
        return merged

    combined_idx = target_index.union(src.index).sort_values()
    aligned = src.reindex(combined_idx).ffill(limit=5).reindex(target_index)
    return aligned


def build_cross_asset_features(
    target_ticker: str,
    target_df: pd.DataFrame,
    ohlcv_map: Dict[str, pd.DataFrame],
    macro_df: Optional[pd.DataFrame] = None,
    is_intraday: bool = False,
    bars_per_year: int = 252,
) -> pd.DataFrame:
    """Build strictly causal cross-asset, correlation, credit, yield-curve, and macro features."""
    if target_df is None or target_df.empty:
        return pd.DataFrame()

    idx = target_df.index
    target_close = target_df["Close"].astype(float)
    target_ret1 = target_close.pct_change(1)
    target_rv21 = np.log(target_close / target_close.shift(1).replace(0.0, np.nan)).rolling(21, min_periods=8).std() * np.sqrt(bars_per_year)

    feat = pd.DataFrame(index=idx)

    # Determine causal lag rules for US vs Asia/India vs 24/7 Crypto
    target_is_asia = is_asia_india_ticker(target_ticker)
    target_is_crypto = is_crypto_ticker(target_ticker)

    def _lag_for(driver_ticker: str) -> int:
        if is_intraday:
            # Always lag intraday cross-asset bars by 1 bar because yfinance timestamps represent bar open.
            return 1
        driver_is_asia = is_asia_india_ticker(driver_ticker)
        driver_is_crypto = is_crypto_ticker(driver_ticker)
        if target_is_crypto and not driver_is_crypto:
            return 1
        if target_is_asia and not driver_is_asia:
            return 1
        return 0

    key_drivers = {
        "spy": "SPY",
        "qqq": "QQQ",
        "iwm": "IWM",
        "vix": "^VIX",
        "vxn": "^VXN",
        "tnx": "^TNX",
        "irx": "^IRX",
        "uup": "UUP",
        "usdinr": "USDINR=X",
        "hyg": "HYG",
        "ief": "IEF",
        "tlt": "TLT",
        "gld": "GLD",
        "slv": "SLV",
        "gold_fut": "GC=F",
        "silver_fut": "SI=F",
        "brent": "BZ=F",
        "uso": "USO",
        "xlk": "XLK",
        "xlp": "XLP",
        "nsei": "^NSEI",
        "nsebank": "^NSEBANK",
        "indiavix": "^INDIAVIX",
        "btc": "BTC-USD",
    }

    aligned_closes: Dict[str, pd.Series] = {}
    for alias, dticker in key_drivers.items():
        ddf = ohlcv_map.get(dticker)
        if ddf is None or ddf.empty:
            continue
        lag = _lag_for(dticker)
        aligned = _align_driver_series(ddf, idx, lag_bars=lag, is_intraday=is_intraday)
        if "Close" in aligned.columns and aligned["Close"].notna().sum() > 20:
            aligned_closes[alias] = aligned["Close"].astype(float)

    # Driver momentum & rolling correlation / beta to target
    corr_drivers = ["spy", "qqq", "vix", "tlt", "gld", "slv", "uup", "usdinr", "gold_fut", "silver_fut", "brent", "btc", "nsei"]
    for alias in corr_drivers:
        if alias not in aligned_closes:
            continue
        dc = aligned_closes[alias]
        d_ret1 = dc.pct_change(1)
        d_ret5 = dc.pct_change(5)
        d_ret21 = dc.pct_change(21)

        if alias in {"spy", "qqq", "tlt", "gld", "slv", "uup", "usdinr", "gold_fut", "silver_fut", "brent", "btc", "nsei"}:
            feat[f"drv_{alias}_ret1"] = d_ret1
            feat[f"drv_{alias}_ret5"] = d_ret5
            feat[f"drv_{alias}_ret21"] = d_ret21

        # Avoid self-correlation = 1 constant
        if key_drivers[alias].upper() != target_ticker.upper():
            feat[f"corr21_{alias}"] = target_ret1.rolling(21, min_periods=10).corr(d_ret1)
            feat[f"corr63_{alias}"] = target_ret1.rolling(63, min_periods=20).corr(d_ret1)
        else:
            feat[f"corr21_{alias}"] = 1.0
            feat[f"corr63_{alias}"] = 1.0

    # Market Beta & Idiosyncratic Alpha (use NIFTY 50 for Indian assets when present, SPY otherwise)
    bench_alias: Optional[str]
    if target_is_asia and "nsei" in aligned_closes:
        bench_alias = "nsei"
    else:
        bench_alias = "spy" if "spy" in aligned_closes else ("nsei" if "nsei" in aligned_closes else None)
    if bench_alias is not None:
        b_ret1 = aligned_closes[bench_alias].pct_change(1)
        cov63 = target_ret1.rolling(63, min_periods=20).cov(b_ret1)
        var63 = b_ret1.rolling(63, min_periods=20).var().replace(0.0, np.nan)
        beta63 = (cov63 / var63).clip(-3.0, 4.0)
        feat["beta_market_63"] = beta63
        feat["idio_ret_1"] = target_ret1 - beta63 * b_ret1
        feat["idio_ret_5"] = feat["idio_ret_1"].rolling(5, min_periods=3).sum()
        feat["idio_ret_21"] = feat["idio_ret_1"].rolling(21, min_periods=10).sum()

    # Cross-asset breadth & risk-appetite ratios
    ratio_pairs = [
        ("qqq_spy", "qqq", "spy"),
        ("iwm_spy", "iwm", "spy"),
        ("hyg_ief", "hyg", "ief"),
        ("xlk_xlp", "xlk", "xlp"),
        ("gld_spy", "gld", "spy"),
        ("gold_silver", "gld", "slv"),
        ("nsebank_nsei", "nsebank", "nsei"),
    ]
    for rname, num, den in ratio_pairs:
        if num in aligned_closes and den in aligned_closes:
            ratio = aligned_closes[num] / aligned_closes[den].replace(0.0, np.nan)
            feat[f"ratio_{rname}_ret5"] = ratio.pct_change(5)
            feat[f"ratio_{rname}_ret21"] = ratio.pct_change(21)

    # Volatility & Variance Risk Premium features
    if "vix" in aligned_closes:
        vix = aligned_closes["vix"]
        feat["vix_level"] = vix
        feat["vix_chg_1"] = vix.pct_change(1)
        feat["vix_chg_5"] = vix.pct_change(5)
        feat["vix_pct_rank_252"] = vix.rolling(252, min_periods=30).rank(pct=True)
        feat["vix_to_rv21"] = (vix / 100.0) / target_rv21.replace(0.0, np.nan)
        if "vxn" in aligned_closes:
            feat["vxn_vix_spread"] = aligned_closes["vxn"] - vix

    # Yield curve slope proxy (^TNX - ^IRX)
    if "tnx" in aligned_closes:
        feat["tnx_level"] = aligned_closes["tnx"]
        feat["tnx_chg_5"] = aligned_closes["tnx"].diff(5)
        if "irx" in aligned_closes:
            feat["yield_curve_10y_3m"] = aligned_closes["tnx"] - aligned_closes["irx"]
            feat["yield_curve_chg_21"] = feat["yield_curve_10y_3m"].diff(21)

    # FRED public macro series (ffill sparse weekly/holiday gaps on native calendar, then shift(1) for publication-lag safety)
    if macro_df is not None and not macro_df.empty:
        macro_native = macro_df.sort_index().ffill(limit=10)
        macro_chg = macro_native.diff(21)
        macro_chg.columns = [f"{c}_chg21" for c in macro_chg.columns]
        macro_combined = pd.concat([macro_native, macro_chg], axis=1).shift(1)
        macro_aligned = _align_driver_series(macro_combined, idx, lag_bars=0, is_intraday=is_intraday)
        for col in macro_df.columns:
            s = pd.to_numeric(macro_aligned[col], errors="coerce")
            s_chg = pd.to_numeric(macro_aligned[f"{col}_chg21"], errors="coerce")
            if s.notna().sum() > 20:
                feat[f"fred_{col.lower()}"] = s
                feat[f"fred_{col.lower()}_chg21"] = s_chg

    # Composite Macro Risk-On Score in [-1, +1]
    feat = feat.copy()
    risk_components = []
    if "ratio_hyg_ief_ret21" in feat.columns:
        risk_components.append(np.tanh(feat["ratio_hyg_ief_ret21"].fillna(0.0) * 35.0))
    if "ratio_xlk_xlp_ret21" in feat.columns:
        risk_components.append(np.tanh(feat["ratio_xlk_xlp_ret21"].fillna(0.0) * 15.0))
    if "vix_chg_5" in feat.columns:
        risk_components.append(np.tanh(-feat["vix_chg_5"].fillna(0.0) * 4.0))
    if "ratio_gld_spy_ret21" in feat.columns:
        risk_components.append(np.tanh(-feat["ratio_gld_spy_ret21"].fillna(0.0) * 10.0))

    if risk_components:
        feat["sig_macro_risk_on"] = np.mean(risk_components, axis=0)
    else:
        feat["sig_macro_risk_on"] = 0.0

    return feat.replace([np.inf, -np.inf], np.nan)
