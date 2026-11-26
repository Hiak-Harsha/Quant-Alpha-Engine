"""Shared portfolio construction layer used by both backtesting and live market scans."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from quant_alpha.config import EngineSettings, HorizonConfig, PortfolioConfig, is_crypto_ticker


def compute_signal_raw_weights(
    pred_vol: np.ndarray,
    prob_up: np.ndarray,
    est_ann_vol: np.ndarray,
    crisis_prob: np.ndarray,
    horizon_cfg: HorizonConfig,
    settings: EngineSettings,
) -> np.ndarray:
    """Compute pre-constraint causal volatility-targeted raw weights shared by backtest and live."""
    vcfg = settings.verdicts
    pv = np.nan_to_num(np.asarray(pred_vol, dtype=float), nan=0.0)
    pu = np.nan_to_num(np.asarray(prob_up, dtype=float), nan=0.5)
    ev = np.clip(np.nan_to_num(np.asarray(est_ann_vol, dtype=float), nan=0.18), 0.05, None)
    cp = np.nan_to_num(np.asarray(crisis_prob, dtype=float), nan=0.2)

    conf = vcfg.conf_base + vcfg.conf_prob_slope * np.abs(pu - 0.5)
    raw_sig = np.where(
        np.abs(pv) < settings.min_signal_threshold,
        0.0,
        np.tanh(pv * vcfg.signal_tanh_scale) * conf,
    )
    vol_scale = np.clip(horizon_cfg.target_vol_annual / ev, vcfg.min_vol_scale, settings.max_leverage)
    regime_scale = np.clip(1.0 - vcfg.crisis_damp_slope * cp, vcfg.crisis_damp_floor, 1.0)
    return np.clip(raw_sig * vol_scale * regime_scale, -settings.max_leverage, settings.max_leverage)


_GOLD_TOKENS = ("GOLD", "GC=F", "GLD")
_SILVER_TOKENS = ("SILVER", "SI=F", "SLV")
_NIFTY_PROXY = {"^NSEI", "NIFTYBEES.NS", "INDA"}
_BANKNIFTY_PROXY = {"^NSEBANK", "BANKBEES.NS"}
_US_INDEX_PROXY = {"SPY", "QQQ", "IWM"}
_US_TREASURY_PROXY = {"TLT", "IEF"}
_CRYPTO_PROXY = {"BTC-USD", "ETH-USD"}

_SECTOR_MAP: Dict[str, str] = {
    "HDFCBANK.NS": "INDIA_BANKS",
    "ICICIBANK.NS": "INDIA_BANKS",
    "SBIN.NS": "INDIA_BANKS",
    "KOTAKBANK.NS": "INDIA_BANKS",
    "AXISBANK.NS": "INDIA_BANKS",
    "INDUSINDBK.NS": "INDIA_BANKS",
    "BANKBEES.NS": "INDIA_BANKS",
    "^NSEBANK": "INDIA_BANKS",
    "BAJFINANCE.NS": "INDIA_NBFC",
    "JIOFIN.NS": "INDIA_NBFC",
    "SHRIRAMFIN.NS": "INDIA_NBFC",
    "TCS.NS": "INDIA_IT",
    "INFY.NS": "INDIA_IT",
    "HCLTECH.NS": "INDIA_IT",
    "WIPRO.NS": "INDIA_IT",
    "TECHM.NS": "INDIA_IT",
    "ITBEES.NS": "INDIA_IT",
    "RELIANCE.NS": "INDIA_ENERGY_INFRA",
    "ONGC.NS": "INDIA_ENERGY_INFRA",
    "NTPC.NS": "INDIA_ENERGY_INFRA",
    "POWERGRID.NS": "INDIA_ENERGY_INFRA",
    "COALINDIA.NS": "INDIA_ENERGY_INFRA",
    "LT.NS": "INDIA_ENERGY_INFRA",
    "ADANIENT.NS": "INDIA_ENERGY_INFRA",
    "ADANIPORTS.NS": "INDIA_ENERGY_INFRA",
    "HAL.NS": "INDIA_DEFENCE_AUTO",
    "BEL.NS": "INDIA_DEFENCE_AUTO",
    "TATAMOTORS.NS": "INDIA_DEFENCE_AUTO",
    "M&M.NS": "INDIA_DEFENCE_AUTO",
    "MARUTI.NS": "INDIA_DEFENCE_AUTO",
    "BAJAJ-AUTO.NS": "INDIA_DEFENCE_AUTO",
    "BHARTIARTL.NS": "INDIA_CONSUMER_TELECOM",
    "ITC.NS": "INDIA_CONSUMER_TELECOM",
    "HINDUNILVR.NS": "INDIA_CONSUMER_TELECOM",
    "TITAN.NS": "INDIA_CONSUMER_TELECOM",
    "TRENT.NS": "INDIA_CONSUMER_TELECOM",
    "ETERNAL.NS": "INDIA_CONSUMER_TELECOM",
    "ASIANPAINT.NS": "INDIA_CONSUMER_TELECOM",
    "SUNPHARMA.NS": "INDIA_PHARMA_METALS",
    "TATASTEEL.NS": "INDIA_PHARMA_METALS",
    "HINDALCO.NS": "INDIA_PHARMA_METALS",
    "JSWSTEEL.NS": "INDIA_PHARMA_METALS",
    "VEDL.NS": "INDIA_PHARMA_METALS",
    "HINDCOPPER.NS": "INDIA_PHARMA_METALS",
}


def is_cash_long_only_asset(ticker: str, portfolio_cfg: Optional[PortfolioConfig] = None) -> bool:
    """Return True when cash delivery / index constraints disallow shorting the ticker."""
    cfg = portfolio_cfg or PortfolioConfig()
    if not cfg.long_only_cash_india:
        return False
    upper = ticker.upper()
    return upper.endswith(cfg.cash_long_only_suffixes) or upper in cfg.cash_long_only_tickers


def get_exposure_cluster(ticker: str) -> str:
    """Group duplicate or near-identical underlying instruments under one cluster ID."""
    upper = ticker.upper()
    if any(tok in upper for tok in _GOLD_TOKENS):
        return "CLUSTER_GOLD"
    if any(tok in upper for tok in _SILVER_TOKENS):
        return "CLUSTER_SILVER"
    if upper in _NIFTY_PROXY:
        return "CLUSTER_NIFTY"
    if upper in _BANKNIFTY_PROXY:
        return "CLUSTER_BANKNIFTY"
    if upper in _US_INDEX_PROXY:
        return "CLUSTER_US_EQUITY"
    if upper in _US_TREASURY_PROXY:
        return "CLUSTER_US_RATES"
    if upper in _CRYPTO_PROXY or is_crypto_ticker(upper):
        return "CLUSTER_CRYPTO"
    return f"ASSET_{upper}"


def get_sector_bucket(ticker: str) -> str:
    """Map ticker to broad sector/asset-class cap bucket."""
    upper = ticker.upper()
    if any(tok in upper for tok in _GOLD_TOKENS) or any(tok in upper for tok in _SILVER_TOKENS):
        return "PRECIOUS_METALS"
    if upper in _SECTOR_MAP:
        return _SECTOR_MAP[upper]
    if upper in _NIFTY_PROXY:
        return "INDIA_BROAD_INDEX"
    if upper.endswith((".NS", ".BO")):
        return "INDIA_OTHER"
    if is_crypto_ticker(upper):
        return "CRYPTO"
    if upper in _US_INDEX_PROXY or upper in {"NVDA", "MSFT", "AAPL", "AMZN", "META", "GOOGL"}:
        return "US_EQUITY"
    if upper in {"TLT", "IEF", "HYG", "UUP", "USO"}:
        return "GLOBAL_MACRO"
    return "OTHER"


def _scale_groups_to_cap(weights: pd.Series, group_labels: pd.Series, max_group_gross: float) -> pd.Series:
    """Proportionally scale down any group whose sum of absolute weights exceeds max_group_gross."""
    if max_group_gross <= 0.0 or weights.empty:
        return weights * 0.0
    out = weights.copy()
    for _, idx in out.groupby(group_labels).groups.items():
        grp_gross = float(out.loc[idx].abs().sum())
        if grp_gross > max_group_gross and grp_gross > 1e-12:
            out.loc[idx] = out.loc[idx] * (max_group_gross / grp_gross)
    return out


def apply_portfolio_constraints(
    raw_weights: pd.Series,
    portfolio_cfg: Optional[PortfolioConfig] = None,
) -> pd.Series:
    """Project raw cross-asset weights onto valid cash-long-only, single, cluster, sector, and gross/net caps."""
    cfg = portfolio_cfg or PortfolioConfig()
    if raw_weights.empty:
        return raw_weights.astype(float)

    w = raw_weights.astype(float).fillna(0.0).copy()
    tickers = pd.Index([str(t) for t in w.index])

    # 1. Enforce long-only on cash Indian equities/ETFs (cannot short delivery cash)
    long_only_mask = np.array([is_cash_long_only_asset(t, cfg) for t in tickers], dtype=bool)
    if long_only_mask.any():
        vals = w.to_numpy(copy=True)
        vals[long_only_mask] = np.maximum(0.0, vals[long_only_mask])
        w = pd.Series(vals, index=w.index)

    # 2. Single-asset cap
    if cfg.max_single_asset_weight > 0.0:
        w = w.clip(-cfg.max_single_asset_weight, cfg.max_single_asset_weight)

    # 3. Duplicate-exposure cluster cap
    cluster_labels = pd.Series([get_exposure_cluster(t) for t in tickers], index=w.index)
    w = _scale_groups_to_cap(w, cluster_labels, cfg.max_cluster_weight)

    # 4. Sector/asset-class concentration cap
    sector_labels = pd.Series([get_sector_bucket(t) for t in tickers], index=w.index)
    w = _scale_groups_to_cap(w, sector_labels, cfg.max_sector_weight)

    # 5. Total portfolio gross and net exposure caps
    gross = float(w.abs().sum())
    if gross > cfg.max_gross_exposure and gross > 1e-12:
        w = w * (cfg.max_gross_exposure / gross)

    net = float(w.sum())
    if abs(net) > cfg.max_net_exposure and abs(net) > 1e-12:
        w = w * (cfg.max_net_exposure / abs(net))

    return w


def construct_portfolio_weights_df(
    raw_weights_wide: pd.DataFrame,
    portfolio_cfg: Optional[PortfolioConfig] = None,
) -> pd.DataFrame:
    """Apply identical cross-sectional portfolio construction constraints at every timestamp."""
    if raw_weights_wide.empty:
        return raw_weights_wide.copy()

    cfg = portfolio_cfg or PortfolioConfig()
    cols = list(raw_weights_wide.columns)
    long_only_mask = np.array([is_cash_long_only_asset(str(t), cfg) for t in cols], dtype=bool)

    arr = raw_weights_wide.to_numpy(dtype=float, copy=True)
    arr = np.nan_to_num(arr, nan=0.0)

    # 1. Long-only on Indian cash stocks/ETFs
    if long_only_mask.any():
        arr[:, long_only_mask] = np.maximum(0.0, arr[:, long_only_mask])

    # 2. Single-asset cap
    if cfg.max_single_asset_weight > 0.0:
        arr = np.clip(arr, -cfg.max_single_asset_weight, cfg.max_single_asset_weight)

    # 3. Exposure cluster caps
    cluster_groups: Dict[str, list[int]] = {}
    for j, t in enumerate(cols):
        cluster_groups.setdefault(get_exposure_cluster(str(t)), []).append(j)
    for idxs in cluster_groups.values():
        if len(idxs) > 1 or cfg.max_cluster_weight < cfg.max_single_asset_weight:
            grp_gross = np.abs(arr[:, idxs]).sum(axis=1)
            scale = np.where(grp_gross > cfg.max_cluster_weight, cfg.max_cluster_weight / np.maximum(grp_gross, 1e-12), 1.0)
            arr[:, idxs] *= scale[:, None]

    # 4. Sector concentration caps
    sector_groups: Dict[str, list[int]] = {}
    for j, t in enumerate(cols):
        sector_groups.setdefault(get_sector_bucket(str(t)), []).append(j)
    for idxs in sector_groups.values():
        grp_gross = np.abs(arr[:, idxs]).sum(axis=1)
        scale = np.where(grp_gross > cfg.max_sector_weight, cfg.max_sector_weight / np.maximum(grp_gross, 1e-12), 1.0)
        arr[:, idxs] *= scale[:, None]

    # 5. Gross & net exposure caps
    gross = np.abs(arr).sum(axis=1)
    gross_scale = np.where(gross > cfg.max_gross_exposure, cfg.max_gross_exposure / np.maximum(gross, 1e-12), 1.0)
    arr *= gross_scale[:, None]

    net = np.abs(arr.sum(axis=1))
    net_scale = np.where(net > cfg.max_net_exposure, cfg.max_net_exposure / np.maximum(net, 1e-12), 1.0)
    arr *= net_scale[:, None]

    return pd.DataFrame(arr, index=raw_weights_wide.index, columns=raw_weights_wide.columns)
