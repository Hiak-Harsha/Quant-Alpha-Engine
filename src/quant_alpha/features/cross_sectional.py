"""Strictly causal cross-sectional peer rankings and duplicate-cluster deduplication."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from quant_alpha.portfolio.construct import get_exposure_cluster

_RANK_SOURCE_COLS = (
    ("ret_1", "cs_rank_ret1"),
    ("ret_5", "cs_rank_ret5"),
    ("ret_21", "cs_rank_ret21"),
    ("rv_21", "cs_rank_rv21"),
    ("rsi_14", "cs_rank_rsi14"),
    ("amihud_21", "cs_rank_amihud21"),
    ("idio_ret_21", "cs_rank_idio21"),
)

_REL_DIFF_COLS = (
    ("ret_5", "cs_rel_ret5_diff"),
    ("ret_21", "cs_rel_ret21_diff"),
)

CROSS_SECTIONAL_FEATURE_COLS: List[str] = [
    out_c for _, out_c in _RANK_SOURCE_COLS
] + [out_c for _, out_c in _REL_DIFF_COLS]


def add_cross_sectional_features(panel_df: pd.DataFrame) -> pd.DataFrame:
    """Compute contemporaneous cross-sectional peer ranks and market-neutral diffs at each Timestamp."""
    if panel_df.empty or "Timestamp" not in panel_df.columns:
        return panel_df

    out = panel_df.copy()
    ts_key = pd.to_datetime(out["Timestamp"])
    group_sizes = out.groupby(ts_key)["ticker"].transform("count") if "ticker" in out.columns else pd.Series(len(out), index=out.index)

    for src_col, dst_col in _RANK_SOURCE_COLS:
        if src_col in out.columns:
            ranked = out.groupby(ts_key)[src_col].rank(pct=True, method="average")
            out[dst_col] = np.where(group_sizes >= 2, ranked.fillna(0.5), 0.5)
        else:
            out[dst_col] = 0.5

    for src_col, dst_col in _REL_DIFF_COLS:
        if src_col in out.columns:
            med = out.groupby(ts_key)[src_col].transform("median")
            out[dst_col] = np.where(group_sizes >= 2, (out[src_col] - med).fillna(0.0), 0.0)
        else:
            out[dst_col] = 0.0

    return out


_PREFERRED_CLUSTER_CANONICAL: Dict[str, str] = {
    "CLUSTER_GOLD": "GOLDBEES.NS",
    "CLUSTER_SILVER": "SILVERBEES.NS",
    "CLUSTER_NIFTY": "^NSEI",
    "CLUSTER_BANKNIFTY": "^NSEBANK",
    "CLUSTER_US_EQUITY": "SPY",
    "CLUSTER_US_RATES": "TLT",
    "CLUSTER_CRYPTO": "BTC-USD",
}


def select_canonical_tickers(tickers: List[str]) -> List[str]:
    """Return one canonical representative per duplicate exposure cluster to prevent CV inflation."""
    by_cluster: Dict[str, List[str]] = {}
    for t in tickers:
        by_cluster.setdefault(get_exposure_cluster(t), []).append(t)

    canonical: List[str] = []
    for cl, members in by_cluster.items():
        pref = _PREFERRED_CLUSTER_CANONICAL.get(cl)
        if pref and pref in members:
            canonical.append(pref)
        else:
            canonical.append(members[0])
    return canonical
