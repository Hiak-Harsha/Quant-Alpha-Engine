"""Causal market-level Hidden Markov Regime detector with forward-only filtering."""

from __future__ import annotations

import logging
import warnings
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

REGIME_LABELS = {
    0: "Low-Vol Bull / Calm",
    1: "Normal / Transition",
    2: "High-Vol Stress / Crisis",
}


class MarketRegimeDetector:
    """Causal 3-state regime detector fit on chronological market-level series."""

    def __init__(self, n_states: int = 3, random_state: int = 42) -> None:
        self.n_states = n_states
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.hmm = None
        self.state_order_: List[int] = list(range(n_states))
        self.vol_thresholds_: Optional[np.ndarray] = None
        self.used_hmm_: bool = False
        self.regime_cols_: List[str] = []

    def _select_regime_inputs(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in ("ret_5", "rv_21", "vol_ratio_5_21", "vix_level") if c in df.columns]
        if not cols:
            cols = [c for c in df.columns if str(c).startswith(("ret_", "rv_"))][:3]
        if not cols:
            return pd.DataFrame(np.zeros((len(df), 2)), index=df.index, columns=["f0", "f1"])
        self.regime_cols_ = cols
        sub = df[cols].copy().ffill().bfill().fillna(0.0)
        return sub

    def _aggregate_market_series(
        self,
        feat_df: pd.DataFrame,
        timestamps: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Collapse panel rows into one chronological market series per unique Timestamp."""
        if timestamps is None:
            return feat_df.reset_index(drop=True)
        ts = pd.to_datetime(timestamps).reset_index(drop=True)
        tmp = feat_df.reset_index(drop=True).copy()
        tmp["_ts"] = ts
        mkt = tmp.groupby("_ts", sort=True).median(numeric_only=True)
        return mkt

    def _forward_filter_probs(self, arr_scaled: np.ndarray) -> np.ndarray:
        """Strictly causal HMM forward filter: P(s_t | x_{1:t}) using zero future bars."""
        model = self.hmm
        if model is None:
            raise RuntimeError("HMM model is not initialized.")
        log_lik = model._compute_log_likelihood(arr_scaled)
        startprob = np.clip(model.startprob_, 1e-12, 1.0)
        transmat = np.clip(model.transmat_, 1e-12, 1.0)
        log_start = np.log(startprob)
        log_trans = np.log(transmat)

        t_len, n_comp = log_lik.shape
        probs = np.empty((t_len, n_comp), dtype=float)
        log_alpha = log_start + log_lik[0]
        log_alpha -= logsumexp(log_alpha)
        probs[0] = np.exp(log_alpha)

        for t in range(1, t_len):
            log_alpha = logsumexp(log_alpha[:, None] + log_trans, axis=0) + log_lik[t]
            log_alpha -= logsumexp(log_alpha)
            probs[t] = np.exp(log_alpha)
        return probs

    def fit(
        self,
        X_train: pd.DataFrame,
        timestamps: Optional[pd.Series] = None,
    ) -> "MarketRegimeDetector":
        """Fit HMM on chronological market series from the training fold only."""
        sub = self._select_regime_inputs(X_train)
        mkt_series = self._aggregate_market_series(sub, timestamps=timestamps)
        arr_scaled = self.scaler.fit_transform(mkt_series.to_numpy(dtype=float))

        vol_col = mkt_series["rv_21"] if "rv_21" in mkt_series.columns else pd.Series(np.zeros(len(mkt_series)))
        vol_clean = vol_col.dropna()
        if len(vol_clean) >= 30:
            self.vol_thresholds_ = np.quantile(vol_clean, [0.35, 0.78])
        else:
            self.vol_thresholds_ = np.array([0.14, 0.25])

        self.used_hmm_ = False
        if len(arr_scaled) >= 80:
            try:
                from hmmlearn.hmm import GaussianHMM

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = GaussianHMM(
                        n_components=self.n_states,
                        covariance_type="diag",
                        n_iter=150,
                        random_state=self.random_state,
                    )
                    model.fit(arr_scaled)
                    vol_idx = 1 if arr_scaled.shape[1] > 1 else 0
                    state_vols = model.means_[:, vol_idx]
                    self.state_order_ = [int(i) for i in np.argsort(state_vols)]
                    self.hmm = model
                    self.used_hmm_ = True
            except Exception as exc:
                logger.debug("GaussianHMM fallback triggered: %s", exc)
                self.hmm = None

        return self

    def predict_proba(
        self,
        X: pd.DataFrame,
        timestamps: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Return strictly causal ordered regime state & probabilities aligned to X.index."""
        sub = self._select_regime_inputs(X)
        n = len(X)

        if self.used_hmm_ and self.hmm is not None:
            try:
                if timestamps is not None:
                    ts = pd.to_datetime(timestamps).reset_index(drop=True)
                    mkt_series = self._aggregate_market_series(sub, timestamps=ts)
                    arr_scaled = self.scaler.transform(mkt_series.to_numpy(dtype=float))
                    raw_probs = self._forward_filter_probs(arr_scaled)
                    ordered = raw_probs[:, self.state_order_]
                    mkt_out = pd.DataFrame(
                        {
                            "regime_state": np.argmax(ordered, axis=1).astype(float),
                            "prob_regime_bull": ordered[:, 0],
                            "prob_regime_chop": ordered[:, 1],
                            "prob_regime_crisis": ordered[:, 2],
                        },
                        index=mkt_series.index,
                    )
                    mapped = mkt_out.reindex(ts).reset_index(drop=True)
                    mapped.index = X.index
                    return mapped

                arr_scaled = self.scaler.transform(sub.to_numpy(dtype=float))
                raw_probs = self._forward_filter_probs(arr_scaled)
                ordered_probs = raw_probs[:, self.state_order_]
                states = np.argmax(ordered_probs, axis=1)
                return pd.DataFrame(
                    {
                        "regime_state": states.astype(float),
                        "prob_regime_bull": ordered_probs[:, 0],
                        "prob_regime_chop": ordered_probs[:, 1],
                        "prob_regime_crisis": ordered_probs[:, 2],
                    },
                    index=X.index,
                )
            except Exception:
                pass

        # Deterministic quantile fallback
        vol = X["rv_21"].ffill().fillna(0.18).to_numpy() if "rv_21" in X.columns else np.full(n, 0.18)
        q_low, q_high = self.vol_thresholds_ if self.vol_thresholds_ is not None else (0.14, 0.25)
        states = np.where(vol <= q_low, 0, np.where(vol >= q_high, 2, 1))
        p_bull = np.clip(1.0 - (vol - q_low) / max(q_high - q_low, 1e-4), 0.05, 0.9)
        p_crisis = np.clip((vol - q_low) / max(q_high - q_low, 1e-4), 0.05, 0.9)
        p_chop = np.maximum(0.1, 1.0 - np.abs(p_bull - p_crisis))
        total = p_bull + p_chop + p_crisis
        return pd.DataFrame(
            {
                "regime_state": states.astype(float),
                "prob_regime_bull": p_bull / total,
                "prob_regime_chop": p_chop / total,
                "prob_regime_crisis": p_crisis / total,
            },
            index=X.index,
        )
