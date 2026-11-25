"""Leak-free unsupervised market regime detector (Gaussian HMM with quantile fallback)."""

from __future__ import annotations

import logging
import warnings
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

REGIME_LABELS = {
    0: "Low-Vol Bull / Calm",
    1: "Normal / Transition",
    2: "High-Vol Stress / Crisis",
}


class MarketRegimeDetector:
    """Causal 3-state regime detector fit strictly on training folds."""

    def __init__(self, n_states: int = 3, random_state: int = 42) -> None:
        self.n_states = n_states
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.hmm = None
        self.state_order_: List[int] = list(range(n_states))
        self.vol_thresholds_: Optional[np.ndarray] = None
        self.used_hmm_: bool = False

    def _select_regime_inputs(self, df: pd.DataFrame) -> np.ndarray:
        cols = [c for c in ("ret_5", "rv_21", "vol_ratio_5_21", "vix_level") if c in df.columns]
        if not cols:
            cols = [c for c in df.columns if c.startswith(("ret_", "rv_"))][:3]
        sub = df[cols].copy() if cols else pd.DataFrame(np.zeros((len(df), 2)), index=df.index)
        sub = sub.ffill().bfill().fillna(0.0)
        return sub.to_numpy(dtype=float)

    def fit(self, X_train: pd.DataFrame) -> "MarketRegimeDetector":
        """Fit HMM or fallback volatility quantiles on training data only."""
        arr = self._select_regime_inputs(X_train)
        arr_scaled = self.scaler.fit_transform(arr)

        # Fallback volatility thresholds from training distribution
        vol_col = X_train["rv_21"] if "rv_21" in X_train.columns else pd.Series(np.zeros(len(X_train)))
        vol_clean = vol_col.dropna()
        if len(vol_clean) >= 30:
            self.vol_thresholds_ = np.quantile(vol_clean, [0.35, 0.78])
        else:
            self.vol_thresholds_ = np.array([0.14, 0.25])

        self.used_hmm_ = False
        if len(arr_scaled) >= 120:
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
                    # Order states from lowest volatility (index 1 in inputs if rv_21 present) to highest
                    vol_idx = 1 if arr_scaled.shape[1] > 1 else 0
                    state_vols = model.means_[:, vol_idx]
                    self.state_order_ = [int(i) for i in np.argsort(state_vols)]
                    self.hmm = model
                    self.used_hmm_ = True
            except Exception as exc:
                logger.debug("GaussianHMM fallback triggered: %s", exc)
                self.hmm = None

        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return ordered regime state and state probabilities for rows in X."""
        arr = self._select_regime_inputs(X)
        arr_scaled = self.scaler.transform(arr)
        n = len(X)

        if self.used_hmm_ and self.hmm is not None:
            try:
                raw_probs = self.hmm.predict_proba(arr_scaled)
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
