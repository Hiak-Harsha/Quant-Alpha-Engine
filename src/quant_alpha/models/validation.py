"""Purged and embargoed chronological walk-forward cross-validation for panel financial data."""

from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
import pandas as pd


class PurgedWalkForwardSplit:
    """Walk-forward time splitter on unique timestamps with purge and embargo gaps.

    Ensures:
    1. All assets sharing a timestamp are placed in the same split (no cross-sectional leakage).
    2. Training timestamps end at least `purge_bars` timestamps before the test fold starts,
       preventing multi-bar forward target overlap.
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_bars: int = 10,
        embargo_bars: int = 5,
        min_train_frac: float = 0.45,
    ) -> None:
        self.n_splits = max(2, n_splits)
        self.purge_bars = max(0, purge_bars)
        self.embargo_bars = max(0, embargo_bars)
        self.min_train_frac = float(np.clip(min_train_frac, 0.25, 0.8))

    def split(self, timestamps: pd.Series) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield (train_indices, test_indices) strictly ordered in time."""
        ts_series = pd.to_datetime(timestamps).reset_index(drop=True)
        unique_ts = np.sort(ts_series.unique())
        n_ts = len(unique_ts)
        if n_ts < 40:
            raise ValueError(f"Insufficient unique timestamps ({n_ts}) for purged walk-forward CV.")

        init_train_end = max(int(n_ts * self.min_train_frac), 30)
        rem_ts = n_ts - init_train_end
        if rem_ts < self.n_splits * 5:
            # Adjust folds gracefully on shorter intraday histories
            effective_splits = max(2, rem_ts // 8)
        else:
            effective_splits = self.n_splits

        fold_size = max(5, rem_ts // effective_splits)

        for k in range(effective_splits):
            test_start_pos = init_train_end + k * fold_size
            test_end_pos = n_ts if k == effective_splits - 1 else min(n_ts, test_start_pos + fold_size)
            if test_start_pos >= n_ts or test_end_pos <= test_start_pos:
                continue

            train_end_pos = test_start_pos - self.purge_bars
            if train_end_pos < 20:
                continue

            train_ts_set = set(unique_ts[:train_end_pos])
            test_ts_set = set(unique_ts[test_start_pos:test_end_pos])

            train_idx = np.flatnonzero(ts_series.isin(train_ts_set).to_numpy())
            test_idx = np.flatnonzero(ts_series.isin(test_ts_set).to_numpy())
            if len(train_idx) >= 30 and len(test_idx) >= 5:
                yield train_idx, test_idx
