"""Optuna hyperparameter tuning using strictly purged chronological inner walk-forward CV."""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import optuna
import pandas as pd
from scipy.stats import spearmanr

from quant_alpha.models.validation import PurgedWalkForwardSplit
from quant_alpha.models.zoo import build_model_pair

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)


def _safe_spearman_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 20 or np.std(y_pred[mask]) < 1e-10:
        return 0.0
    corr, _ = spearmanr(y_true[mask], y_pred[mask])
    return float(corr) if np.isfinite(corr) else 0.0


def _tune_single_family(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    timestamps: pd.Series,
    purge_bars: int,
    embargo_bars: int,
    n_trials: int,
    random_state: int,
) -> Dict[str, Any]:
    splitter = PurgedWalkForwardSplit(
        n_splits=3,
        purge_bars=purge_bars,
        embargo_bars=embargo_bars,
        min_train_frac=0.45,
    )
    try:
        folds = list(splitter.split(timestamps))
    except ValueError:
        return {}
    if not folds:
        return {}

    def objective(trial: optuna.Trial) -> float:
        if model_name == "lightgbm":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 80, 320),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 8, 36),
                "max_depth": trial.suggest_int("max_depth", 3, 7),
                "min_child_samples": trial.suggest_int("min_child_samples", 20, 90),
                "subsample": trial.suggest_float("subsample", 0.6, 0.95),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.95),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 20.0, log=True),
            }
        elif model_name == "xgboost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 80, 300),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 6),
                "min_child_weight": trial.suggest_float("min_child_weight", 5.0, 60.0),
                "subsample": trial.suggest_float("subsample", 0.6, 0.95),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.95),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 25.0, log=True),
            }
        elif model_name == "linear_ridge":
            params = {
                "alpha": trial.suggest_float("alpha", 1.0, 500.0, log=True),
                "C": trial.suggest_float("C", 0.01, 2.0, log=True),
            }
        else:
            return 0.0

        ics = []
        for tr_idx, va_idx in folds:
            reg, _ = build_model_pair(model_name, params=params, random_state=random_state)
            reg.fit(X_train.iloc[tr_idx], y_train[tr_idx])
            preds = reg.predict(X_train.iloc[va_idx])
            ics.append(_safe_spearman_ic(y_train[va_idx], preds))

        mean_ic = float(np.mean(ics))
        std_ic = float(np.std(ics))
        # Reward consistent positive rank IC across folds rather than one lucky spike
        return mean_ic - 0.25 * std_ic

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=max(5, n_trials), show_progress_bar=False)
    return dict(study.best_params)


def tune_hyperparameters(
    X: pd.DataFrame,
    y_vol_ret: np.ndarray,
    timestamps: pd.Series,
    purge_bars: int,
    embargo_bars: int,
    n_trials: int = 12,
    random_state: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """Tune key model families strictly on the initial training window to prevent test leakage."""
    ts = pd.to_datetime(timestamps).reset_index(drop=True)
    unique_ts = np.sort(ts.unique())
    # Tune strictly on the earliest 60% of timeline so later walk-forward test folds remain unseen
    cutoff_idx = max(int(len(unique_ts) * 0.6), min(len(unique_ts) - 1, 40))
    cutoff_ts = unique_ts[cutoff_idx]
    mask = (ts <= cutoff_ts).to_numpy()

    X_sub = X.iloc[mask].reset_index(drop=True)
    y_sub = y_vol_ret[mask]
    ts_sub = ts.iloc[mask].reset_index(drop=True)

    best_params: Dict[str, Dict[str, Any]] = {}
    for family in ("linear_ridge", "lightgbm", "xgboost"):
        try:
            best_params[family] = _tune_single_family(
                model_name=family,
                X_train=X_sub,
                y_train=y_sub,
                timestamps=ts_sub,
                purge_bars=purge_bars,
                embargo_bars=embargo_bars,
                n_trials=n_trials,
                random_state=random_state,
            )
            logger.info("Tuned %s params: %s", family, best_params[family])
        except Exception as exc:
            logger.warning("Tuning failed for %s (%s); using defaults.", family, exc)
            best_params[family] = {}

    return best_params
