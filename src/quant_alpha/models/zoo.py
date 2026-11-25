"""Diverse model zoo: Linear baseline, LightGBM, XGBoost, Random Forest, Neural MLP, and Classical Quant."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler
from xgboost import XGBClassifier, XGBRegressor


MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "linear_ridge": "Ridge / Logistic Baseline",
    "lightgbm": "LightGBM (Leaf-wise GBDT)",
    "xgboost": "XGBoost (Depth-wise GBDT)",
    "extra_trees": "ExtraTrees Ensemble",
    "neural_mlp": "Regularized Neural MLP",
    "classical_quant": "Classical Multi-Signal Quant",
}


class ClassicalSignalRegressor(BaseEstimator, RegressorMixin):
    """Interpretable rule-based quantitative signal model combining trend, MR, momentum, macro & regime."""

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> "ClassicalSignalRegressor":
        self.feature_names_ = list(X.columns) if isinstance(X, pd.DataFrame) else []
        return self

    def _compute_raw_score(self, X: pd.DataFrame) -> np.ndarray:
        if not isinstance(X, pd.DataFrame):
            return np.zeros(len(X), dtype=float)
        trend = X.get("sig_classical_trend", pd.Series(0.0, index=X.index)).fillna(0.0).to_numpy()
        mom = X.get("sig_classical_momentum", pd.Series(0.0, index=X.index)).fillna(0.0).to_numpy()
        mr = X.get("sig_classical_mean_rev", pd.Series(0.0, index=X.index)).fillna(0.0).to_numpy()
        macro = X.get("sig_macro_risk_on", pd.Series(0.0, index=X.index)).fillna(0.0).to_numpy()
        bull_p = X.get("prob_regime_bull", pd.Series(0.33, index=X.index)).fillna(0.33).to_numpy()
        crisis_p = X.get("prob_regime_crisis", pd.Series(0.33, index=X.index)).fillna(0.33).to_numpy()

        # Blend trend + momentum + mean reversion, modulated by macro risk-on & crisis probability
        regime_mult = np.clip(1.0 + 0.35 * bull_p - 0.55 * crisis_p, 0.35, 1.35)
        raw = (0.35 * trend + 0.30 * mom + 0.15 * mr + 0.20 * macro) * regime_mult
        return np.clip(raw, -1.5, 1.5)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Express score in approximate volatility-normalized return units
        return self._compute_raw_score(X) * 0.35


class ClassicalSignalClassifier(BaseEstimator, ClassifierMixin):
    """Probability mapping for the classical quantitative signal baseline."""

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> "ClassicalSignalClassifier":
        self._reg = ClassicalSignalRegressor().fit(X, y)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        score = self._reg._compute_raw_score(X)
        prob_up = 1.0 / (1.0 + np.exp(-1.5 * score))
        prob_up = np.clip(prob_up, 0.05, 0.95)
        return np.column_stack([1.0 - prob_up, prob_up])


def _clip_scaled(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, -8.0, 8.0)


def _make_winsorized_scaler_pipeline(estimator: Any) -> Pipeline:
    """Create leak-free preprocessing pipeline with median imputation, robust scaling, and clipping."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", RobustScaler(quantile_range=(5.0, 95.0))),
            ("clipper", FunctionTransformer(_clip_scaled, validate=False)),
            ("model", estimator),
        ]
    )


def build_model_pair(
    model_name: str,
    params: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
) -> Tuple[Any, Any]:
    """Return (regressor, classifier) pair for the specified model family."""
    p = dict(params or {})

    if model_name == "linear_ridge":
        alpha = float(p.get("alpha", 25.0))
        c_val = float(p.get("C", 0.2))
        reg = _make_winsorized_scaler_pipeline(Ridge(alpha=alpha, random_state=random_state))
        clf = _make_winsorized_scaler_pipeline(
            LogisticRegression(C=c_val, max_iter=500, random_state=random_state)
        )
        return reg, clf

    if model_name == "lightgbm":
        lgb_params = {
            "n_estimators": int(p.get("n_estimators", 220)),
            "learning_rate": float(p.get("learning_rate", 0.03)),
            "num_leaves": int(p.get("num_leaves", 20)),
            "max_depth": int(p.get("max_depth", 5)),
            "min_child_samples": int(p.get("min_child_samples", 40)),
            "subsample": float(p.get("subsample", 0.8)),
            "subsample_freq": 1,
            "colsample_bytree": float(p.get("colsample_bytree", 0.7)),
            "reg_alpha": float(p.get("reg_alpha", 1.0)),
            "reg_lambda": float(p.get("reg_lambda", 5.0)),
            "random_state": random_state,
            "n_jobs": -1,
            "verbose": -1,
        }
        return LGBMRegressor(**lgb_params), LGBMClassifier(**lgb_params)

    if model_name == "xgboost":
        xgb_params = {
            "n_estimators": int(p.get("n_estimators", 220)),
            "learning_rate": float(p.get("learning_rate", 0.03)),
            "max_depth": int(p.get("max_depth", 4)),
            "min_child_weight": float(p.get("min_child_weight", 20.0)),
            "subsample": float(p.get("subsample", 0.8)),
            "colsample_bytree": float(p.get("colsample_bytree", 0.7)),
            "reg_alpha": float(p.get("reg_alpha", 1.0)),
            "reg_lambda": float(p.get("reg_lambda", 5.0)),
            "random_state": random_state,
            "n_jobs": -1,
            "verbosity": 0,
        }
        return (
            XGBRegressor(**xgb_params, objective="reg:squarederror"),
            XGBClassifier(**xgb_params, eval_metric="logloss"),
        )

    if model_name == "extra_trees":
        et_params = {
            "n_estimators": int(p.get("n_estimators", 120)),
            "max_depth": int(p.get("max_depth", 7)),
            "min_samples_leaf": int(p.get("min_samples_leaf", 30)),
            "max_features": float(p.get("max_features", 0.5)),
            "bootstrap": True,
            "max_samples": 0.6,
            "random_state": random_state,
            "n_jobs": -1,
        }
        reg = Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), ("model", ExtraTreesRegressor(**et_params))])
        clf = Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), ("model", ExtraTreesClassifier(**et_params))])
        return reg, clf

    if model_name == "neural_mlp":
        hidden = p.get("hidden_layer_sizes", (48, 24))
        alpha = float(p.get("alpha", 1.5))
        reg = _make_winsorized_scaler_pipeline(
            MLPRegressor(
                hidden_layer_sizes=hidden,
                alpha=alpha,
                batch_size=512,
                learning_rate_init=0.001,
                max_iter=100,
                early_stopping=True,
                n_iter_no_change=10,
                random_state=random_state,
            )
        )
        clf = _make_winsorized_scaler_pipeline(
            MLPClassifier(
                hidden_layer_sizes=hidden,
                alpha=alpha,
                batch_size=512,
                learning_rate_init=0.001,
                max_iter=100,
                early_stopping=True,
                n_iter_no_change=10,
                random_state=random_state,
            )
        )
        return reg, clf

    if model_name == "classical_quant":
        return ClassicalSignalRegressor(), ClassicalSignalClassifier()

    raise ValueError(f"Unknown model_name: {model_name}")
