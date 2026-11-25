"""Walk-forward multi-model training, dynamic regime fitting, ensemble blending, and SHAP explainability."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from quant_alpha.config import ARTIFACTS_DIR, HorizonConfig, ensure_directories
from quant_alpha.features.pipeline import PreparedDataset
from quant_alpha.features.regime import MarketRegimeDetector
from quant_alpha.models.tuner import tune_hyperparameters
from quant_alpha.models.validation import PurgedWalkForwardSplit
from quant_alpha.models.zoo import build_model_pair

logger = logging.getLogger(__name__)

DEFAULT_MODELS: Tuple[str, ...] = (
    "linear_ridge",
    "lightgbm",
    "xgboost",
    "extra_trees",
    "neural_mlp",
    "classical_quant",
)


@dataclass
class MultiModelEnsembleResult:
    """Artifacts and out-of-sample walk-forward predictions from multi-model training."""

    horizon_name: str
    oof_predictions: pd.DataFrame
    feature_cols: List[str]
    model_names: List[str]
    ensemble_weights: Dict[str, float]
    tuned_params: Dict[str, Dict[str, Any]]
    final_models: Dict[str, Tuple[Any, Any]]
    final_regime_detector: MarketRegimeDetector
    ticker_to_id: Dict[str, float] = field(default_factory=dict)
    shap_importance: pd.DataFrame = field(default_factory=pd.DataFrame)


def _compute_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 20 or np.std(y_pred[mask]) < 1e-10:
        return 0.0
    corr, _ = spearmanr(y_true[mask], y_pred[mask])
    return float(corr) if np.isfinite(corr) else 0.0


def _learn_ensemble_weights(
    oof_df: pd.DataFrame,
    model_names: List[str],
    target_col: str = "target_vol_norm_ret",
) -> Dict[str, float]:
    """Compute non-negative out-of-fold weights emphasizing positive IC & Ridge meta-blending."""
    pred_cols = [f"pred_vol_{m}" for m in model_names]
    valid = oof_df.dropna(subset=[target_col] + pred_cols)
    if len(valid) < 50:
        w = 1.0 / len(model_names)
        return {m: w for m in model_names}

    # Combine positive-clipped IC score and non-negative Ridge meta-regression
    ics = np.array([max(0.0, _compute_ic(valid[target_col].to_numpy(), valid[f"pred_vol_{m}"].to_numpy())) for m in model_names])
    try:
        meta = Ridge(alpha=50.0, positive=True, fit_intercept=False)
        meta.fit(valid[pred_cols].to_numpy(), valid[target_col].to_numpy())
        ridge_w = np.maximum(0.0, meta.coef_)
    except Exception:
        ridge_w = np.ones(len(model_names))

    ic_norm = ics / ics.sum() if ics.sum() > 1e-6 else np.ones(len(model_names)) / len(model_names)
    ridge_norm = ridge_w / ridge_w.sum() if ridge_w.sum() > 1e-6 else np.ones(len(model_names)) / len(model_names)

    # Blend 50% IC weighting + 50% positive Ridge weights, with small floor for diversity
    blended = 0.5 * ic_norm + 0.5 * ridge_norm
    blended = 0.85 * blended + 0.15 * (1.0 / len(model_names))
    blended = blended / blended.sum()
    return {m: round(float(w), 4) for m, w in zip(model_names, blended)}


def _compute_global_shap(
    lgb_regressor: Any,
    X_sample: pd.DataFrame,
) -> pd.DataFrame:
    """Compute mean |SHAP| feature importance using TreeExplainer with safe fallback."""
    if X_sample.empty:
        return pd.DataFrame()
    sample = X_sample.tail(min(600, len(X_sample))).copy()
    try:
        import shap

        explainer = shap.TreeExplainer(lgb_regressor)
        shap_vals = explainer.shap_values(sample)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        mean_abs = np.abs(shap_vals).mean(axis=0)
        imp = pd.DataFrame({"feature": sample.columns, "mean_abs_shap": mean_abs})
        return imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    except Exception:
        if hasattr(lgb_regressor, "feature_importances_"):
            imp = pd.DataFrame(
                {"feature": sample.columns, "mean_abs_shap": lgb_regressor.feature_importances_}
            )
            return imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        return pd.DataFrame()


def train_and_evaluate_walkforward(
    dataset: PreparedDataset,
    horizon_cfg: HorizonConfig,
    model_names: Optional[List[str]] = None,
    tune: bool = True,
    n_optuna_trials: int = 12,
    random_state: int = 42,
    save_artifacts: bool = True,
) -> MultiModelEnsembleResult:
    """Run leak-free purged walk-forward training, regime detection, tuning, and ensemble stacking."""
    models = list(model_names or DEFAULT_MODELS)
    panel = dataset.panel_df.copy().reset_index(drop=True)
    base_features = list(dataset.feature_cols)

    # Initial causal regime features on tuning subset only inside tuner
    X_base = panel[base_features].copy()
    y_vol = panel[dataset.target_vol_ret_col].to_numpy(dtype=float)
    y_dir = panel[dataset.target_dir_col].to_numpy(dtype=int)
    timestamps = panel["Timestamp"]

    tuned_params: Dict[str, Dict[str, Any]] = {m: {} for m in models}
    if tune:
        tuned_params.update(
            tune_hyperparameters(
                X=X_base,
                y_vol_ret=y_vol,
                timestamps=timestamps,
                purge_bars=horizon_cfg.purge_bars,
                embargo_bars=horizon_cfg.embargo_bars,
                n_trials=n_optuna_trials,
                random_state=random_state,
            )
        )

    splitter = PurgedWalkForwardSplit(
        n_splits=horizon_cfg.n_splits,
        purge_bars=horizon_cfg.purge_bars,
        embargo_bars=horizon_cfg.embargo_bars,
        min_train_frac=0.42,
    )

    regime_cols = ["regime_state", "prob_regime_bull", "prob_regime_chop", "prob_regime_crisis"]
    full_features = base_features + regime_cols

    oof_chunks: List[pd.DataFrame] = []

    for fold_id, (tr_idx, te_idx) in enumerate(splitter.split(timestamps)):
        logger.info("Walk-forward fold %d: train=%d rows, test=%d rows", fold_id + 1, len(tr_idx), len(te_idx))
        # Fit HMM regime detector strictly on training fold to prevent regime leakage
        reg_detector = MarketRegimeDetector(random_state=random_state).fit(X_base.iloc[tr_idx])
        tr_regime = reg_detector.predict_proba(X_base.iloc[tr_idx])
        te_regime = reg_detector.predict_proba(X_base.iloc[te_idx])

        X_tr = pd.concat([X_base.iloc[tr_idx].reset_index(drop=True), tr_regime.reset_index(drop=True)], axis=1)
        X_te = pd.concat([X_base.iloc[te_idx].reset_index(drop=True), te_regime.reset_index(drop=True)], axis=1)

        fold_out = panel.loc[
            te_idx,
            [
                "Timestamp",
                "ticker",
                "close_price",
                "bar_ret_1",
                "est_ann_vol",
                "horizon_vol",
                "target_fwd_ret",
                "target_vol_norm_ret",
                "target_up",
                "sig_classical_trend",
                "sig_classical_mean_rev",
                "sig_classical_momentum",
                "sig_macro_risk_on",
            ],
        ].copy().reset_index(drop=True)
        fold_out["fold"] = fold_id
        for rc in regime_cols:
            fold_out[rc] = te_regime[rc].to_numpy()

        for m in models:
            reg_model, clf_model = build_model_pair(m, params=tuned_params.get(m), random_state=random_state)
            reg_model.fit(X_tr, y_vol[tr_idx])
            clf_model.fit(X_tr, y_dir[tr_idx])

            pred_vol = np.clip(reg_model.predict(X_te), -3.0, 3.0)
            prob_up = clf_model.predict_proba(X_te)[:, 1]
            pred_ret = pred_vol * fold_out["horizon_vol"].to_numpy()

            fold_out[f"pred_vol_{m}"] = pred_vol
            fold_out[f"pred_ret_{m}"] = pred_ret
            fold_out[f"prob_up_{m}"] = prob_up

        oof_chunks.append(fold_out)

    if not oof_chunks:
        raise ValueError("Purged walk-forward split yielded zero valid evaluation folds.")

    oof_df = pd.concat(oof_chunks, ignore_index=True).sort_values(["Timestamp", "ticker"]).reset_index(drop=True)

    # Strictly causal fold-by-fold ensemble weighting: fold k uses weights learned ONLY on folds < k
    oof_df["pred_vol_ensemble"] = np.nan
    oof_df["prob_up_ensemble"] = np.nan
    equal_w = {m: 1.0 / len(models) for m in models}
    unique_folds = sorted(oof_df["fold"].unique())

    for fid in unique_folds:
        fold_mask = oof_df["fold"] == fid
        prior_mask = oof_df["fold"] < fid
        if prior_mask.sum() >= 50:
            fold_w = _learn_ensemble_weights(oof_df.loc[prior_mask], models, target_col="target_vol_norm_ret")
        else:
            fold_w = equal_w
        w_vec_f = np.array([fold_w[m] for m in models])
        vol_mat_f = np.column_stack([oof_df.loc[fold_mask, f"pred_vol_{m}"].to_numpy() for m in models])
        prob_mat_f = np.column_stack([oof_df.loc[fold_mask, f"prob_up_{m}"].to_numpy() for m in models])
        oof_df.loc[fold_mask, "pred_vol_ensemble"] = vol_mat_f @ w_vec_f
        oof_df.loc[fold_mask, "prob_up_ensemble"] = prob_mat_f @ w_vec_f

    # Production ensemble weights learned across all OOF history for future live inference
    weights = _learn_ensemble_weights(oof_df, models, target_col="target_vol_norm_ret")

    vol_preds_mat = np.column_stack([oof_df[f"pred_vol_{m}"].to_numpy() for m in models])
    oof_df["pred_ret_ensemble"] = oof_df["pred_vol_ensemble"] * oof_df["horizon_vol"]
    oof_df["model_disagreement_std"] = np.std(vol_preds_mat, axis=1)
    ens_sign = np.sign(oof_df["pred_vol_ensemble"].to_numpy())[:, None]
    oof_df["model_agreement_pct"] = (np.sign(vol_preds_mat) == ens_sign).mean(axis=1) * 100.0

    # Fit final production models on 100% of available labeled history
    final_regime = MarketRegimeDetector(random_state=random_state).fit(X_base)
    full_regime_df = final_regime.predict_proba(X_base)
    X_full = pd.concat([X_base.reset_index(drop=True), full_regime_df.reset_index(drop=True)], axis=1)

    final_models: Dict[str, Tuple[Any, Any]] = {}
    for m in models:
        reg_model, clf_model = build_model_pair(m, params=tuned_params.get(m), random_state=random_state)
        reg_model.fit(X_full, y_vol)
        clf_model.fit(X_full, y_dir)
        final_models[m] = (reg_model, clf_model)

    shap_df = pd.DataFrame()
    if "lightgbm" in final_models:
        shap_df = _compute_global_shap(final_models["lightgbm"][0], X_full)

    result = MultiModelEnsembleResult(
        horizon_name=horizon_cfg.name,
        oof_predictions=oof_df,
        feature_cols=full_features,
        model_names=models,
        ensemble_weights=weights,
        tuned_params=tuned_params,
        final_models=final_models,
        final_regime_detector=final_regime,
        ticker_to_id=dataset.ticker_to_id,
        shap_importance=shap_df,
    )

    if save_artifacts:
        save_ensemble_artifacts(result)

    return result


def save_ensemble_artifacts(result: MultiModelEnsembleResult) -> Path:
    """Persist trained models, OOF predictions, SHAP importance, and metadata to disk."""
    ensure_directories()
    out_dir = ARTIFACTS_DIR / result.horizon_name
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "horizon_name": result.horizon_name,
            "feature_cols": result.feature_cols,
            "model_names": result.model_names,
            "ensemble_weights": result.ensemble_weights,
            "tuned_params": result.tuned_params,
            "final_models": result.final_models,
            "final_regime_detector": result.final_regime_detector,
            "ticker_to_id": result.ticker_to_id,
        },
        out_dir / "ensemble_bundle.joblib",
        compress=3,
    )
    result.oof_predictions.to_parquet(out_dir / "oof_predictions.parquet", index=False)
    if not result.shap_importance.empty:
        result.shap_importance.to_parquet(out_dir / "shap_importance.parquet", index=False)
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "horizon_name": result.horizon_name,
                "model_names": result.model_names,
                "ensemble_weights": result.ensemble_weights,
                "tuned_params": result.tuned_params,
                "n_oof_rows": int(len(result.oof_predictions)),
                "n_features": int(len(result.feature_cols)),
            },
            f,
            indent=2,
        )
    return out_dir


def load_ensemble_artifacts(horizon_name: str) -> Dict[str, Any]:
    """Load persisted model bundle, OOF predictions, and SHAP table for a horizon."""
    out_dir = ARTIFACTS_DIR / horizon_name
    bundle_path = out_dir / "ensemble_bundle.joblib"
    if not bundle_path.exists():
        raise FileNotFoundError(f"No trained model bundle found for horizon '{horizon_name}' at {bundle_path}.")
    bundle = joblib.load(bundle_path)
    oof_path = out_dir / "oof_predictions.parquet"
    shap_path = out_dir / "shap_importance.parquet"
    bundle["oof_predictions"] = pd.read_parquet(oof_path) if oof_path.exists() else pd.DataFrame()
    bundle["shap_importance"] = pd.read_parquet(shap_path) if shap_path.exists() else pd.DataFrame()
    return bundle
