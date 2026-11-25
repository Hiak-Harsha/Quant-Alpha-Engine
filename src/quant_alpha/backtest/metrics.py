"""Financial performance, predictive quality, and bootstrap statistical significance metrics."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_1samp
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def compute_predictive_metrics(
    y_true_ret: np.ndarray,
    y_true_up: np.ndarray,
    pred_vol_norm: np.ndarray,
    prob_up: np.ndarray,
) -> Dict[str, float]:
    """Compute Spearman Rank IC, Hit Rate, ROC-AUC, PR-AUC, and Brier Score."""
    mask = np.isfinite(y_true_ret) & np.isfinite(pred_vol_norm) & np.isfinite(prob_up)
    if mask.sum() < 20:
        return {"spearman_ic": 0.0, "hit_rate_pct": 50.0, "roc_auc": 0.5, "pr_auc": 0.5, "brier": 0.25}

    yt_r = y_true_ret[mask]
    yt_u = y_true_up[mask].astype(int)
    pr_v = pred_vol_norm[mask]
    pb_u = np.clip(prob_up[mask], 1e-4, 1.0 - 1e-4)

    ic = 0.0
    if np.std(pr_v) > 1e-10:
        c, _ = spearmanr(yt_r, pr_v)
        ic = float(c) if np.isfinite(c) else 0.0

    hit_rate = float(((pr_v > 0.0) == (yt_r > 0.0)).mean() * 100.0)
    try:
        auc = float(roc_auc_score(yt_u, pb_u)) if len(np.unique(yt_u)) > 1 else 0.5
        pr_auc = float(average_precision_score(yt_u, pb_u)) if len(np.unique(yt_u)) > 1 else 0.5
    except Exception:
        auc, pr_auc = 0.5, 0.5

    brier = float(brier_score_loss(yt_u, pb_u))
    return {
        "spearman_ic": round(ic, 4),
        "hit_rate_pct": round(hit_rate, 2),
        "roc_auc": round(auc, 4),
        "pr_auc": round(pr_auc, 4),
        "brier": round(brier, 4),
    }


def compute_strategy_metrics(
    net_returns: pd.Series,
    turnover: pd.Series,
    bars_per_year: int = 252,
    bootstrap_samples: int = 500,
    random_state: int = 42,
) -> Dict[str, float]:
    """Compute risk-adjusted returns, drawdowns, and bootstrap 95% CI significance."""
    r = net_returns.dropna().astype(float)
    if len(r) < 10:
        return {
            "total_return_pct": 0.0,
            "cagr_pct": 0.0,
            "ann_vol_pct": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "calmar": 0.0,
            "max_drawdown_pct": 0.0,
            "win_bar_pct": 0.0,
            "profit_factor": 0.0,
            "ann_turnover": 0.0,
            "sharpe_ci_low": 0.0,
            "sharpe_ci_high": 0.0,
            "p_value": 1.0,
            "edge_verdict": "Insufficient Data",
        }

    equity = (1.0 + r).cumprod()
    total_ret = float(equity.iloc[-1] - 1.0)
    if isinstance(r.index, pd.DatetimeIndex) and len(r) >= 2:
        elapsed_years = (r.index.max() - r.index.min()).total_seconds() / (365.25 * 86400.0)
        if elapsed_years > 0.05:
            years = float(elapsed_years)
            effective_bpy = float(len(r)) / years
        else:
            years = max(len(r) / float(bars_per_year), 1.0 / float(bars_per_year))
            effective_bpy = float(bars_per_year)
    else:
        years = max(len(r) / float(bars_per_year), 1.0 / float(bars_per_year))
        effective_bpy = float(bars_per_year)

    cagr = float((max(equity.iloc[-1], 1e-6) ** (1.0 / years)) - 1.0)

    mean_r = float(r.mean())
    std_r = float(r.std(ddof=1)) if len(r) > 1 else 0.0
    ann_vol = std_r * np.sqrt(effective_bpy)
    sharpe = (mean_r / std_r * np.sqrt(effective_bpy)) if std_r > 1e-10 else 0.0

    downside = r[r < 0.0]
    down_std = float(np.sqrt((downside ** 2).mean())) if len(downside) > 0 else 0.0
    sortino = (mean_r / down_std * np.sqrt(effective_bpy)) if down_std > 1e-10 else 0.0

    running_max = equity.cummax()
    dd = (equity / running_max) - 1.0
    max_dd = float(dd.min())
    calmar = (cagr / abs(max_dd)) if abs(max_dd) > 1e-6 else 0.0

    gains = float(r[r > 0.0].sum())
    losses = float(abs(r[r < 0.0].sum()))
    profit_factor = gains / losses if losses > 1e-10 else (2.0 if gains > 0 else 0.0)
    win_bar_pct = float((r > 0.0).mean() * 100.0)
    ann_turnover = float(turnover.mean() * effective_bpy) if len(turnover) else 0.0

    # Bootstrap 95% CI for Sharpe ratio
    rng = np.random.default_rng(random_state)
    r_arr = r.to_numpy()
    boot_sharpes = []
    if len(r_arr) >= 20 and std_r > 1e-10:
        for _ in range(bootstrap_samples):
            sample = rng.choice(r_arr, size=len(r_arr), replace=True)
            s_std = np.std(sample, ddof=1)
            if s_std > 1e-10:
                boot_sharpes.append(np.mean(sample) / s_std * np.sqrt(effective_bpy))
    if boot_sharpes:
        ci_low, ci_high = np.quantile(boot_sharpes, [0.025, 0.975])
    else:
        ci_low, ci_high = sharpe, sharpe

    # One-sided t-test: H0: mean_r <= 0 vs H1: mean_r > 0
    if std_r > 1e-10:
        _, two_p = ttest_1samp(r_arr, 0.0)
        p_val = float(two_p / 2.0) if mean_r > 0 else float(1.0 - two_p / 2.0)
    else:
        p_val = 1.0

    if p_val < 0.05 and sharpe > 0.5:
        verdict = "Statistically Significant Edge"
    elif p_val < 0.15 and sharpe > 0.2:
        verdict = "Marginal / Moderate Edge"
    else:
        verdict = "No Clear Edge After Costs"

    return {
        "total_return_pct": round(total_ret * 100.0, 2),
        "cagr_pct": round(cagr * 100.0, 2),
        "ann_vol_pct": round(ann_vol * 100.0, 2),
        "sharpe": round(float(sharpe), 3),
        "sortino": round(float(sortino), 3),
        "calmar": round(float(calmar), 3),
        "max_drawdown_pct": round(max_dd * 100.0, 2),
        "win_bar_pct": round(win_bar_pct, 2),
        "profit_factor": round(float(profit_factor), 3),
        "ann_turnover": round(ann_turnover, 2),
        "sharpe_ci_low": round(float(ci_low), 3),
        "sharpe_ci_high": round(float(ci_high), 3),
        "p_value": round(float(p_val), 4),
        "edge_verdict": verdict,
    }
