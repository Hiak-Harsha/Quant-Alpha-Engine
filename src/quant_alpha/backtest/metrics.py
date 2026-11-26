"""Financial performance, benchmark-relative HAC alpha, DSR, and gating metrics."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import kurtosis, norm, skew, spearmanr
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from quant_alpha.config import PassGateConfig

EULER_MASCHERONI = 0.5772156649015329


def compute_predictive_metrics(
    y_true_ret: np.ndarray,
    y_true_up: np.ndarray,
    pred_vol_norm: np.ndarray,
    prob_up: np.ndarray,
) -> Dict[str, float]:
    """Compute Spearman Rank IC, Hit Rate vs Always-Long base rate, ROC-AUC, PR-AUC, and Brier."""
    mask = np.isfinite(y_true_ret) & np.isfinite(pred_vol_norm) & np.isfinite(prob_up)
    if mask.sum() < 20:
        return {
            "spearman_ic": 0.0,
            "hit_rate_pct": 50.0,
            "always_long_hit_pct": 50.0,
            "hit_edge_pct": 0.0,
            "roc_auc": 0.5,
            "pr_auc": 0.5,
            "brier": 0.25,
        }

    yt_r = y_true_ret[mask]
    yt_u = y_true_up[mask].astype(int)
    pr_v = pred_vol_norm[mask]
    pb_u = np.clip(prob_up[mask], 1e-4, 1.0 - 1e-4)

    ic = 0.0
    if np.std(pr_v) > 1e-10:
        c, _ = spearmanr(yt_r, pr_v)
        ic = float(c) if np.isfinite(c) else 0.0

    hit_rate = float(((pr_v > 0.0) == (yt_r > 0.0)).mean() * 100.0)
    always_long_hit = float((yt_r > 0.0).mean() * 100.0)
    hit_edge = hit_rate - always_long_hit

    try:
        auc = float(roc_auc_score(yt_u, pb_u)) if len(np.unique(yt_u)) > 1 else 0.5
        pr_auc = float(average_precision_score(yt_u, pb_u)) if len(np.unique(yt_u)) > 1 else 0.5
    except Exception:
        auc, pr_auc = 0.5, 0.5

    brier = float(brier_score_loss(yt_u, pb_u))
    return {
        "spearman_ic": round(ic, 4),
        "hit_rate_pct": round(hit_rate, 2),
        "always_long_hit_pct": round(always_long_hit, 2),
        "hit_edge_pct": round(hit_edge, 2),
        "roc_auc": round(auc, 4),
        "pr_auc": round(pr_auc, 4),
        "brier": round(brier, 4),
    }


def _compute_hac_alpha_beta(
    strategy_ret: pd.Series,
    benchmark_ret: Optional[pd.Series],
    effective_bpy: float,
) -> Dict[str, float]:
    """Regress strategy returns on benchmark using Newey-West (HAC) covariance."""
    neutral = {
        "alpha_ann_pct": 0.0,
        "beta_to_bh": 1.0 if benchmark_ret is None else 0.0,
        "hac_t_stat": 0.0,
        "p_value": 1.0,
        "info_ratio": 0.0,
    }
    if benchmark_ret is None:
        return neutral

    aligned = pd.concat([strategy_ret.rename("y"), benchmark_ret.rename("x")], axis=1).dropna()
    if len(aligned) < 15 or aligned["x"].std(ddof=1) <= 1e-10 or aligned["y"].std(ddof=1) <= 1e-10:
        return neutral

    if np.allclose(aligned["y"].to_numpy(), aligned["x"].to_numpy(), atol=1e-12):
        return {
            "alpha_ann_pct": 0.0,
            "beta_to_bh": 1.0,
            "hac_t_stat": 0.0,
            "p_value": 1.0,
            "info_ratio": 0.0,
        }

    y = aligned["y"].to_numpy(dtype=float)
    x = aligned["x"].to_numpy(dtype=float)
    X = sm.add_constant(x, has_constant="add")
    maxlags = max(1, int(np.floor(4.0 * (len(y) / 100.0) ** (2.0 / 9.0))))
    fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})

    alpha_bar = float(fit.params[0])
    beta = float(fit.params[1])
    hac_t = float(fit.tvalues[0])
    two_p = float(fit.pvalues[0])
    one_p = (two_p / 2.0) if alpha_bar > 0.0 else (1.0 - two_p / 2.0)

    active = y - x
    active_std = float(np.std(active, ddof=1))
    info_ratio = (float(np.mean(active)) / active_std * np.sqrt(effective_bpy)) if active_std > 1e-10 else 0.0

    return {
        "alpha_ann_pct": round(alpha_bar * effective_bpy * 100.0, 2),
        "beta_to_bh": round(beta, 3),
        "hac_t_stat": round(hac_t, 3),
        "p_value": round(float(np.clip(one_p, 0.0, 1.0)), 4),
        "info_ratio": round(float(info_ratio), 3),
    }


def _compute_deflated_sharpe_ratio(
    r_arr: np.ndarray,
    n_trials: int = 1,
    trial_sharpes_per_bar: Optional[Sequence[float]] = None,
) -> float:
    """Compute Deflated Sharpe Ratio probability (Bailey & Lopez de Prado, 2014)."""
    t_obs = len(r_arr)
    if t_obs < 15:
        return 0.0
    std_r = float(np.std(r_arr, ddof=1))
    if std_r <= 1e-10:
        return 0.0

    sr_hat = float(np.mean(r_arr)) / std_r
    gamma3 = float(skew(r_arr, bias=False)) if t_obs > 3 else 0.0
    gamma4 = float(kurtosis(r_arr, fisher=False, bias=False)) if t_obs > 4 else 3.0
    if not np.isfinite(gamma3):
        gamma3 = 0.0
    if not np.isfinite(gamma4):
        gamma4 = 3.0

    if n_trials <= 1:
        sr_star = 0.0
    else:
        if trial_sharpes_per_bar is not None and len(trial_sharpes_per_bar) >= 2:
            sr_var = float(np.var(np.asarray(trial_sharpes_per_bar, dtype=float), ddof=1))
            sr_std = max(np.sqrt(max(sr_var, 0.0)), 1.0 / np.sqrt(max(t_obs, 2)))
        else:
            sr_std = 1.0 / np.sqrt(max(t_obs, 2))

        p1 = np.clip(1.0 - 1.0 / float(n_trials), 1e-6, 1.0 - 1e-6)
        p2 = np.clip(1.0 - 1.0 / (float(n_trials) * np.e), 1e-6, 1.0 - 1e-6)
        sr_star = sr_std * (
            (1.0 - EULER_MASCHERONI) * norm.ppf(p1) + EULER_MASCHERONI * norm.ppf(p2)
        )

    denom_inner = 1.0 - gamma3 * sr_hat + ((gamma4 - 1.0) / 4.0) * (sr_hat ** 2)
    denom = np.sqrt(max(denom_inner, 1e-8))
    z_val = ((sr_hat - sr_star) * np.sqrt(max(t_obs - 1, 1))) / denom
    return round(float(norm.cdf(z_val)), 4)


def _moving_block_bootstrap_sharpe_ci(
    r_arr: np.ndarray,
    effective_bpy: float,
    bootstrap_samples: int,
    block_len: int,
    random_state: int,
    fallback_sharpe: float,
) -> Tuple[float, float]:
    """Compute moving-block bootstrap 95% CI for annualized Sharpe ratio."""
    n = len(r_arr)
    if n < 20 or float(np.std(r_arr, ddof=1)) <= 1e-10 or bootstrap_samples <= 0:
        return fallback_sharpe, fallback_sharpe

    b = max(1, min(int(block_len), n // 2))
    n_blocks = int(np.ceil(n / b))
    max_start = max(1, n - b + 1)
    rng = np.random.default_rng(random_state)

    boot_sharpes = []
    for _ in range(bootstrap_samples):
        starts = rng.integers(0, max_start, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + b) for s in starts])[:n]
        sample = r_arr[idx]
        s_std = float(np.std(sample, ddof=1))
        if s_std > 1e-10:
            boot_sharpes.append(float(np.mean(sample)) / s_std * np.sqrt(effective_bpy))

    if not boot_sharpes:
        return fallback_sharpe, fallback_sharpe
    lo, hi = np.quantile(boot_sharpes, [0.025, 0.975])
    return float(lo), float(hi)


def evaluate_pass_gate_from_metrics(
    metrics: Dict[str, float],
    pass_gate: Optional[PassGateConfig] = None,
) -> Tuple[bool, str]:
    """Check whether strategy metrics satisfy pre-registered benchmark-relative pass rules."""
    cfg = pass_gate or PassGateConfig()
    if not cfg.enable_pass_gate:
        return True, "Pass gate disabled"

    alpha_ann = float(metrics.get("alpha_ann_pct", 0.0))
    hac_t = float(metrics.get("hac_t_stat", 0.0))
    dsr = float(metrics.get("dsr_prob", 0.0))

    reasons = []
    if alpha_ann <= cfg.min_alpha_ann_pct:
        reasons.append(f"alpha={alpha_ann:+.2f}%<={cfg.min_alpha_ann_pct:.2f}%")
    if hac_t < cfg.min_hac_tstat:
        reasons.append(f"HAC_t={hac_t:.2f}<{cfg.min_hac_tstat:.2f}")
    if dsr < cfg.min_dsr_prob:
        reasons.append(f"DSR={dsr:.3f}<{cfg.min_dsr_prob:.2f}")

    if reasons:
        return False, "Failed gate: " + ", ".join(reasons)
    return True, f"Passed (alpha={alpha_ann:+.2f}%, HAC_t={hac_t:.2f}, DSR={dsr:.3f})"


def evaluate_asset_pass_gate(
    asset_row: Optional[Dict[str, float]],
    horizon_passed: bool,
    horizon_reason: str,
    pass_gate: Optional[PassGateConfig] = None,
) -> Tuple[bool, str]:
    """Gate an asset verdict based on horizon edge first, then per-asset OOS rules."""
    cfg = pass_gate or PassGateConfig()
    if not cfg.enable_pass_gate:
        return True, "Pass gate disabled"
    if not horizon_passed:
        return False, f"Horizon gated ({horizon_reason})"
    if not asset_row:
        return False, "No per-asset OOS evaluation record"

    sharpe = float(asset_row.get("sharpe", 0.0))
    hit_edge = float(asset_row.get("hit_edge_pct", 0.0))
    if sharpe <= cfg.min_asset_sharpe or hit_edge < cfg.min_asset_hit_edge_pct:
        return (
            False,
            f"Asset failed OOS (sharpe={sharpe:.2f}, hit_edge={hit_edge:+.2f}%)",
        )
    return True, f"Asset passed (sharpe={sharpe:.2f}, hit_edge={hit_edge:+.2f}%)"


def compute_strategy_metrics(
    net_returns: pd.Series,
    turnover: pd.Series,
    bars_per_year: int = 252,
    bootstrap_samples: int = 500,
    bootstrap_block_len: int = 5,
    random_state: int = 42,
    benchmark_returns: Optional[pd.Series] = None,
    n_trials: int = 1,
    trial_sharpes_per_bar: Optional[Sequence[float]] = None,
    is_benchmark: bool = False,
    pass_gate: Optional[PassGateConfig] = None,
) -> Dict[str, float | str | bool]:
    """Compute risk-adjusted returns, HAC benchmark alpha/beta, DSR, block-bootstrap CI, and verdict."""
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
            "alpha_ann_pct": 0.0,
            "beta_to_bh": 0.0,
            "hac_t_stat": 0.0,
            "info_ratio": 0.0,
            "dsr_prob": 0.0,
            "p_value": 1.0,
            "gate_passed": False,
            "edge_verdict": "Benchmark (Not Rated)" if is_benchmark else "Insufficient Data",
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

    r_arr = r.to_numpy(dtype=float)
    ci_low, ci_high = _moving_block_bootstrap_sharpe_ci(
        r_arr=r_arr,
        effective_bpy=effective_bpy,
        bootstrap_samples=bootstrap_samples,
        block_len=bootstrap_block_len,
        random_state=random_state,
        fallback_sharpe=sharpe,
    )

    hac_stats = _compute_hac_alpha_beta(r, benchmark_returns, effective_bpy)
    dsr_prob = _compute_deflated_sharpe_ratio(
        r_arr=r_arr,
        n_trials=1 if is_benchmark else n_trials,
        trial_sharpes_per_bar=None if is_benchmark else trial_sharpes_per_bar,
    )

    out: Dict[str, float | str | bool] = {
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
        "alpha_ann_pct": hac_stats["alpha_ann_pct"],
        "beta_to_bh": hac_stats["beta_to_bh"],
        "hac_t_stat": hac_stats["hac_t_stat"],
        "info_ratio": hac_stats["info_ratio"],
        "dsr_prob": dsr_prob,
        "p_value": hac_stats["p_value"],
    }

    if is_benchmark:
        out["gate_passed"] = False
        out["edge_verdict"] = "Benchmark (Not Rated)"
        return out

    gate_passed, _ = evaluate_pass_gate_from_metrics(
        {k: float(v) for k, v in out.items() if isinstance(v, (int, float))},
        pass_gate=pass_gate,
    )
    out["gate_passed"] = gate_passed
    if gate_passed:
        out["edge_verdict"] = "Pass (Stat. Alpha)"
    elif hac_stats["alpha_ann_pct"] > 0.0 and hac_stats["hac_t_stat"] >= 1.0:
        out["edge_verdict"] = "Marginal (Unproven Alpha)"
    else:
        out["edge_verdict"] = "No Edge vs Benchmark"
    return out
