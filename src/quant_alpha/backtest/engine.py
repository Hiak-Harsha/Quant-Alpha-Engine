"""Out-of-sample volatility-targeted multi-asset portfolio backtesting engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from quant_alpha.backtest.metrics import compute_predictive_metrics, compute_strategy_metrics
from quant_alpha.config import EngineSettings, HorizonConfig
from quant_alpha.models.zoo import MODEL_DISPLAY_NAMES


@dataclass
class BacktestOutput:
    """Complete out-of-sample leaderboard, equity curves, and per-asset summary."""

    leaderboard: pd.DataFrame
    equity_curves: pd.DataFrame
    drawdown_curves: pd.DataFrame
    per_asset_summary: pd.DataFrame


def _simulate_model_portfolio(
    oof_df: pd.DataFrame,
    pred_vol_col: str,
    prob_up_col: str,
    horizon_cfg: HorizonConfig,
    settings: EngineSettings,
) -> tuple[pd.Series, pd.Series]:
    """Simulate causal staggered-tranche volatility-targeted portfolio across OOF timestamps."""
    df = oof_df[
        ["Timestamp", "ticker", "bar_ret_1", "est_ann_vol", "prob_regime_crisis", pred_vol_col, prob_up_col]
    ].copy()
    df["bar_ret_1"] = df["bar_ret_1"].fillna(0.0).clip(-0.35, 0.35)

    # Signal conviction with tanh scaling (maps ~0.25-0.40 sigma forecast to strong conviction) & deadband filter
    pred_v = df[pred_vol_col].fillna(0.0).to_numpy()
    prob_u = df[prob_up_col].fillna(0.5).to_numpy()
    conf = 0.65 + 0.70 * np.abs(prob_u - 0.5)
    raw_sig = np.where(
        np.abs(pred_v) < settings.min_signal_threshold,
        0.0,
        np.tanh(pred_v * 3.0) * conf,
    )

    # Inverse-volatility sizing + crisis regime damping
    est_vol = df["est_ann_vol"].fillna(0.18).clip(lower=0.05).to_numpy()
    vol_scale = np.clip(horizon_cfg.target_vol_annual / est_vol, 0.35, settings.max_leverage)
    crisis_prob = df["prob_regime_crisis"].fillna(0.2).to_numpy()
    regime_scale = np.clip(1.0 - 0.50 * crisis_prob, 0.40, 1.0)

    df["raw_weight"] = np.clip(raw_sig * vol_scale * regime_scale, -settings.max_leverage, settings.max_leverage)

    # Staggered H-bar tranche holding avoids overlapping-horizon over-trading
    df = df.sort_values(["ticker", "Timestamp"])
    df["tranche_weight"] = (
        df.groupby("ticker")["raw_weight"]
        .transform(lambda s: s.rolling(horizon_cfg.forecast_horizon, min_periods=1).mean())
    )

    weights_wide = df.pivot_table(index="Timestamp", columns="ticker", values="tranche_weight", aggfunc="last").sort_index()
    # Hold previous position across closed market bars (e.g., weekends/overnight vs 24/7 crypto) so no fake turnover occurs
    weights_wide = weights_wide.ffill().fillna(0.0)
    rets_wide = df.pivot_table(index="Timestamp", columns="ticker", values="bar_ret_1", aggfunc="last").sort_index().fillna(0.0)

    # Equal capital budget per asset, capped by total max leverage
    n_assets = max(1, weights_wide.shape[1])
    w_norm = weights_wide / float(n_assets)
    gross_lev = w_norm.abs().sum(axis=1)
    cap_factor = np.minimum(1.0, settings.max_leverage / gross_lev.replace(0.0, np.nan)).fillna(1.0)
    w_norm = w_norm.mul(cap_factor, axis=0)

    # Shift weights by 1 bar so weight decided at close t-1 earns return at t
    w_prev = w_norm.shift(1).fillna(0.0)
    gross_ret = (w_prev * rets_wide).sum(axis=1)
    turnover = (w_norm - w_prev).abs().sum(axis=1)
    cost_rate = (horizon_cfg.tx_cost_bps + horizon_cfg.slippage_bps) / 10000.0
    net_ret = gross_ret - turnover * cost_rate
    return net_ret, turnover


def run_backtest_comparison(
    oof_df: pd.DataFrame,
    model_names: List[str],
    horizon_cfg: HorizonConfig,
    settings: EngineSettings | None = None,
) -> BacktestOutput:
    """Evaluate all individual models, the meta-ensemble, and Buy & Hold on OOF predictions."""
    settings = settings or EngineSettings()
    candidates = list(model_names) + ["ensemble"]

    leaderboard_rows = []
    equity_dict: Dict[str, pd.Series] = {}
    dd_dict: Dict[str, pd.Series] = {}

    # Buy & Hold equal-weight benchmark across the exact same timestamps and universe
    rets_wide = (
        oof_df.pivot_table(index="Timestamp", columns="ticker", values="bar_ret_1", aggfunc="last")
        .fillna(0.0)
        .clip(-0.35, 0.35)
    )
    bh_ret = rets_wide.mean(axis=1)
    bh_turnover = pd.Series(0.0, index=bh_ret.index)
    bh_strat = compute_strategy_metrics(
        bh_ret,
        bh_turnover,
        bars_per_year=horizon_cfg.bars_per_year,
        bootstrap_samples=settings.bootstrap_samples,
        random_state=settings.random_state,
    )
    bh_eq = (1.0 + bh_ret).cumprod()
    equity_dict["Buy & Hold (Equal Weight)"] = bh_eq
    dd_dict["Buy & Hold (Equal Weight)"] = (bh_eq / bh_eq.cummax() - 1.0) * 100.0
    leaderboard_rows.append(
        {
            "strategy": "Buy & Hold (Equal Weight)",
            "spearman_ic": 0.0,
            "hit_rate_pct": round(float((bh_ret > 0).mean() * 100.0), 2),
            "roc_auc": 0.5,
            "pr_auc": 0.5,
            "brier": 0.25,
            **bh_strat,
        }
    )

    for m in candidates:
        pred_col = f"pred_vol_{m}"
        prob_col = f"prob_up_{m}"
        if pred_col not in oof_df.columns or prob_col not in oof_df.columns:
            continue

        display_name = "Meta-Ensemble (Blended)" if m == "ensemble" else MODEL_DISPLAY_NAMES.get(m, m)
        pred_metrics = compute_predictive_metrics(
            y_true_ret=oof_df["target_fwd_ret"].to_numpy(),
            y_true_up=oof_df["target_up"].to_numpy(),
            pred_vol_norm=oof_df[pred_col].to_numpy(),
            prob_up=oof_df[prob_col].to_numpy(),
        )
        net_ret, turnover = _simulate_model_portfolio(
            oof_df=oof_df,
            pred_vol_col=pred_col,
            prob_up_col=prob_col,
            horizon_cfg=horizon_cfg,
            settings=settings,
        )
        strat_metrics = compute_strategy_metrics(
            net_ret,
            turnover,
            bars_per_year=horizon_cfg.bars_per_year,
            bootstrap_samples=settings.bootstrap_samples,
            random_state=settings.random_state,
        )
        eq = (1.0 + net_ret).cumprod()
        equity_dict[display_name] = eq
        dd_dict[display_name] = (eq / eq.cummax() - 1.0) * 100.0

        leaderboard_rows.append({"strategy": display_name, **pred_metrics, **strat_metrics})

    leaderboard = pd.DataFrame(leaderboard_rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
    equity_curves = pd.DataFrame(equity_dict).sort_index()
    drawdown_curves = pd.DataFrame(dd_dict).sort_index()

    # Per-asset out-of-sample predictive & strategy summary for the Ensemble
    asset_rows = []
    for ticker, grp in oof_df.groupby("ticker"):
        pm = compute_predictive_metrics(
            grp["target_fwd_ret"].to_numpy(),
            grp["target_up"].to_numpy(),
            grp["pred_vol_ensemble"].to_numpy(),
            grp["prob_up_ensemble"].to_numpy(),
        )
        net_r, turn = _simulate_model_portfolio(
            grp, "pred_vol_ensemble", "prob_up_ensemble", horizon_cfg, settings
        )
        sm = compute_strategy_metrics(
            net_r,
            turn,
            bars_per_year=horizon_cfg.bars_per_year,
            bootstrap_samples=200,
            random_state=settings.random_state,
        )
        bh_asset_ret = grp.sort_values("Timestamp")["bar_ret_1"].fillna(0.0).clip(-0.35, 0.35)
        bh_asset_eq = float((1.0 + bh_asset_ret).prod() - 1.0) * 100.0
        asset_rows.append(
            {
                "ticker": ticker,
                "oof_bars": int(len(grp)),
                "ensemble_ic": pm["spearman_ic"],
                "hit_rate_pct": pm["hit_rate_pct"],
                "roc_auc": pm["roc_auc"],
                "strategy_ret_pct": sm["total_return_pct"],
                "buy_hold_ret_pct": round(bh_asset_eq, 2),
                "sharpe": sm["sharpe"],
                "max_dd_pct": sm["max_drawdown_pct"],
                "p_value": sm["p_value"],
            }
        )

    per_asset_summary = pd.DataFrame(asset_rows).sort_values("sharpe", ascending=False).reset_index(drop=True)

    return BacktestOutput(
        leaderboard=leaderboard,
        equity_curves=equity_curves,
        drawdown_curves=drawdown_curves,
        per_asset_summary=per_asset_summary,
    )
