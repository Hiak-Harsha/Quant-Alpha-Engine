"""Out-of-sample volatility-targeted multi-asset portfolio backtesting engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from quant_alpha.backtest.metrics import (
    compute_predictive_metrics,
    compute_strategy_metrics,
    evaluate_asset_pass_gate,
    evaluate_pass_gate_from_metrics,
)
from quant_alpha.config import EngineSettings, HorizonConfig, PortfolioConfig
from quant_alpha.models.zoo import MODEL_DISPLAY_NAMES
from quant_alpha.portfolio.construct import compute_signal_raw_weights, construct_portfolio_weights_df


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
    portfolio_cfg: Optional[PortfolioConfig] = None,
) -> tuple[pd.Series, pd.Series]:
    """Simulate causal staggered-tranche volatility-targeted portfolio across OOF timestamps."""
    df = oof_df[
        ["Timestamp", "ticker", "bar_ret_1", "est_ann_vol", "prob_regime_crisis", pred_vol_col, prob_up_col]
    ].copy()
    df["bar_ret_1"] = df["bar_ret_1"].fillna(0.0).clip(-0.35, 0.35)

    df["raw_weight"] = compute_signal_raw_weights(
        pred_vol=df[pred_vol_col].to_numpy(),
        prob_up=df[prob_up_col].to_numpy(),
        est_ann_vol=df["est_ann_vol"].to_numpy(),
        crisis_prob=df["prob_regime_crisis"].to_numpy(),
        horizon_cfg=horizon_cfg,
        settings=settings,
    )

    # Staggered H-bar tranche holding avoids overlapping-horizon over-trading
    df = df.sort_values(["ticker", "Timestamp"])
    df["tranche_weight"] = (
        df.groupby("ticker")["raw_weight"]
        .transform(lambda s: s.rolling(horizon_cfg.forecast_horizon, min_periods=1).mean())
    )

    weights_wide = df.pivot_table(index="Timestamp", columns="ticker", values="tranche_weight", aggfunc="last").sort_index()
    # Hold previous position across closed market bars so no fake weekend/overnight turnover occurs
    weights_wide = weights_wide.ffill().fillna(0.0)
    rets_wide = df.pivot_table(index="Timestamp", columns="ticker", values="bar_ret_1", aggfunc="last").sort_index().fillna(0.0)

    pcfg = portfolio_cfg or settings.portfolio
    w_norm = construct_portfolio_weights_df(weights_wide, portfolio_cfg=pcfg)

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
    """Evaluate all individual models, the meta-ensemble, and Buy & Hold benchmarks on OOF predictions."""
    settings = settings or EngineSettings()
    candidates = [m for m in (list(model_names) + ["ensemble"]) if f"pred_vol_{m}" in oof_df.columns and f"prob_up_{m}" in oof_df.columns]
    n_trials = max(1, len(candidates))

    # Buy & Hold equal-weight benchmark across the exact same timestamps and universe
    rets_wide = (
        oof_df.pivot_table(index="Timestamp", columns="ticker", values="bar_ret_1", aggfunc="last")
        .fillna(0.0)
        .clip(-0.35, 0.35)
    )
    bh_ret = rets_wide.mean(axis=1)
    bh_turnover = pd.Series(0.0, index=bh_ret.index)

    # Simulate all candidate models first so DSR can use cross-sectional trial Sharpe dispersion
    sim_results: Dict[str, tuple[pd.Series, pd.Series, Dict[str, float]]] = {}
    trial_sharpes_per_bar: List[float] = []
    for m in candidates:
        pred_col = f"pred_vol_{m}"
        prob_col = f"prob_up_{m}"
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
            portfolio_cfg=settings.portfolio,
        )
        sim_results[m] = (net_ret, turnover, pred_metrics)
        r_clean = net_ret.dropna().to_numpy(dtype=float)
        if len(r_clean) >= 15 and float(np.std(r_clean, ddof=1)) > 1e-10:
            trial_sharpes_per_bar.append(float(np.mean(r_clean) / np.std(r_clean, ddof=1)))

    leaderboard_rows = []
    equity_dict: Dict[str, pd.Series] = {}
    dd_dict: Dict[str, pd.Series] = {}

    bh_always_long_hit = round(float((oof_df["target_fwd_ret"].dropna() > 0).mean() * 100.0), 2)
    bh_strat = compute_strategy_metrics(
        bh_ret,
        bh_turnover,
        bars_per_year=horizon_cfg.bars_per_year,
        bootstrap_samples=settings.bootstrap_samples,
        bootstrap_block_len=settings.bootstrap_block_len,
        random_state=settings.random_state,
        benchmark_returns=None,
        n_trials=1,
        is_benchmark=True,
        pass_gate=settings.pass_gate,
    )
    bh_eq = (1.0 + bh_ret).cumprod()
    equity_dict["Buy & Hold (Equal Weight)"] = bh_eq
    dd_dict["Buy & Hold (Equal Weight)"] = (bh_eq / bh_eq.cummax() - 1.0) * 100.0
    leaderboard_rows.append(
        {
            "strategy": "Buy & Hold (Equal Weight)",
            "spearman_ic": 0.0,
            "hit_rate_pct": bh_always_long_hit,
            "always_long_hit_pct": bh_always_long_hit,
            "hit_edge_pct": 0.0,
            "roc_auc": 0.5,
            "pr_auc": 0.5,
            "brier": 0.25,
            **bh_strat,
        }
    )

    for m in candidates:
        display_name = "Meta-Ensemble (Blended)" if m == "ensemble" else MODEL_DISPLAY_NAMES.get(m, m)
        net_ret, turnover, pred_metrics = sim_results[m]
        strat_metrics = compute_strategy_metrics(
            net_ret,
            turnover,
            bars_per_year=horizon_cfg.bars_per_year,
            bootstrap_samples=settings.bootstrap_samples,
            bootstrap_block_len=settings.bootstrap_block_len,
            random_state=settings.random_state,
            benchmark_returns=bh_ret,
            n_trials=n_trials,
            trial_sharpes_per_bar=trial_sharpes_per_bar,
            is_benchmark=False,
            pass_gate=settings.pass_gate,
        )
        eq = (1.0 + net_ret).cumprod()
        equity_dict[display_name] = eq
        dd_dict[display_name] = (eq / eq.cummax() - 1.0) * 100.0
        leaderboard_rows.append({"strategy": display_name, **pred_metrics, **strat_metrics})

    # Add Vol-Matched Buy & Hold so returns are directly comparable to Meta-Ensemble risk level
    ens_ret = sim_results["ensemble"][0] if "ensemble" in sim_results else next(iter(sim_results.values()))[0]
    bh_std = float(bh_ret.std(ddof=1)) if len(bh_ret) > 1 else 0.0
    ens_std = float(ens_ret.std(ddof=1)) if len(ens_ret) > 1 else 0.0
    if bh_std > 1e-10 and ens_std > 1e-10:
        vol_match_scale = ens_std / bh_std
        bh_vm_ret = (bh_ret * vol_match_scale).clip(-0.95, 0.95)
        bh_vm_strat = compute_strategy_metrics(
            bh_vm_ret,
            bh_turnover,
            bars_per_year=horizon_cfg.bars_per_year,
            bootstrap_samples=settings.bootstrap_samples,
            bootstrap_block_len=settings.bootstrap_block_len,
            random_state=settings.random_state,
            benchmark_returns=None,
            n_trials=1,
            is_benchmark=True,
            pass_gate=settings.pass_gate,
        )
        bh_vm_eq = (1.0 + bh_vm_ret).cumprod()
        equity_dict["Buy & Hold (Vol-Matched)"] = bh_vm_eq
        dd_dict["Buy & Hold (Vol-Matched)"] = (bh_vm_eq / bh_vm_eq.cummax() - 1.0) * 100.0
        leaderboard_rows.append(
            {
                "strategy": "Buy & Hold (Vol-Matched)",
                "spearman_ic": 0.0,
                "hit_rate_pct": bh_always_long_hit,
                "always_long_hit_pct": bh_always_long_hit,
                "hit_edge_pct": 0.0,
                "roc_auc": 0.5,
                "pr_auc": 0.5,
                "brier": 0.25,
                **bh_vm_strat,
            }
        )

    leaderboard = pd.DataFrame(leaderboard_rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
    equity_curves = pd.DataFrame(equity_dict).sort_index()
    drawdown_curves = pd.DataFrame(dd_dict).sort_index()

    # Check whether Meta-Ensemble passes the horizon gate
    ens_rows = leaderboard[leaderboard["strategy"] == "Meta-Ensemble (Blended)"]
    if not ens_rows.empty:
        horizon_passed, horizon_reason = evaluate_pass_gate_from_metrics(
            ens_rows.iloc[0].to_dict(), pass_gate=settings.pass_gate
        )
    else:
        horizon_passed, horizon_reason = False, "No ensemble metrics"

    # Per-asset out-of-sample standalone evaluation (same cash long-only constraint, unclipped by cross-asset caps)
    standalone_cfg = PortfolioConfig(
        max_gross_exposure=1.0,
        max_net_exposure=1.0,
        max_single_asset_weight=1.0,
        max_cluster_weight=1.0,
        max_sector_weight=1.0,
        long_only_cash_india=settings.portfolio.long_only_cash_india,
        cash_long_only_suffixes=settings.portfolio.cash_long_only_suffixes,
        cash_long_only_tickers=settings.portfolio.cash_long_only_tickers,
    )
    asset_rows = []
    for ticker, grp in oof_df.groupby("ticker"):
        pm = compute_predictive_metrics(
            grp["target_fwd_ret"].to_numpy(),
            grp["target_up"].to_numpy(),
            grp["pred_vol_ensemble"].to_numpy(),
            grp["prob_up_ensemble"].to_numpy(),
        )
        net_r, turn = _simulate_model_portfolio(
            grp,
            "pred_vol_ensemble",
            "prob_up_ensemble",
            horizon_cfg,
            settings,
            portfolio_cfg=standalone_cfg,
        )
        bh_asset_ret = grp.sort_values("Timestamp")["bar_ret_1"].fillna(0.0).clip(-0.35, 0.35)
        bh_asset_series = pd.Series(bh_asset_ret.to_numpy(), index=pd.to_datetime(grp.sort_values("Timestamp")["Timestamp"]))
        sm = compute_strategy_metrics(
            net_r,
            turn,
            bars_per_year=horizon_cfg.bars_per_year,
            bootstrap_samples=min(200, settings.bootstrap_samples),
            bootstrap_block_len=settings.bootstrap_block_len,
            random_state=settings.random_state,
            benchmark_returns=bh_asset_series,
            n_trials=1,
            is_benchmark=False,
            pass_gate=settings.pass_gate,
        )
        bh_asset_eq = float((1.0 + bh_asset_ret).prod() - 1.0) * 100.0
        row_summary = {
            "ticker": ticker,
            "oof_bars": int(len(grp)),
            "ensemble_ic": pm["spearman_ic"],
            "hit_rate_pct": pm["hit_rate_pct"],
            "always_long_hit_pct": pm["always_long_hit_pct"],
            "hit_edge_pct": pm["hit_edge_pct"],
            "roc_auc": pm["roc_auc"],
            "strategy_ret_pct": sm["total_return_pct"],
            "buy_hold_ret_pct": round(bh_asset_eq, 2),
            "alpha_ann_pct": sm["alpha_ann_pct"],
            "hac_t_stat": sm["hac_t_stat"],
            "sharpe": sm["sharpe"],
            "max_dd_pct": sm["max_drawdown_pct"],
            "p_value": sm["p_value"],
        }
        asset_gate_ok, asset_gate_reason = evaluate_asset_pass_gate(
            row_summary,
            horizon_passed=horizon_passed,
            horizon_reason=horizon_reason,
            pass_gate=settings.pass_gate,
        )
        row_summary["gate_passed"] = asset_gate_ok
        row_summary["gate_reason"] = asset_gate_reason
        asset_rows.append(row_summary)

    per_asset_summary = pd.DataFrame(asset_rows).sort_values("sharpe", ascending=False).reset_index(drop=True)

    return BacktestOutput(
        leaderboard=leaderboard,
        equity_curves=equity_curves,
        drawdown_curves=drawdown_curves,
        per_asset_summary=per_asset_summary,
    )
