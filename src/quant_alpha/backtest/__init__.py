"""Realistic volatility-targeted backtesting and statistical validation metrics."""

from quant_alpha.backtest.engine import BacktestOutput, run_backtest_comparison
from quant_alpha.backtest.metrics import compute_predictive_metrics, compute_strategy_metrics

__all__ = [
    "BacktestOutput",
    "compute_predictive_metrics",
    "compute_strategy_metrics",
    "run_backtest_comparison",
]
