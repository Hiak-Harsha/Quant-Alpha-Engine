# Phase 0 Verification & Out-of-Sample Results

Generated after completing all Phase 0 research-integrity, portfolio-construction, benchmark-relative attribution, and verdict-gating fixes.

## 1. Phase 0 Fixes Implemented
- **Feature & Train/Serve Hygiene**:
  - Removed arbitrary integer `asset_id` from model `feature_cols` (`src/quant_alpha/features/pipeline.py`).
  - Removed constant daily `hour_sin` / `hour_cos` features (`src/quant_alpha/features/technical.py`).
  - Defaulted `closed_bars_only=True` in `predict_asset`, `scan_market_predictions`, CLI, and Streamlit.
- **Benchmark-Relative Statistical Testing (`src/quant_alpha/backtest/metrics.py`)**:
  - Added Newey-West (HAC) regression of strategy net returns on Equal-Weight Buy & Hold (`alpha_ann_pct`, `beta_to_bh`, `hac_t_stat`, one-sided HAC `p_value`).
  - Added Moving-Block Bootstrap 95% Sharpe CIs, Deflated Sharpe Ratio (`dsr_prob` across all `n_trials` compared candidates), and `hit_edge_pct` vs. `always_long_hit_pct`.
  - Added explicit `Buy & Hold (Vol-Matched)` benchmark scaled to Meta-Ensemble realized volatility and labeled Buy & Hold rows `"Benchmark (Not Rated)"`.
- **Shared Portfolio Construction & Cash Constraints (`src/quant_alpha/portfolio/construct.py`)**:
  - Identical constraint projection used in both walk-forward backtests (`backtest/engine.py`) and live cross-sectional scans (`predictor.py`).
  - Enforces strict long-only (`>= 0.0`) weights on Indian cash instruments (`.NS`, `.BO`, `^NSEI`, `^NSEBANK`, `^CNXIT`, `INDA`), `15%` single-asset cap, `25%` exposure-cluster cap (Gold, Silver, Nifty, BankNifty), `35%` sector cap, and `100%` portfolio gross/net caps.
- **Pre-Registered Verdict Gating (`PassGateConfig` in `config.py`)**:
  - Gates any horizon or asset failing `alpha_ann_pct > 0`, `HAC t-stat >= 2.0`, and `DSR >= 0.95` to `verdict = "NO EDGE / GATED"` and `0.0%` allocation.

## 2. Out-of-Sample Walk-Forward Results (`india_market_gold_silver`, `--trials 6`)

### Swing Horizon (`1d -> 5d`, 109,408 panel rows, 169 causal features, 51 target assets)
| Strategy | Return % | Ann Vol % | Sharpe | Alpha %/yr | Beta | HAC t-stat | DSR | Hit % (Base %) | MaxDD % | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Buy & Hold (Equal Weight) | +155.46% | 12.46% | 1.402 | 0.00% | 1.000 | 0.000 | 0.9994 | 54.61% (54.61%) | -15.66% | Benchmark (Not Rated) |
| Buy & Hold (Vol-Matched) | +155.20% | 12.45% | 1.402 | 0.00% | 1.000 | 0.000 | 0.9994 | 54.61% (54.61%) | -15.65% | Benchmark (Not Rated) |
| Meta-Ensemble (Blended) | +145.46% | 12.45% | 1.346 | +3.18% | 0.777 | 0.881 | 0.9619 | 52.65% (54.61%) | -17.78% | No Edge vs Benchmark |
| Classical Multi-Signal Quant | +117.31% | 13.34% | 1.102 | -1.52% | 0.929 | -0.552 | 0.8821 | 51.65% (54.61%) | -22.70% | No Edge vs Benchmark |
| Ridge / Logistic Baseline | +101.83% | 12.63% | 1.052 | -1.36% | 0.839 | -0.399 | 0.8635 | 53.06% (54.61%) | -16.10% | No Edge vs Benchmark |
| Regularized Neural MLP | +99.94% | 12.94% | 1.018 | -1.24% | 0.825 | -0.339 | 0.8439 | 51.68% (54.61%) | -18.86% | No Edge vs Benchmark |
| LightGBM (Leaf-wise GBDT) | +61.87% | 11.75% | 0.788 | -3.50% | 0.730 | -1.013 | 0.6829 | 50.84% (54.61%) | -17.30% | No Edge vs Benchmark |
| ExtraTrees Ensemble | +57.60% | 11.95% | 0.737 | -5.10% | 0.796 | -1.594 | 0.6390 | 51.66% (54.61%) | -15.89% | No Edge vs Benchmark |
| XGBoost (Depth-wise GBDT) | +48.47% | 11.82% | 0.654 | -5.39% | 0.751 | -1.590 | 0.5645 | 50.78% (54.61%) | -19.21% | No Edge vs Benchmark |

### Intraday Horizon (`1h -> 4h`)
| Strategy | Return % | Ann Vol % | Sharpe | Alpha %/yr | Beta | HAC t-stat | DSR | Hit % (Base %) | MaxDD % | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Classical Multi-Signal Quant | -1.41% | 11.35% | -0.114 | +2.28% | 0.379 | 0.182 | 0.0688 | 50.40% (47.96%) | -13.63% | No Edge vs Benchmark |
| Meta-Ensemble (Blended) | -1.68% | 11.18% | -0.152 | +2.47% | 0.441 | 0.212 | 0.0647 | 50.16% (47.96%) | -15.86% | No Edge vs Benchmark |
| Ridge / Logistic Baseline | -4.94% | 10.76% | -0.589 | -3.09% | 0.344 | -0.255 | 0.0292 | 49.99% (47.96%) | -11.89% | No Edge vs Benchmark |
| Buy & Hold (Equal Weight) | -7.36% | 14.22% | -0.664 | 0.00% | 1.000 | 0.000 | 0.2845 | 47.96% (47.96%) | -12.45% | Benchmark (Not Rated) |
| Buy & Hold (Vol-Matched) | -5.72% | 11.18% | -0.664 | 0.00% | 1.000 | 0.000 | 0.2845 | 47.96% (47.96%) | -9.90% | Benchmark (Not Rated) |
| XGBoost (Depth-wise GBDT) | -7.43% | 12.40% | -0.789 | -3.74% | 0.640 | -0.342 | 0.0188 | 48.90% (47.96%) | -13.92% | No Edge vs Benchmark |
| Regularized Neural MLP | -7.82% | 11.71% | -0.892 | -5.22% | 0.554 | -0.501 | 0.0157 | 50.53% (47.96%) | -17.46% | No Edge vs Benchmark |
| LightGBM (Leaf-wise GBDT) | -12.66% | 13.07% | -1.350 | -11.05% | 0.699 | -1.020 | 0.0049 | 50.11% (47.96%) | -13.48% | No Edge vs Benchmark |
| ExtraTrees Ensemble | -14.16% | 10.16% | -2.002 | -16.70% | 0.386 | -1.598 | 0.0007 | 49.16% (47.96%) | -15.09% | No Edge vs Benchmark |

## 3. Conclusion
- Portfolio construction scales the strategy accurately to matched market volatility (`12.45%` vs `12.46%` on swing) while keeping live scans strictly within bounds (`0.0%` gross when gated).
- However, statistical evidence shows **neither horizon currently achieves statistically significant net alpha (`HAC t >= 2.0`, `DSR >= 0.95`)** over Buy & Hold after costs; therefore PassGate correctly blocks live directional verdicts (`NO EDGE / GATED`) until higher-alpha features/signals (next phase) pass the pre-registered bar.
