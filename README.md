# Quant Alpha Engine

A diversified, multi-horizon quantitative market analysis, walk-forward backtesting, portfolio construction, and objective prediction system.

## Architecture
- **Multi-Source Data Ingestion & Validation (`src/quant_alpha/data/`)**:
  - **Market & Volatility Feeds**: Equities, Sector ETFs, Indian NSE stocks & indices (`^NSEI`, `^NSEBANK`, `.NS`), Gold/Silver ETFs (`GOLDBEES.NS`, `SILVERBEES.NS`, `GLD`, `SLV`, `GC=F`, `SI=F`), Treasuries (`TLT`, `IEF`), Oil (`USO`, `BZ=F`), Crypto (`BTC-USD`, `ETH-USD`), FX (`USDINR=X`, `UUP`), Credit (`HYG`), and Volatility (`^VIX`, `^INDIAVIX`).
  - **Macro Feeds**: Public FRED CSV series (`T10Y2Y`, `BAMLH0A0HYM2`, `DGS10`, `VIXCLS`, `DTWEXBGS`) with 1-observation publication lag safety.
  - **Live Financial News Feeds**: Yahoo Finance ticker news, GDELT 2.0 Doc API, financial/macro RSS feeds, SEC EDGAR 8-K Atom feed, plus optional `FINNHUB_API_KEY`, `NEWSAPI_KEY`, `ALPHAVANTAGE_API_KEY`.
- **Causal Feature, Volatility, Cross-Asset & Regime Engine (`src/quant_alpha/features/`)**:
  - Strictly causal technicals (no `asset_id` leakage; intraday-only `hour_sin`/`hour_cos`), GARCH(1,1) + Parkinson/Garman-Klass volatility, cross-asset correlations/beta with timezone-aware lead/lag rules, fold-isolated Gaussian HMM regimes, VADER + domain sentiment, and `assert_causal_features` automated leak detection.
- **Purged Walk-Forward Multi-Model Zoo & Tuning (`src/quant_alpha/models/`)**:
  - Ridge/Logistic Baseline, LightGBM, XGBoost, ExtraTrees, Regularized Neural MLP, and Classical Quant Multi-Signal model with **Optuna** (`--trials 12` default) and fold-causal ensemble weights.
- **Benchmark-Relative Metrics, Shared Portfolio Construction & Edge Gating (`src/quant_alpha/backtest/`, `src/quant_alpha/portfolio/`, `src/quant_alpha/predictor.py`)**:
  - Evaluates models against Equal-Weight and **Vol-Matched Buy & Hold** via Newey-West HAC regression (`alpha_ann_pct`, `beta_to_bh`, `hac_t_stat`), moving-block bootstrap 95% Sharpe CIs, Deflated Sharpe Ratio (`dsr_prob`), and hit-rate edge vs. always-long base rate.
  - Shared `portfolio/construct.py` enforces **long-only cash Indian securities** (`>= 0.0` on `.NS`/`.BO`/India cash proxies), single-asset caps (`15%`), duplicate-exposure cluster caps (`25%` across Gold/Silver/Nifty/BankNifty proxies), sector caps (`35%`), and gross/net exposure caps (`100%`) identically in backtest and live scans.
  - Pre-registered `PassGateConfig` automatically gates horizons/assets without proven statistical alpha (`NO EDGE / GATED`, `0.0%` allocation) and defaults live inference to `closed_bars_only=True`.

## Phase 0 Backtest Summary (Out-of-Sample, `india_market_gold_silver`)
| Horizon | Strategy | Return % | Ann Vol % | Sharpe | Alpha %/yr | HAC t-stat | DSR | Hit % (vs Long %) | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Swing (`1d->5d`) | Buy & Hold (Equal Weight) | +155.46% | 12.46% | 1.402 | — | — | — | 54.61% | Benchmark (Not Rated) |
| Swing (`1d->5d`) | Buy & Hold (Vol-Matched) | +155.20% | 12.45% | 1.402 | — | — | — | 54.61% | Benchmark (Not Rated) |
| Swing (`1d->5d`) | Meta-Ensemble (Blended) | +145.46% | 12.45% | 1.346 | +3.18% | 0.881 | 0.962 | 52.65% (54.61%) | No Edge vs Benchmark |
| Intraday (`1h->4h`) | Buy & Hold (Equal Weight) | -7.36% | 14.22% | -0.664 | — | — | — | 47.96% | Benchmark (Not Rated) |
| Intraday (`1h->4h`) | Buy & Hold (Vol-Matched) | -5.72% | 11.18% | -0.664 | — | — | — | 47.96% | Benchmark (Not Rated) |
| Intraday (`1h->4h`) | Meta-Ensemble (Blended) | -1.68% | 11.18% | -0.152 | +2.47% | 0.212 | 0.065 | 50.16% (47.96%) | No Edge vs Benchmark |

> Neither `swing` nor `intraday` yet clears `PassGateConfig` (`alpha > 0`, `HAC t >= 2.0`, `DSR >= 0.95`), so live predictions are honestly gated (`NO EDGE / GATED`, `0.0%` allocation) until stronger alpha families are added. See [reports/phase0_results.md](reports/phase0_results.md).

## Quickstart (Cross-Platform)

### Install & Environment
```bash
# Linux / macOS
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock && pip install --no-deps -e .
cp .env.example .env
```
```powershell
# Windows PowerShell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.lock; pip install --no-deps -e .
Copy-Item .env.example .env
```

### Run CLI & Dashboard
```bash
# 1. Validate market, macro, and India/metals feeds
python -m quant_alpha.cli validate-feeds --horizon swing --universe india_market_gold_silver

# 2. Train, tune (Optuna default --trials 12), and walk-forward backtest
python -m quant_alpha.cli train-backtest --horizon swing --universe india_market_gold_silver --trials 12

# 3. Portfolio-constrained live scan (defaults closed_bars_only=True, PassGate applied)
python -m quant_alpha.cli scan --horizon swing --universe india_market_gold_silver

# 4. Single-asset prediction & closed-bar paper-trading tracker
python -m quant_alpha.cli predict --ticker GOLDBEES.NS --all-horizons
python -m quant_alpha.cli paper-log --universe india_market_gold_silver --horizons swing,intraday --refresh
python -m quant_alpha.cli paper-eval

# 5. Build frontend (first time) and launch FastAPI + React Quant Terminal Web App
cd web && npm install && npm run build && cd ..
python -m quant_alpha.cli serve --host 0.0.0.0 --port 8501
```

### Docker & Verification Checks
```bash
docker build -t quant-alpha-engine .
docker run -p 8501:8501 --env-file .env quant-alpha-engine

# Linting, static type check, and unit/leakage test suite
ruff check src tests app.py
mypy src/quant_alpha
pytest tests -v
```
