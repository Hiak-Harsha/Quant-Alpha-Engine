# Quant Alpha Engine

A diversified, multi-horizon quantitative market analysis, backtesting, and objective prediction system.

## Architecture
- **Multi-Source Data Ingestion & Validation (`src/quant_alpha/data/`)**:
  - **Market & Volatility Feeds**: Equities, Sector ETFs, India proxies (`^NSEI`, `INDA`, `.NS`), Treasuries (`TLT`), Gold (`GLD`), Oil (`USO`), Crypto (`BTC-USD`, `ETH-USD`), Dollar (`UUP`), Credit (`HYG/IEF`), Volatility (`^VIX`, `^VXN`), and Yields (`^TNX`, `^IRX`).
  - **Macro Feeds**: Public FRED CSV series (`T10Y2Y`, `BAMLH0A0HYM2`, `DGS10`, `VIXCLS`, `DTWEXBGS`) with 1-observation publication lag safety.
  - **Live Financial News Feeds**: Yahoo Finance ticker news, GDELT 2.0 Doc API, financial/macro RSS feeds, SEC EDGAR 8-K Atom feed, plus optional `FINNHUB_API_KEY`, `NEWSAPI_KEY`, `ALPHAVANTAGE_API_KEY`.
- **Causal Feature, Volatility, Cross-Asset & Regime Engine (`src/quant_alpha/features/`)**:
  - Multi-scale returns, RSI, MACD, Bollinger squeeze, ATR, ADX, Amihud illiquidity, variance ratio, cyclical calendar encodings.
  - Realized, Parkinson, Garman-Klass, downside asymmetry, and strictly causal rolling **GARCH(1,1)** volatility forecasts.
  - Causal cross-asset rolling correlations, market beta, idiosyncratic residual returns, breadth/credit ratios, and timezone-aware lead/lag rules.
  - Fold-isolated **Gaussian HMM** regime detector (`Low-Vol Bull`, `Normal/Chop`, `High-Vol Crisis`).
  - Financial NLP sentiment scoring (VADER + financial domain lexicon + recency decay).
- **Purged Walk-Forward Multi-Model Zoo & Tuning (`src/quant_alpha/models/`)**:
  - Ridge/Logistic Baseline, LightGBM, XGBoost, ExtraTrees, Regularized Neural MLP, and Classical Quant Multi-Signal model.
  - **Optuna** tuning on strictly purged inner walk-forward folds and fold-by-fold causal ensemble weighting.
- **Realistic Backtester & Objective Predictor (`src/quant_alpha/backtest/`, `src/quant_alpha/predictor.py`)**:
  - Staggered-tranche volatility-targeted sizing, crisis regime damping, transaction costs & slippage, bootstrap 95% Sharpe confidence intervals, and one-sided statistical significance testing.
  - Supports **Intraday (`1h -> 4h`)**, **Swing (`1d -> 5d`)**, and **Long-Term (`1d -> 21d`)** horizons.

## Quickstart
```powershell
# 1. Validate all data, macro, and news feeds
.venv\Scripts\python.exe -m quant_alpha.cli validate-feeds --horizon swing --universe stepping_stone

# 2. Train, tune (Optuna), and walk-forward backtest
.venv\Scripts\python.exe -m quant_alpha.cli train-backtest --horizon swing --universe stepping_stone --trials 12

# 3. Generate live objective prediction across all horizons (Intraday, Swing, Long-Term)
.venv\Scripts\python.exe -m quant_alpha.cli predict --ticker SPY --all-horizons

# 4. Log & evaluate real-time paper trades (zero-leakage closed-bar ledger)
.venv\Scripts\python.exe -m quant_alpha.cli paper-log --universe stepping_stone --horizons swing,intraday --refresh
.venv\Scripts\python.exe -m quant_alpha.cli paper-eval
.venv\Scripts\python.exe -m quant_alpha.cli paper-report

# 5. Launch interactive Streamlit dashboard (includes Live Paper Tracker tab)
.venv\Scripts\streamlit.exe run app.py
```
