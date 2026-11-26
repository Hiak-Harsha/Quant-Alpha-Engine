"""Central configuration for horizons, diversified asset universes, feeds, and modeling parameters."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, List, Tuple

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent
DATA_CACHE_DIR = PROJECT_ROOT / "data_cache"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"


@dataclass(frozen=True)
class HorizonConfig:
    """Configuration profile for a specific trading style / prediction horizon."""

    name: str
    display_name: str
    interval: str
    period: str
    forecast_horizon: int
    purge_bars: int
    embargo_bars: int
    tx_cost_bps: float
    slippage_bps: float
    bars_per_year: int
    target_vol_annual: float
    min_train_rows: int
    n_splits: int
    description: str


HORIZON_CONFIGS: Dict[str, HorizonConfig] = {
    "intraday": HorizonConfig(
        name="intraday",
        display_name="Intraday (1H Bars -> 4H Horizon)",
        interval="1h",
        period="365d",
        forecast_horizon=4,
        purge_bars=8,
        embargo_bars=4,
        tx_cost_bps=2.0,
        slippage_bps=1.5,
        bars_per_year=252 * 7,
        target_vol_annual=0.12,
        min_train_rows=350,
        n_splits=5,
        description="Hourly momentum, intraday volatility compression, microstructure proxy & session spillover.",
    ),
    "swing": HorizonConfig(
        name="swing",
        display_name="Swing / Short-Term (1D Bars -> 5D Horizon)",
        interval="1d",
        period="10y",
        forecast_horizon=5,
        purge_bars=10,
        embargo_bars=5,
        tx_cost_bps=3.0,
        slippage_bps=2.0,
        bars_per_year=252,
        target_vol_annual=0.15,
        min_train_rows=500,
        n_splits=5,
        description="1-week swing forecasts combining multi-factor technicals, GARCH vol, cross-asset correlations, and macro regimes.",
    ),
    "longterm": HorizonConfig(
        name="longterm",
        display_name="Long-Term Position (1D Bars -> 21D Horizon)",
        interval="1d",
        period="15y",
        forecast_horizon=21,
        purge_bars=25,
        embargo_bars=10,
        tx_cost_bps=5.0,
        slippage_bps=3.0,
        bars_per_year=252,
        target_vol_annual=0.15,
        min_train_rows=600,
        n_splits=5,
        description="1-month trend, macro cycle, yield-curve, credit-spread, and regime-adaptive positioning.",
    ),
}


# Diversified multi-asset universes (India & Precious Metals default)
UNIVERSES: Dict[str, List[str]] = {
    "india_market_gold_silver": [
        # Indian Indices & Broad Market ETFs
        "^NSEI", "^NSEBANK", "^CNXIT", "NIFTYBEES.NS", "BANKBEES.NS", "JUNIORBEES.NS", "INDA",
        # Indian Gold & Silver ETFs + Global Bullion Proxies
        "GOLDBEES.NS", "HDFCGOLD.NS", "GOLDCASE.NS",
        "SILVERBEES.NS", "SILVER.NS", "SILVERIETF.NS", "HDFCSILVER.NS",
        "GLD", "SLV", "GC=F", "SI=F",
        # Top Indian Blue-Chip & High-Attention NSE Shares
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
        "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "LT.NS", "KOTAKBANK.NS",
        "AXISBANK.NS", "BAJFINANCE.NS", "JIOFIN.NS", "HINDUNILVR.NS", "TITAN.NS",
        "MARUTI.NS", "M&M.NS", "TMPV.NS", "SUNPHARMA.NS", "TATASTEEL.NS",
        "JSWSTEEL.NS", "HINDALCO.NS", "HAL.NS", "BEL.NS", "ADANIENT.NS",
        "ADANIPORTS.NS", "NTPC.NS", "ONGC.NS", "POWERGRID.NS", "TRENT.NS",
        "ETERNAL.NS", "HCLTECH.NS", "WIPRO.NS",
    ],
    "india_gold_silver": [
        "GOLDBEES.NS", "HDFCGOLD.NS", "GOLDCASE.NS",
        "SILVERBEES.NS", "SILVER.NS", "SILVERIETF.NS", "HDFCSILVER.NS",
        "GLD", "SLV", "GC=F", "SI=F", "USDINR=X",
    ],
    "india_equities": [
        "^NSEI", "^NSEBANK", "^CNXIT", "NIFTYBEES.NS", "BANKBEES.NS",
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
        "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "LT.NS", "BAJFINANCE.NS",
        "HAL.NS", "BEL.NS", "TATASTEEL.NS", "MARUTI.NS", "TMPV.NS", "SUNPHARMA.NS",
    ],
    "top_talked_leaders": [
        # US Mega-Cap Tech, AI & High-Attention Stocks
        "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA",
        "AMD", "AVGO", "PLTR", "NFLX", "ORCL", "MU", "INTC", "CRM",
        # US Financials, Healthcare, Energy, Industrials & Consumer
        "JPM", "GS", "V", "LLY", "UNH", "XOM", "CVX", "COST", "WMT", "BA",
        # Broad Indices, Bonds, Commodities & Crypto
        "SPY", "QQQ", "IWM", "TLT", "GLD", "SLV", "USO", "BTC-USD", "ETH-USD",
        # India / NSE Leaders & Proxies
        "^NSEI", "^NSEBANK", "INDA", "GOLDBEES.NS", "SILVERBEES.NS",
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "BHARTIARTL.NS",
    ],
    "stepping_stone": [
        "SPY", "QQQ", "IWM", "TLT", "GLD", "USO", "INDA", "BTC-USD",
    ],
    "us_equities": [
        "SPY", "QQQ", "IWM", "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA",
        "AMD", "AVGO", "PLTR", "JPM", "LLY", "XOM", "UNH",
    ],
    "us_sectors": [
        "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU",
    ],
    "diversified_all": [
        "SPY", "QQQ", "IWM", "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA",
        "AMD", "AVGO", "PLTR", "JPM", "XOM", "LLY",
        "XLK", "XLF", "XLE", "XLV",
        "TLT", "GLD", "SLV", "USO",
        "INDA", "^NSEI", "^NSEBANK", "GOLDBEES.NS", "SILVERBEES.NS",
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "BTC-USD", "ETH-USD",
    ],
}


# Global correlation, volatility, yield, credit, commodity, FX, and India drivers
GLOBAL_DRIVERS: List[str] = [
    "SPY",
    "QQQ",
    "IWM",
    "^VIX",
    "^VXN",
    "^TNX",
    "^IRX",
    "UUP",
    "USDINR=X",
    "HYG",
    "IEF",
    "TLT",
    "GLD",
    "SLV",
    "GC=F",
    "SI=F",
    "BZ=F",
    "USO",
    "XLK",
    "XLF",
    "XLE",
    "XLP",
    "XLY",
    "XLU",
    "INDA",
    "^NSEI",
    "^NSEBANK",
    "^INDIAVIX",
    "BTC-USD",
]


# Free public FRED macro series (CSV endpoint requires no API key)
FRED_SERIES: Dict[str, str] = {
    "T10Y2Y": "10Y-2Y Treasury Yield Spread",
    "BAMLH0A0HYM2": "ICE BofA US High Yield OAS",
    "DGS10": "10Y Treasury Constant Maturity Rate",
    "VIXCLS": "CBOE Volatility Index (FRED)",
    "DTWEXBGS": "Nominal Broad US Dollar Index",
}


# Public financial & macro RSS feeds (India + Gold/Silver + Global)
RSS_FEEDS: Dict[str, str] = {
    "economic_times_markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "economic_times_stocks": "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
    "google_news_india_markets": "https://news.google.com/rss/search?q=(Nifty+OR+Sensex+OR+NSE+OR+RBI+OR+%22gold+etf%22+OR+%22silver+etf%22+OR+goldbees+OR+silverbees+OR+%22Indian+stocks%22)+when:3d&hl=en-IN&gl=IN&ceid=IN:en",
    "reuters_business_google": "https://news.google.com/rss/search?q=stock+market+OR+gold+price+OR+silver+price+OR+federal+reserve+when:3d&hl=en-US&gl=US&ceid=US:en",
    "marketwatch_top": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "cnbc_markets": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664",
    "investing_news": "https://www.investing.com/rss/news.rss",
    "sec_edgar_8k": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&company=&dateb=&owner=include&start=0&count=40&output=atom",
}


@dataclass
class PortfolioConfig:
    """Portfolio construction constraints shared across backtester and live scan."""

    max_gross_exposure: float = 1.0
    max_net_exposure: float = 1.0
    max_single_asset_weight: float = 0.15
    max_cluster_weight: float = 0.25
    max_sector_weight: float = 0.35
    long_only_cash_india: bool = True
    cash_long_only_suffixes: Tuple[str, ...] = (".NS", ".BO")
    cash_long_only_tickers: Tuple[str, ...] = ("^NSEI", "^NSEBANK", "^CNXIT", "INDA")


@dataclass
class PassGateConfig:
    """Pre-registered statistical edge gates required before emitting live directional verdicts."""

    enable_pass_gate: bool = True
    min_alpha_ann_pct: float = 0.0
    min_hac_tstat: float = 2.0
    min_dsr_prob: float = 0.95
    min_asset_sharpe: float = 0.0
    min_asset_hit_edge_pct: float = 0.0


@dataclass
class VerdictConfig:
    """Config-driven signal, sizing, and sentiment thresholds (no magic numbers in code)."""

    strong_vol_norm_threshold: float = 0.30
    strong_agreement_pct: float = 66.0
    min_agreement_pct: float = 55.0
    signal_tanh_scale: float = 3.0
    conf_base: float = 0.65
    conf_prob_slope: float = 0.70
    crisis_damp_slope: float = 0.50
    crisis_damp_floor: float = 0.40
    min_vol_scale: float = 0.35
    atr_stop_mult: float = 1.5
    atr_tp_mult: float = 2.5
    neutral_tp_mult: float = 2.0
    news_min_articles: int = 2
    news_nudge_scale: float = 0.30
    news_blend_weight: float = 0.15


@dataclass
class EngineSettings:
    """Runtime settings for training, tuning, and inference."""

    cache_ttl_hours_daily: float = 6.0
    cache_ttl_hours_intraday: float = 0.5
    cache_ttl_hours_news: float = 0.5
    n_optuna_trials: int = 12
    random_state: int = 42
    max_leverage: float = 1.0
    min_signal_threshold: float = 0.05
    bootstrap_samples: int = 500
    bootstrap_block_len: int = 5
    enable_shap: bool = True
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    pass_gate: PassGateConfig = field(default_factory=PassGateConfig)
    verdicts: VerdictConfig = field(default_factory=VerdictConfig)
    ticker_timezone_hints: Dict[str, str] = field(
        default_factory=lambda: {
            ".NS": "Asia/Kolkata",
            ".BO": "Asia/Kolkata",
            "^NSEI": "Asia/Kolkata",
            "^NSEBANK": "Asia/Kolkata",
            "^INDIAVIX": "Asia/Kolkata",
            "-USD": "UTC",
        }
    )


def ensure_directories() -> None:
    """Create local cache, artifact, report, and log directories if absent."""
    for directory in (DATA_CACHE_DIR, ARTIFACTS_DIR, REPORTS_DIR, LOGS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure root console and rotating file logging for quant_alpha."""
    ensure_directories()
    root_logger = logging.getLogger("quant_alpha")
    if root_logger.handlers:
        root_logger.setLevel(level)
        return root_logger

    root_logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    root_logger.addHandler(stream_handler)

    file_handler = RotatingFileHandler(
        LOGS_DIR / "quant_alpha.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    root_logger.addHandler(file_handler)
    return root_logger


def is_asia_india_ticker(ticker: str) -> bool:
    """Return True if ticker trades in Indian/Asian session prior to US close."""
    upper = ticker.upper()
    return upper.endswith((".NS", ".BO")) or upper in {"^NSEI", "^NSEBANK", "^INDIAVIX", "^CNXIT"}


def is_crypto_ticker(ticker: str) -> bool:
    """Return True if ticker trades 24/7 (crypto)."""
    return ticker.upper().endswith(("-USD", "-USDT", "-BTC"))
