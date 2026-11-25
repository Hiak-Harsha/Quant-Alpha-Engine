"""Central configuration for horizons, diversified asset universes, feeds, and modeling parameters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent
DATA_CACHE_DIR = PROJECT_ROOT / "data_cache"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"


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


# Diversified multi-asset universes
UNIVERSES: Dict[str, List[str]] = {
    "top_talked_leaders": [
        # US Mega-Cap Tech, AI & High-Attention Stocks
        "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA",
        "AMD", "AVGO", "PLTR", "NFLX", "ORCL", "MU", "INTC", "CRM",
        # US Financials, Healthcare, Energy, Industrials & Consumer
        "JPM", "GS", "V", "LLY", "UNH", "XOM", "CVX", "COST", "WMT", "BA",
        # Broad Indices, Bonds, Commodities & Crypto
        "SPY", "QQQ", "IWM", "TLT", "GLD", "SLV", "USO", "BTC-USD", "ETH-USD",
        # India / NSE Leaders & Proxies
        "^NSEI", "^NSEBANK", "INDA", "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "BHARTIARTL.NS",
    ],
    "stepping_stone": [
        "SPY",       # US Large Cap (S&P 500)
        "QQQ",       # US Tech / Growth (NASDAQ-100)
        "IWM",       # US Small Cap (Russell 2000)
        "TLT",       # US 20Y+ Treasury Bonds
        "GLD",       # Gold
        "USO",       # Crude Oil
        "INDA",      # India Equities ETF (US-listed macro proxy)
        "BTC-USD",   # Bitcoin
    ],
    "us_equities": [
        "SPY", "QQQ", "IWM", "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA",
        "AMD", "AVGO", "PLTR", "JPM", "LLY", "XOM", "UNH",
    ],
    "us_sectors": [
        "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU",
    ],
    "india_equities": [
        "^NSEI", "^NSEBANK", "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS",
        "ICICIBANK.NS", "BHARTIARTL.NS", "SBIN.NS", "LT.NS",
    ],
    "diversified_all": [
        "SPY", "QQQ", "IWM", "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA",
        "AMD", "AVGO", "PLTR", "JPM", "XOM", "LLY",
        "XLK", "XLF", "XLE", "XLV",
        "TLT", "GLD", "SLV", "USO",
        "INDA", "^NSEI", "^NSEBANK", "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "BTC-USD", "ETH-USD",
    ],
}


# Global correlation, volatility, yield, credit, and currency drivers always fetched for context
GLOBAL_DRIVERS: List[str] = [
    "SPY",
    "QQQ",
    "IWM",
    "^VIX",
    "^VXN",
    "^TNX",
    "^IRX",
    "UUP",
    "HYG",
    "IEF",
    "TLT",
    "GLD",
    "USO",
    "XLK",
    "XLF",
    "XLE",
    "XLP",
    "XLY",
    "XLU",
    "INDA",
    "^NSEI",
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


# Public financial & macro RSS feeds
RSS_FEEDS: Dict[str, str] = {
    "reuters_business_google": "https://news.google.com/rss/search?q=stock+market+OR+federal+reserve+OR+inflation+when:3d&hl=en-US&gl=US&ceid=US:en",
    "marketwatch_top": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "cnbc_markets": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664",
    "investing_news": "https://www.investing.com/rss/news.rss",
    "economic_times_markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "sec_edgar_8k": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&company=&dateb=&owner=include&start=0&count=40&output=atom",
}


@dataclass
class EngineSettings:
    """Runtime settings for training, tuning, and inference."""

    cache_ttl_hours_daily: float = 6.0
    cache_ttl_hours_intraday: float = 0.5
    cache_ttl_hours_news: float = 0.5
    n_optuna_trials: int = 20
    random_state: int = 42
    max_leverage: float = 1.5
    min_signal_threshold: float = 0.05
    bootstrap_samples: int = 500
    enable_shap: bool = True
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
    """Create local cache, artifact, and report directories if absent."""
    for directory in (DATA_CACHE_DIR, ARTIFACTS_DIR, REPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def is_asia_india_ticker(ticker: str) -> bool:
    """Return True if ticker trades in Indian/Asian session prior to US close."""
    upper = ticker.upper()
    return upper.endswith((".NS", ".BO")) or upper in {"^NSEI", "^NSEBANK", "^INDIAVIX"}


def is_crypto_ticker(ticker: str) -> bool:
    """Return True if ticker trades 24/7 (crypto)."""
    return ticker.upper().endswith(("-USD", "-USDT", "-BTC"))
