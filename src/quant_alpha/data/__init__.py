"""Multi-source data ingestion and feed validation modules."""

from quant_alpha.data.macro_feed import fetch_fred_macro, load_all_macro_series
from quant_alpha.data.market_feed import fetch_ohlcv, fetch_universe_ohlcv
from quant_alpha.data.news_feed import fetch_all_news
from quant_alpha.data.trending_feed import discover_trending_stocks
from quant_alpha.data.validator import FeedValidationResult, run_full_feed_audit

__all__ = [
    "FeedValidationResult",
    "discover_trending_stocks",
    "fetch_all_news",
    "fetch_fred_macro",
    "fetch_ohlcv",
    "fetch_universe_ohlcv",
    "load_all_macro_series",
    "run_full_feed_audit",
]
