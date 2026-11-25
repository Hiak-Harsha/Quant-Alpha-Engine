"""Causal technical, volatility, cross-asset correlation, regime, and NLP sentiment features."""

from quant_alpha.features.cross_asset import build_cross_asset_features
from quant_alpha.features.regime import MarketRegimeDetector
from quant_alpha.features.sentiment import compute_news_sentiment_summary, score_news_dataframe
from quant_alpha.features.technical import compute_technical_features
from quant_alpha.features.volatility import compute_volatility_features

__all__ = [
    "MarketRegimeDetector",
    "build_cross_asset_features",
    "compute_news_sentiment_summary",
    "compute_technical_features",
    "compute_volatility_features",
    "score_news_dataframe",
]
