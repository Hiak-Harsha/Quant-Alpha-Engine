"""Financial NLP sentiment engine combining VADER and domain-specific market lexicon."""

from __future__ import annotations

import re
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# Financial lexicon adjustments calibrated for markets, earnings, and central banks
FIN_LEXICON: Dict[str, float] = {
    "beat": 2.1,
    "beats": 2.1,
    "outperform": 2.0,
    "upgrade": 2.2,
    "upgraded": 2.2,
    "bullish": 2.4,
    "rally": 2.0,
    "rallies": 2.0,
    "surge": 2.1,
    "surges": 2.1,
    "soar": 2.2,
    "record high": 2.0,
    "guidance raise": 2.5,
    "raises guidance": 2.5,
    "rate cut": 1.6,
    "dovish": 1.5,
    "cooling inflation": 1.7,
    "buyback": 1.6,
    "dividend hike": 1.8,
    "breakout": 1.6,
    "miss": -2.1,
    "misses": -2.1,
    "downgrade": -2.3,
    "downgraded": -2.3,
    "bearish": -2.4,
    "plunge": -2.4,
    "plunges": -2.4,
    "slump": -2.0,
    "selloff": -2.2,
    "sell-off": -2.2,
    "crash": -2.8,
    "recession": -2.3,
    "stagflation": -2.5,
    "hawkish": -1.6,
    "rate hike": -1.7,
    "default": -2.8,
    "bankruptcy": -3.2,
    "lawsuit": -1.8,
    "layoffs": -1.8,
    "tariff": -1.5,
    "tariffs": -1.5,
    "cuts guidance": -2.6,
    "slashed guidance": -2.8,
    "warning": -1.9,
}

TICKER_ALIASES: Dict[str, List[str]] = {
    "SPY": ["s&p 500", "s&p", "spy", "wall street", "us stocks", "equities"],
    "QQQ": ["nasdaq", "qqq", "tech stocks", "big tech", "ai stocks"],
    "IWM": ["russell", "small cap", "small-cap", "iwm"],
    "AAPL": ["apple", "aapl", "iphone"],
    "MSFT": ["microsoft", "msft", "azure"],
    "NVDA": ["nvidia", "nvda", "blackwell", "gpu"],
    "AMZN": ["amazon", "amzn", "aws"],
    "GOOGL": ["alphabet", "google", "googl", "goog", "gemini"],
    "GOOG": ["alphabet", "google", "googl", "goog"],
    "META": ["meta", "facebook", "instagram", "whatsapp"],
    "TSLA": ["tesla", "tsla", "musk", "cybertruck"],
    "AMD": ["amd", "advanced micro", "ryzen", "radeon"],
    "AVGO": ["broadcom", "avgo"],
    "PLTR": ["palantir", "pltr"],
    "NFLX": ["netflix", "nflx"],
    "ORCL": ["oracle", "orcl"],
    "MU": ["micron", "hbm", "dram"],
    "INTC": ["intel", "intc"],
    "CRM": ["salesforce", "crm"],
    "JPM": ["jpmorgan", "jp morgan", "jpm"],
    "GS": ["goldman sachs", "goldman"],
    "V": ["visa"],
    "LLY": ["eli lilly", "lilly", "lly", "mounjaro", "zepbound"],
    "UNH": ["unitedhealth", "unh"],
    "XOM": ["exxon", "xom", "oil major"],
    "CVX": ["chevron", "cvx"],
    "COST": ["costco", "cost"],
    "WMT": ["walmart", "wmt"],
    "BA": ["boeing"],
    "TLT": ["treasury", "treasuries", "bond yield", "bonds", "fed", "fomc"],
    "GLD": ["gold", "bullion", "gld"],
    "SLV": ["silver", "slv"],
    "USO": ["crude", "oil", "opec", "brent", "wti"],
    "INDA": ["india", "nifty", "sensex", "rbi", "indian market"],
    "^NSEI": ["nifty", "sensex", "india", "nse", "rbi", "indian stocks"],
    "^NSEBANK": ["nifty bank", "bank nifty", "indian banks", "rbi"],
    "RELIANCE.NS": ["reliance", "jio", "ambani"],
    "TCS.NS": ["tcs", "tata consultancy"],
    "HDFCBANK.NS": ["hdfc bank", "hdfc"],
    "INFY.NS": ["infosys", "infy"],
    "ICICIBANK.NS": ["icici bank", "icici"],
    "BHARTIARTL.NS": ["airtel", "bharti airtel"],
    "BTC-USD": ["bitcoin", "btc", "crypto", "spot etf"],
    "ETH-USD": ["ethereum", "eth", "crypto"],
}


def _build_analyzer() -> SentimentIntensityAnalyzer:
    analyzer = SentimentIntensityAnalyzer()
    analyzer.lexicon.update({k: v for k, v in FIN_LEXICON.items() if " " not in k})
    return analyzer


_ANALYZER = _build_analyzer()


def score_headline(text: str) -> float:
    """Return sentiment score in [-1.0, +1.0] combining VADER + multi-word financial phrases."""
    if not text:
        return 0.0
    base = float(_ANALYZER.polarity_scores(text)["compound"])
    lower = text.lower()
    phrase_bonus = 0.0
    for phrase, weight in FIN_LEXICON.items():
        if " " in phrase and phrase in lower:
            phrase_bonus += np.sign(weight) * 0.22
    return float(np.clip(base + phrase_bonus, -1.0, 1.0))


def score_news_dataframe(news_df: pd.DataFrame) -> pd.DataFrame:
    """Score every headline in news_df with financial sentiment and polarity label."""
    if news_df is None or news_df.empty:
        return pd.DataFrame(
            columns=["published_at", "title", "summary", "source", "feed", "ticker", "url", "sentiment_score", "sentiment_label"]
        )

    out = news_df.copy()
    combined_text = (out["title"].fillna("") + ". " + out["summary"].fillna("").str[:200]).str.strip()
    scores = [score_headline(t) for t in combined_text]
    out["sentiment_score"] = scores
    out["sentiment_label"] = np.where(
        out["sentiment_score"] >= 0.15,
        "Bullish",
        np.where(out["sentiment_score"] <= -0.15, "Bearish", "Neutral"),
    )
    return out


def compute_news_sentiment_summary(scored_news: pd.DataFrame, target_ticker: str) -> Dict[str, Any]:
    """Aggregate asset-specific + macro news sentiment with exponential recency decay."""
    if scored_news is None or scored_news.empty:
        return {
            "asset_sentiment": 0.0,
            "macro_sentiment": 0.0,
            "composite_sentiment": 0.0,
            "asset_article_count": 0,
            "macro_article_count": 0,
            "bullish_pct": 0.0,
            "bearish_pct": 0.0,
            "top_headlines": [],
        }

    df = scored_news.copy()
    now = pd.Timestamp.utcnow().tz_localize(None)
    pub = pd.to_datetime(df["published_at"], errors="coerce").fillna(now)
    age_hours = np.maximum(0.0, (now - pub).dt.total_seconds() / 3600.0)
    # 36-hour half-life decay
    df["recency_weight"] = np.exp(-age_hours / 36.0)

    aliases = TICKER_ALIASES.get(target_ticker.upper(), [target_ticker.split(".")[0].lower()])
    pattern = "|".join(re.escape(a) for a in aliases)
    title_lower = df["title"].fillna("").str.lower()
    asset_mask = (df["ticker"].str.upper() == target_ticker.upper()) | title_lower.str.contains(pattern, regex=True)

    asset_df = df[asset_mask]
    macro_df = df[df["ticker"].str.upper() == "MACRO"]

    def _weighted_mean(sub: pd.DataFrame) -> float:
        if sub.empty:
            return 0.0
        w = sub["recency_weight"].to_numpy()
        s = sub["sentiment_score"].to_numpy()
        return float(np.sum(w * s) / max(np.sum(w), 1e-8))

    asset_sent = _weighted_mean(asset_df)
    macro_sent = _weighted_mean(macro_df)

    if not asset_df.empty and not macro_df.empty:
        composite = 0.65 * asset_sent + 0.35 * macro_sent
    elif not asset_df.empty:
        composite = asset_sent
    else:
        composite = macro_sent

    relevant = asset_df if not asset_df.empty else macro_df
    bullish_pct = float((relevant["sentiment_score"] >= 0.15).mean() * 100.0) if not relevant.empty else 0.0
    bearish_pct = float((relevant["sentiment_score"] <= -0.15).mean() * 100.0) if not relevant.empty else 0.0

    top_rows = (
        relevant.sort_values(["recency_weight", "published_at"], ascending=False)
        .head(6)[["published_at", "title", "source", "sentiment_score", "sentiment_label"]]
        .to_dict(orient="records")
    )

    return {
        "asset_sentiment": round(asset_sent, 4),
        "macro_sentiment": round(macro_sent, 4),
        "composite_sentiment": round(float(composite), 4),
        "asset_article_count": int(len(asset_df)),
        "macro_article_count": int(len(macro_df)),
        "bullish_pct": round(bullish_pct, 1),
        "bearish_pct": round(bearish_pct, 1),
        "top_headlines": top_rows,
    }
