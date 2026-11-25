"""Multi-source financial news ingestion (yfinance, GDELT 2.0, RSS, SEC EDGAR, optional API keys)."""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import quote_plus

import feedparser
import pandas as pd
import requests
import yfinance as yf

from quant_alpha.config import DATA_CACHE_DIR, RSS_FEEDS, EngineSettings, ensure_directories

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "QuantAlphaEngine/0.1 (research@quantalpha.local)",
    "Accept": "application/json, application/xml, text/xml, */*",
}


def _news_cache_path() -> Path:
    ensure_directories()
    folder = DATA_CACHE_DIR / "news"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / "latest_news.parquet"


def _parse_datetime(raw_value: Any) -> pd.Timestamp:
    if raw_value is None or raw_value == "":
        return pd.Timestamp.now(tz="UTC").tz_localize(None)
    if isinstance(raw_value, (int, float)):
        try:
            return pd.to_datetime(raw_value, unit="s", utc=True).tz_localize(None)
        except Exception:
            pass
    try:
        ts = pd.to_datetime(raw_value, utc=True)
        return ts.tz_localize(None)
    except Exception:
        return pd.Timestamp.now(tz="UTC").tz_localize(None)


def fetch_yfinance_news(tickers: Iterable[str], limit_per_ticker: int = 12) -> List[Dict[str, Any]]:
    """Fetch recent ticker-linked news items from Yahoo Finance."""
    records: List[Dict[str, Any]] = []
    for ticker in tickers:
        try:
            items = yf.Ticker(ticker).news or []
            for item in items[:limit_per_ticker]:
                content = item.get("content", {}) if isinstance(item, dict) else {}
                title = (
                    item.get("title")
                    or content.get("title")
                    or ""
                )
                summary = content.get("summary") or item.get("summary") or ""
                link = (
                    item.get("link")
                    or (content.get("canonicalUrl") or {}).get("url")
                    or ""
                )
                publisher = (
                    item.get("publisher")
                    or (content.get("provider") or {}).get("displayName")
                    or "yfinance"
                )
                pub_time = (
                    item.get("providerPublishTime")
                    or content.get("pubDate")
                    or content.get("displayTime")
                )
                if title:
                    records.append(
                        {
                            "published_at": _parse_datetime(pub_time),
                            "title": str(title).strip(),
                            "summary": str(summary).strip(),
                            "source": f"yf:{publisher}",
                            "feed": "yfinance_news",
                            "ticker": ticker,
                            "url": str(link).strip(),
                        }
                    )
        except Exception as exc:
            logger.debug("yfinance news error for %s: %s", ticker, exc)
    return records


def fetch_rss_news(feeds: Optional[Dict[str, str]] = None, limit_per_feed: int = 30) -> List[Dict[str, Any]]:
    """Fetch macro, markets, and SEC 8-K Atom/RSS feeds."""
    feed_map = feeds or RSS_FEEDS
    records: List[Dict[str, Any]] = []
    for feed_name, url in feed_map.items():
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            resp.raise_for_status()
            parsed = feedparser.parse(resp.content)
            for entry in parsed.entries[:limit_per_feed]:
                title = getattr(entry, "title", "") or ""
                summary = getattr(entry, "summary", "") or ""
                link = getattr(entry, "link", "") or ""
                published = getattr(entry, "published", None) or getattr(entry, "updated", None)
                if title:
                    records.append(
                        {
                            "published_at": _parse_datetime(published),
                            "title": str(title).strip(),
                            "summary": str(summary).strip()[:500],
                            "source": feed_name,
                            "feed": "rss_atom",
                            "ticker": "MACRO",
                            "url": str(link).strip(),
                        }
                    )
        except Exception as exc:
            logger.debug("RSS fetch error for %s: %s", feed_name, exc)
    return records


def fetch_gdelt_news(
    query: str = '("stock market" OR "Federal Reserve" OR "inflation" OR "earnings" OR "crude oil" OR "bitcoin") sourcelang:english',
    max_records: int = 50,
    timespan: str = "3d",
) -> List[Dict[str, Any]]:
    """Fetch recent global macro/financial news via free GDELT 2.0 Doc API."""
    url = (
        "https://api.gdeltproject.org/api/v2/doc/doc"
        f"?query={quote_plus(query)}&mode=ArtList&maxrecords={max_records}&timespan={timespan}&format=json"
    )
    records: List[Dict[str, Any]] = []
    try:
        resp = requests.get(url, headers=HEADERS, timeout=12)
        resp.raise_for_status()
        payload = resp.json()
        for art in payload.get("articles", []):
            title = art.get("title", "")
            if not title:
                continue
            seendate = art.get("seendate", "")
            # GDELT format: YYYYMMDDTHHMMSSZ
            parsed_ts = _parse_datetime(seendate)
            records.append(
                {
                    "published_at": parsed_ts,
                    "title": str(title).strip(),
                    "summary": "",
                    "source": str(art.get("domain", "gdelt")).strip(),
                    "feed": "gdelt_v2",
                    "ticker": "MACRO",
                    "url": str(art.get("url", "")).strip(),
                }
            )
    except Exception as exc:
        logger.debug("GDELT fetch error: %s", exc)
    return records


def fetch_optional_api_news(tickers: Iterable[str]) -> List[Dict[str, Any]]:
    """Query Finnhub / NewsAPI / AlphaVantage automatically if API keys exist in env."""
    records: List[Dict[str, Any]] = []

    finnhub_key = os.getenv("FINNHUB_API_KEY")
    if finnhub_key:
        try:
            resp = requests.get(
                f"https://finnhub.io/api/v1/news?category=general&token={finnhub_key}",
                timeout=10,
            )
            if resp.ok:
                for item in resp.json()[:40]:
                    records.append(
                        {
                            "published_at": _parse_datetime(item.get("datetime")),
                            "title": str(item.get("headline", "")).strip(),
                            "summary": str(item.get("summary", "")).strip()[:500],
                            "source": f"finnhub:{item.get('source', 'api')}",
                            "feed": "finnhub_api",
                            "ticker": "MACRO",
                            "url": str(item.get("url", "")).strip(),
                        }
                    )
        except Exception as exc:
            logger.debug("Finnhub news error: %s", exc)

    newsapi_key = os.getenv("NEWSAPI_KEY")
    if newsapi_key:
        try:
            q = quote_plus("stock market OR equities OR Federal Reserve OR NIFTY")
            resp = requests.get(
                f"https://newsapi.org/v2/everything?q={q}&language=en&sortBy=publishedAt&pageSize=30&apiKey={newsapi_key}",
                timeout=10,
            )
            if resp.ok:
                for art in resp.json().get("articles", []):
                    records.append(
                        {
                            "published_at": _parse_datetime(art.get("publishedAt")),
                            "title": str(art.get("title", "")).strip(),
                            "summary": str(art.get("description", "")).strip()[:500],
                            "source": f"newsapi:{(art.get('source') or {}).get('name', 'api')}",
                            "feed": "newsapi",
                            "ticker": "MACRO",
                            "url": str(art.get("url", "")).strip(),
                        }
                    )
        except Exception as exc:
            logger.debug("NewsAPI error: %s", exc)

    av_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if av_key:
        try:
            ticker_str = ",".join([t for t in tickers if "^" not in t and "." not in t][:5]) or "SPY"
            resp = requests.get(
                f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker_str}&limit=30&apikey={av_key}",
                timeout=10,
            )
            if resp.ok:
                for item in resp.json().get("feed", []):
                    records.append(
                        {
                            "published_at": _parse_datetime(item.get("time_published")),
                            "title": str(item.get("title", "")).strip(),
                            "summary": str(item.get("summary", "")).strip()[:500],
                            "source": f"alphavantage:{item.get('source', 'api')}",
                            "feed": "alphavantage_news",
                            "ticker": "MACRO",
                            "url": str(item.get("url", "")).strip(),
                        }
                    )
        except Exception as exc:
            logger.debug("AlphaVantage news error: %s", exc)

    return records


def fetch_all_news(
    tickers: Iterable[str],
    use_cache: bool = True,
    settings: Optional[EngineSettings] = None,
) -> pd.DataFrame:
    """Aggregate, deduplicate, and cache live financial & macro news from all active feeds."""
    settings = settings or EngineSettings()
    cpath = _news_cache_path()

    if use_cache and cpath.exists():
        age_hours = (time.time() - cpath.stat().st_mtime) / 3600.0
        if age_hours <= settings.cache_ttl_hours_news:
            try:
                cached = pd.read_parquet(cpath)
                if not cached.empty:
                    return cached
            except Exception:
                pass

    ticker_list = list(tickers)
    all_records: List[Dict[str, Any]] = []
    all_records.extend(fetch_yfinance_news(ticker_list))
    all_records.extend(fetch_rss_news())
    all_records.extend(fetch_gdelt_news())
    all_records.extend(fetch_optional_api_news(ticker_list))

    if not all_records:
        if cpath.exists():
            try:
                return pd.read_parquet(cpath)
            except Exception:
                pass
        return pd.DataFrame(
            columns=["published_at", "title", "summary", "source", "feed", "ticker", "url"]
        )

    df = pd.DataFrame(all_records)
    df["title_norm"] = df["title"].str.lower().str.replace(r"\W+", " ", regex=True).str.strip()
    df = df[df["title_norm"].str.len() > 8]
    df = df.sort_values("published_at", ascending=False)
    df = df.drop_duplicates(subset=["title_norm", "ticker"], keep="first").drop(columns=["title_norm"])
    df = df.reset_index(drop=True)

    try:
        df.to_parquet(cpath, index=False)
    except Exception as exc:
        logger.warning("Failed to cache news dataframe: %s", exc)

    return df
