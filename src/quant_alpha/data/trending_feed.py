"""Dynamic discovery of currently trending, most active, and news-buzzed stocks."""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import yfinance as yf

from quant_alpha.config import DATA_CACHE_DIR, EngineSettings, ensure_directories

logger = logging.getLogger(__name__)

_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
_SKIP_SYMBOLS = {"^GSPC", "^DJI", "^IXIC", "^VIX", "^VXN", "^TNX", "^IRX", "^INDIAVIX"}


def _trending_cache_path(india_focus: bool = False) -> Path:
    ensure_directories()
    folder = DATA_CACHE_DIR / "trending"
    folder.mkdir(parents=True, exist_ok=True)
    suffix = "india" if india_focus else "global"
    return folder / f"trending_stocks_{suffix}.parquet"


def _fetch_yf_trending_region(region: str = "US", count: int = 20) -> List[str]:
    url = f"https://query1.finance.yahoo.com/v1/finance/trending/{region}?count={count}"
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=8)
        resp.raise_for_status()
        quotes = resp.json().get("finance", {}).get("result", [{}])[0].get("quotes", [])
        return [str(q["symbol"]).strip().upper() for q in quotes if q.get("symbol")]
    except Exception as exc:
        logger.debug("Yahoo trending fetch failed for %s: %s", region, exc)
        return []


def _fetch_yf_screener(query_name: str, count: int = 15) -> List[Dict[str, str]]:
    try:
        res = yf.screen(query_name, count=count)
        out = []
        for q in res.get("quotes", []):
            sym = str(q.get("symbol", "")).strip().upper()
            name = str(q.get("shortName") or q.get("longName") or "").strip()
            if sym:
                out.append({"symbol": sym, "name": name})
        return out
    except Exception as exc:
        logger.debug("yfinance screen(%s) failed: %s", query_name, exc)
        return []


def _is_tradable_symbol(sym: str) -> bool:
    if not sym or sym in _SKIP_SYMBOLS:
        return False
    # Exclude futures/options/warrants noise while preserving equities, ETFs, .NS/.BO, COMEX gold/silver, FX & -USD crypto
    if sym in {"GC=F", "SI=F", "BZ=F", "USDINR=X"}:
        return True
    if "=" in sym or sym.startswith("^"):
        return sym in {"^NSEI", "^NSEBANK", "^CNXIT"}
    return bool(re.match(r"^[A-Z0-9.\-&]{1,18}$", sym))


def discover_trending_stocks(
    news_df: Optional[pd.DataFrame] = None,
    candidate_tickers: Optional[List[str]] = None,
    use_cache: bool = True,
    settings: Optional[EngineSettings] = None,
    india_focus: bool = True,
) -> pd.DataFrame:
    """Discover currently trending, most-active, big-mover, and news-buzzed tickers.

    Returns a DataFrame ranked by composite buzz_score with columns:
    [symbol, name, buzz_score, news_mentions, sources].
    """
    settings = settings or EngineSettings()
    cpath = _trending_cache_path(india_focus=india_focus)

    if use_cache and cpath.exists():
        age_hours = (time.time() - cpath.stat().st_mtime) / 3600.0
        if age_hours <= settings.cache_ttl_hours_news:
            try:
                cached = pd.read_parquet(cpath)
                if not cached.empty:
                    return cached
            except Exception:
                pass

    records: Dict[str, Dict[str, Any]] = {}

    def _add(sym: str, source_tag: str, weight: float, name: str = "") -> None:
        if not _is_tradable_symbol(sym):
            return
        entry = records.setdefault(
            sym,
            {"symbol": sym, "name": name, "buzz_score": 0.0, "news_mentions": 0, "sources": set()},
        )
        if name and not entry["name"]:
            entry["name"] = name
        entry["buzz_score"] += weight
        entry["sources"].add(source_tag)

    for rank, sym in enumerate(_fetch_yf_trending_region("IN", 20)):
        boost = 1.5 if (india_focus and (sym.endswith((".NS", ".BO")) or sym.startswith("^NSE"))) else 1.0
        _add(sym, "YF Trending India", max(1.0, (3.2 - rank * 0.1) * boost))

    if not india_focus:
        for rank, sym in enumerate(_fetch_yf_trending_region("US", 20)):
            _add(sym, "YF Trending US", max(1.0, 3.0 - rank * 0.1))

        for qname, tag, base_w in (
            ("most_actives", "Most Active", 2.5),
            ("day_gainers", "Top Gainer", 2.0),
            ("day_losers", "Top Mover (Down)", 2.0),
            ("growth_technology_stocks", "Tech Growth", 1.5),
        ):
            for rank, item in enumerate(_fetch_yf_screener(qname, count=15)):
                _add(item["symbol"], tag, max(0.8, base_w - rank * 0.08), name=item["name"])

    # Score live news headline buzz against candidate & discovered tickers
    if news_df is not None and not news_df.empty:
        from quant_alpha.features.sentiment import TICKER_ALIASES

        titles = news_df["title"].fillna("").str.lower()
        all_check = set(records.keys()) | set(candidate_tickers or []) | set(TICKER_ALIASES.keys())
        for sym in all_check:
            if not _is_tradable_symbol(sym):
                continue
            aliases = TICKER_ALIASES.get(sym, [sym.split(".")[0].lower()])
            # Require at least 3-char token or exact ticker mention to avoid false substring hits
            valid_aliases = [a for a in aliases if len(a) >= 3]
            if not valid_aliases:
                continue
            pattern = r"\b(?:" + "|".join(re.escape(a) for a in valid_aliases) + r")\b"
            ticker_hits = int((news_df["ticker"].str.upper() == sym).sum())
            text_hits = int(titles.str.contains(pattern, regex=True).sum())
            mentions = max(ticker_hits, text_hits)
            if mentions > 0:
                _add(sym, f"News Buzz ({mentions})", min(4.0, mentions * 0.6))
                records[sym]["news_mentions"] = mentions

    if not records:
        if cpath.exists():
            try:
                return pd.read_parquet(cpath)
            except Exception:
                pass
        return pd.DataFrame(columns=["symbol", "name", "buzz_score", "news_mentions", "sources"])

    out_rows = []
    for item in records.values():
        out_rows.append(
            {
                "symbol": item["symbol"],
                "name": item["name"],
                "buzz_score": round(float(item["buzz_score"]), 2),
                "news_mentions": int(item["news_mentions"]),
                "sources": ", ".join(sorted(item["sources"])),
            }
        )

    out_df = (
        pd.DataFrame(out_rows)
        .sort_values(["buzz_score", "news_mentions"], ascending=False)
        .reset_index(drop=True)
    )
    try:
        out_df.to_parquet(cpath, index=False)
    except Exception as exc:
        logger.debug("Failed caching trending_stocks: %s", exc)
    return out_df
