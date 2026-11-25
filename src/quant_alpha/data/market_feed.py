"""Market OHLCV and volatility index data ingestion via Yahoo Finance with caching."""

from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import yfinance as yf

from quant_alpha.config import DATA_CACHE_DIR, EngineSettings, ensure_directories

logger = logging.getLogger(__name__)


def _sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def _cache_path(ticker: str, period: str, interval: str) -> Path:
    ensure_directories()
    safe = _sanitize_filename(f"{ticker}__{period}__{interval}")
    return DATA_CACHE_DIR / "ohlcv" / f"{safe}.parquet"


def _is_cache_fresh(path: Path, ttl_hours: float) -> bool:
    if not path.exists():
        return False
    age_hours = (time.time() - path.stat().st_mtime) / 3600.0
    return age_hours <= ttl_hours


def _normalize_ohlcv_df(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Normalize columns, index timezone, sorting, and types."""
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [str(col[0]) for col in out.columns]
    out.columns = [str(c).strip().title() for c in out.columns]

    required = ["Open", "High", "Low", "Close", "Volume"]
    for col in required:
        if col not in out.columns:
            if col == "Volume":
                out["Volume"] = 0.0
            else:
                return pd.DataFrame()

    out = out[required].copy()
    idx = pd.to_datetime(out.index, utc=True)
    if interval in {"1d", "1wk", "1mo"}:
        # Daily bars are indexed by normalized UTC midnight date for cross-asset joins.
        idx = idx.tz_convert("UTC").normalize().tz_localize(None)
    else:
        # Intraday bars remain UTC-naive timestamps preserving exact bar close alignment.
        idx = idx.tz_convert("UTC").tz_localize(None)

    out.index = idx
    out.index.name = "Timestamp"
    out = out[~out.index.duplicated(keep="last")].sort_index()
    for col in required:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["Close"])
    out = out[out["Close"] > 0]
    return out


def fetch_ohlcv(
    ticker: str,
    period: str = "10y",
    interval: str = "1d",
    use_cache: bool = True,
    settings: Optional[EngineSettings] = None,
    retries: int = 2,
) -> pd.DataFrame:
    """Fetch OHLCV data for a single ticker with disk caching and retries."""
    settings = settings or EngineSettings()
    ttl = (
        settings.cache_ttl_hours_intraday
        if interval not in {"1d", "1wk", "1mo"}
        else settings.cache_ttl_hours_daily
    )
    cpath = _cache_path(ticker, period, interval)
    cpath.parent.mkdir(parents=True, exist_ok=True)

    if use_cache and _is_cache_fresh(cpath, ttl):
        try:
            cached = pd.read_parquet(cpath)
            if not cached.empty:
                cached.index = pd.to_datetime(cached.index)
                return cached
        except Exception as exc:
            logger.warning("Failed reading cache for %s: %s", ticker, exc)

    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            tk = yf.Ticker(ticker)
            raw = tk.history(period=period, interval=interval, auto_adjust=True, actions=False)
            norm = _normalize_ohlcv_df(raw, interval=interval)
            if not norm.empty:
                norm.to_parquet(cpath)
                return norm
        except Exception as exc:
            last_err = exc
            time.sleep(0.5 * (attempt + 1))

    # Fallback to stale cache if offline or throttled.
    if cpath.exists():
        try:
            logger.warning("Using stale cache for %s after error: %s", ticker, last_err)
            cached = pd.read_parquet(cpath)
            cached.index = pd.to_datetime(cached.index)
            return cached
        except Exception:
            pass

    logger.warning("No OHLCV data retrieved for %s (%s, %s): %s", ticker, period, interval, last_err)
    return pd.DataFrame()


def fetch_universe_ohlcv(
    tickers: Iterable[str],
    period: str = "10y",
    interval: str = "1d",
    use_cache: bool = True,
    max_workers: int = 6,
    settings: Optional[EngineSettings] = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
    """Fetch OHLCV for multiple tickers concurrently, returning data and errors."""
    ensure_directories()
    tz_cache_dir = DATA_CACHE_DIR / "yf_tz_cache"
    tz_cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        yf.set_tz_cache_location(str(tz_cache_dir))
    except Exception:
        pass
    unique_tickers: List[str] = []
    seen = set()
    for t in tickers:
        clean = t.strip()
        if clean and clean not in seen:
            seen.add(clean)
            unique_tickers.append(clean)

    data: Dict[str, pd.DataFrame] = {}
    errors: Dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=min(max_workers, max(1, len(unique_tickers)))) as pool:
        futures = {
            pool.submit(
                fetch_ohlcv,
                ticker=t,
                period=period,
                interval=interval,
                use_cache=use_cache,
                settings=settings,
            ): t
            for t in unique_tickers
        }
        for fut in as_completed(futures):
            ticker = futures[fut]
            try:
                df = fut.result()
                if df.empty:
                    errors[ticker] = "empty_response"
                else:
                    data[ticker] = df
            except Exception as exc:
                errors[ticker] = str(exc)

    return data, errors
