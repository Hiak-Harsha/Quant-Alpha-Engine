"""Market OHLCV and volatility index data ingestion via Yahoo Finance with caching."""

from __future__ import annotations

import logging
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import yfinance as yf

from quant_alpha.config import DATA_CACHE_DIR, EngineSettings, ensure_directories

logger = logging.getLogger(__name__)


def _sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def _cache_path(ticker: str, period: str, interval: str) -> Path:
    ensure_directories()
    safe = _sanitize_filename(f"{ticker}__{period}__{interval}")
    return DATA_CACHE_DIR / "ohlcv" / f"{safe}__v2.parquet"


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
    raw_idx = pd.to_datetime(out.index)
    if interval in {"1d", "1wk", "1mo"}:
        # Preserve the exchange calendar date (e.g., NSE 2026-11-26 00:00+05:30 stays 2026-11-26)
        if getattr(raw_idx, "tz", None) is not None:
            idx = raw_idx.tz_localize(None).normalize()
        else:
            idx = raw_idx.normalize()
    else:
        # Intraday bars remain UTC-naive timestamps preserving exact bar close alignment.
        idx = pd.to_datetime(raw_idx, utc=True).tz_convert("UTC").tz_localize(None)

    out.index = idx
    out.index.name = "Timestamp"
    out = out[~out.index.duplicated(keep="last")].sort_index()
    for col in required:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["Close"])
    out = out[out["Close"] > 0]
    return out


def _repair_trailing_daily_nan(
    tk: Any,
    raw: pd.DataFrame,
    norm: pd.DataFrame,
) -> pd.DataFrame:
    """Backfill trailing daily bar if Yahoo returns NaN Close while 1h/fast_info has settled data."""
    if raw is None or raw.empty or norm is None or norm.empty:
        return norm

    raw_idx = pd.to_datetime(raw.index)
    raw_local = raw_idx.tz_localize(None).normalize() if getattr(raw_idx, "tz", None) is not None else raw_idx.normalize()
    raw_max_date = pd.Timestamp(raw_local.max())
    norm_max_date = pd.Timestamp(norm.index.max())
    if raw_max_date <= norm_max_date:
        return norm

    try:
        h1 = tk.history(period="5d", interval="1h", auto_adjust=True, actions=False)
    except Exception:
        return norm
    if h1 is None or h1.empty:
        return norm

    h1_norm = _normalize_ohlcv_df(h1, interval="1h")
    if h1_norm.empty:
        return norm

    # Use exchange calendar date from original intraday index before normalization
    h1_raw_idx = pd.to_datetime(h1.index)
    h1_dates = (
        h1_raw_idx.tz_localize(None).normalize()
        if getattr(h1_raw_idx, "tz", None) is not None
        else h1_raw_idx.normalize()
    )
    h1_work = h1.copy()
    if isinstance(h1_work.columns, pd.MultiIndex):
        h1_work.columns = [str(col[0]) for col in h1_work.columns]
    h1_work.columns = [str(c).strip().title() for c in h1_work.columns]
    h1_work["_day"] = h1_dates
    missing_day_df = h1_work[h1_work["_day"] > norm_max_date].dropna(subset=["Close"])
    if missing_day_df.empty:
        return norm

    fast_last: Optional[float] = None
    try:
        lp = getattr(tk.fast_info, "last_price", None)
        if lp is not None and math.isfinite(float(lp)) and float(lp) > 0.0:
            fast_last = float(lp)
    except Exception:
        fast_last = None

    new_rows = []
    max_missing_day = missing_day_df["_day"].max()
    for day_ts, grp in missing_day_df.groupby("_day", sort=True):
        close_val = float(grp["Close"].iloc[-1])
        if fast_last is not None and day_ts == max_missing_day and day_ts == raw_max_date:
            close_val = fast_last
        hi_val = max(float(grp["High"].max()), close_val)
        lo_val = min(float(grp["Low"].min()), close_val)
        vol_val = float(pd.to_numeric(grp.get("Volume", 0.0), errors="coerce").fillna(0.0).sum())
        new_rows.append(
            {
                "Timestamp": day_ts,
                "Open": float(grp["Open"].iloc[0]),
                "High": hi_val,
                "Low": lo_val,
                "Close": close_val,
                "Volume": vol_val,
            }
        )

    if not new_rows:
        return norm
    add_df = pd.DataFrame(new_rows).set_index("Timestamp")
    merged = pd.concat([norm, add_df], axis=0)
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    merged.index.name = "Timestamp"
    return merged


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
            if interval == "1d" and not norm.empty:
                norm = _repair_trailing_daily_nan(tk, raw, norm)
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
