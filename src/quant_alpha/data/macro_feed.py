"""Public FRED macroeconomic data ingestion with local Parquet caching."""

from __future__ import annotations

import io
import logging
import time
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import requests

from quant_alpha.config import DATA_CACHE_DIR, FRED_SERIES, EngineSettings, ensure_directories

logger = logging.getLogger(__name__)

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


def _macro_cache_path(series_id: str) -> Path:
    ensure_directories()
    folder = DATA_CACHE_DIR / "macro"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{series_id}.parquet"


def fetch_fred_macro(
    series_id: str,
    use_cache: bool = True,
    settings: Optional[EngineSettings] = None,
    timeout: int = 12,
) -> pd.DataFrame:
    """Fetch a single FRED series via public CSV endpoint (no API key required)."""
    settings = settings or EngineSettings()
    cpath = _macro_cache_path(series_id)

    if use_cache and cpath.exists():
        age_hours = (time.time() - cpath.stat().st_mtime) / 3600.0
        if age_hours <= settings.cache_ttl_hours_daily:
            try:
                cached = pd.read_parquet(cpath)
                cached.index = pd.to_datetime(cached.index)
                return cached
            except Exception:
                pass

    url = FRED_CSV_URL.format(series_id=series_id)
    try:
        resp = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "QuantAlphaEngine/0.1 (research@quantalpha.local)"},
        )
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        if df.shape[1] < 2:
            return pd.DataFrame()
        date_col, val_col = df.columns[0], df.columns[1]
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df[series_id] = pd.to_numeric(df[val_col], errors="coerce")
        out = df[[date_col, series_id]].dropna(subset=[date_col]).set_index(date_col)
        out.index.name = "Timestamp"
        out = out.sort_index()
        out.to_parquet(cpath)
        return out
    except Exception as exc:
        logger.warning("FRED fetch failed for %s: %s", series_id, exc)
        if cpath.exists():
            try:
                cached = pd.read_parquet(cpath)
                cached.index = pd.to_datetime(cached.index)
                return cached
            except Exception:
                pass
        return pd.DataFrame()


def load_all_macro_series(
    series_ids: Optional[Iterable[str]] = None,
    use_cache: bool = True,
    settings: Optional[EngineSettings] = None,
) -> pd.DataFrame:
    """Load and merge multiple FRED series into a single daily macro DataFrame."""
    ids = list(series_ids) if series_ids is not None else list(FRED_SERIES.keys())
    frames = []
    for sid in ids:
        df = fetch_fred_macro(sid, use_cache=use_cache, settings=settings)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, axis=1).sort_index()
    return merged
