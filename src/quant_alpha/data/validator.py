"""Automated feed and data quality validator across OHLCV, macro, and news streams."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class FeedValidationResult:
    """Structured data quality scorecard for an ingested feed or asset."""

    feed_type: str
    name: str
    status: str  # OK | WARN | FAIL
    rows: int
    start_date: str
    end_date: str
    missing_pct: float
    stale_bar_pct: float
    extreme_jump_count: int
    notes: str


def validate_ohlcv_df(name: str, df: pd.DataFrame, feed_type: str = "OHLCV") -> FeedValidationResult:
    """Validate OHLCV integrity: completeness, non-positivity, stale closes, extreme jumps."""
    if df is None or df.empty:
        return FeedValidationResult(
            feed_type=feed_type,
            name=name,
            status="FAIL",
            rows=0,
            start_date="N/A",
            end_date="N/A",
            missing_pct=100.0,
            stale_bar_pct=100.0,
            extreme_jump_count=0,
            notes="Empty dataset",
        )

    rows = len(df)
    start_date = str(df.index.min())[:19]
    end_date = str(df.index.max())[:19]
    missing_pct = float(df[["Open", "High", "Low", "Close"]].isna().mean().mean() * 100.0)

    close = df["Close"].astype(float)
    returns = close.pct_change().dropna()
    stale_bar_pct = float((returns == 0).mean() * 100.0) if len(returns) else 0.0
    extreme_jump_count = int((returns.abs() > 0.45).sum())

    notes: List[str] = []
    status = "OK"
    if rows < 120:
        status = "WARN"
        notes.append(f"short_history({rows})")
    if missing_pct > 2.0:
        status = "WARN"
        notes.append(f"missing({missing_pct:.2f}%)")
    if stale_bar_pct > 15.0:
        status = "WARN"
        notes.append(f"stale_bars({stale_bar_pct:.1f}%)")
    if (close <= 0).any() or (df["High"] < df["Low"]).any():
        status = "WARN"
        notes.append("ohlc_consistency_anomaly")
    if extreme_jump_count > 3:
        notes.append(f"large_jumps({extreme_jump_count})")

    return FeedValidationResult(
        feed_type=feed_type,
        name=name,
        status=status,
        rows=rows,
        start_date=start_date,
        end_date=end_date,
        missing_pct=round(missing_pct, 3),
        stale_bar_pct=round(stale_bar_pct, 2),
        extreme_jump_count=extreme_jump_count,
        notes=", ".join(notes) if notes else "healthy",
    )


def validate_macro_df(macro_df: pd.DataFrame) -> List[FeedValidationResult]:
    """Validate FRED macroeconomic series coverage and missingness."""
    if macro_df is None or macro_df.empty:
        return [
            FeedValidationResult(
                feed_type="FRED_MACRO",
                name="ALL_FRED",
                status="WARN",
                rows=0,
                start_date="N/A",
                end_date="N/A",
                missing_pct=100.0,
                stale_bar_pct=0.0,
                extreme_jump_count=0,
                notes="FRED feed unavailable; using market macro proxies",
            )
        ]

    results: List[FeedValidationResult] = []
    for col in macro_df.columns:
        s = macro_df[col].dropna()
        if s.empty:
            results.append(
                FeedValidationResult("FRED_MACRO", col, "FAIL", 0, "N/A", "N/A", 100.0, 0.0, 0, "empty")
            )
            continue
        missing_pct = float(macro_df[col].isna().mean() * 100.0)
        results.append(
            FeedValidationResult(
                feed_type="FRED_MACRO",
                name=col,
                status="OK" if len(s) >= 200 else "WARN",
                rows=int(len(s)),
                start_date=str(s.index.min())[:10],
                end_date=str(s.index.max())[:10],
                missing_pct=round(missing_pct, 2),
                stale_bar_pct=0.0,
                extreme_jump_count=0,
                notes="healthy (causal 1-day publication lag applied)" if len(s) >= 200 else "sparse",
            )
        )
    return results


def validate_news_df(news_df: pd.DataFrame) -> List[FeedValidationResult]:
    """Validate live news feeds by feed source, count, and recency."""
    if news_df is None or news_df.empty:
        return [
            FeedValidationResult(
                feed_type="NEWS_NLP",
                name="ALL_NEWS",
                status="WARN",
                rows=0,
                start_date="N/A",
                end_date="N/A",
                missing_pct=100.0,
                stale_bar_pct=0.0,
                extreme_jump_count=0,
                notes="No news items retrieved",
            )
        ]

    results: List[FeedValidationResult] = []
    for feed_name, group in news_df.groupby("feed"):
        missing_pct = float(group["title"].isna().mean() * 100.0)
        results.append(
            FeedValidationResult(
                feed_type="NEWS_NLP",
                name=str(feed_name),
                status="OK" if len(group) >= 5 else "WARN",
                rows=int(len(group)),
                start_date=str(group["published_at"].min())[:19],
                end_date=str(group["published_at"].max())[:19],
                missing_pct=round(missing_pct, 2),
                stale_bar_pct=0.0,
                extreme_jump_count=0,
                notes=f"{group['source'].nunique()} unique publishers",
            )
        )
    return results


def run_full_feed_audit(
    ohlcv_map: Dict[str, pd.DataFrame],
    macro_df: pd.DataFrame,
    news_df: pd.DataFrame,
    driver_tickers: Optional[set[str]] = None,
) -> pd.DataFrame:
    """Compile a consolidated audit DataFrame covering every ingested market, macro, and news feed."""
    driver_tickers = driver_tickers or set()
    rows: List[Dict] = []

    for ticker, df in sorted(ohlcv_map.items()):
        ftype = "DRIVER_INDEX_VOL" if ticker in driver_tickers else "ASSET_OHLCV"
        rows.append(asdict(validate_ohlcv_df(ticker, df, feed_type=ftype)))

    for res in validate_macro_df(macro_df):
        rows.append(asdict(res))

    for res in validate_news_df(news_df):
        rows.append(asdict(res))

    return pd.DataFrame(rows)
