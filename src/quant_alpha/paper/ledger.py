"""SQLite forward paper-trading ledger with strict bar-maturity and conservative barrier evaluation."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from quant_alpha.config import (
    HORIZON_CONFIGS,
    PROJECT_ROOT,
    HorizonConfig,
    is_asia_india_ticker,
    is_crypto_ticker,
)
from quant_alpha.data.market_feed import fetch_ohlcv
from quant_alpha.predictor import LivePredictionReport

DEFAULT_LEDGER_PATH = PROJECT_ROOT / "paper_trading" / "paper_ledger.db"

_INTERVAL_DELTAS: Dict[str, pd.Timedelta] = {
    "1m": pd.Timedelta(minutes=1),
    "2m": pd.Timedelta(minutes=2),
    "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15),
    "30m": pd.Timedelta(minutes=30),
    "60m": pd.Timedelta(hours=1),
    "1h": pd.Timedelta(hours=1),
    "90m": pd.Timedelta(minutes=90),
}


def utc_now_naive() -> pd.Timestamp:
    """Return current UTC time as tz-naive Timestamp matching normalized market bars."""
    return pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None)


def filter_closed_bars(
    df: pd.DataFrame,
    interval: str,
    ticker: str,
    now_utc: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Return only fully settled/closed bars relative to now_utc to prevent mid-bar repainting."""
    if df is None or df.empty:
        return pd.DataFrame()

    ref_now = pd.Timestamp(now_utc) if now_utc is not None else utc_now_naive()
    if ref_now.tzinfo is not None:
        ref_now = ref_now.tz_convert("UTC").tz_localize(None)

    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)

    out = df.copy()
    out.index = idx

    if interval in _INTERVAL_DELTAS:
        delta = _INTERVAL_DELTAS[interval]
        mask = (out.index + delta) <= ref_now
        return out.loc[mask]

    # Daily bars: prior calendar dates are always closed; same-date bars check exchange bell + 15m buffer
    bar_dates = out.index.date
    ref_date = ref_now.date()
    prior_mask = bar_dates < ref_date

    if is_crypto_ticker(ticker):
        # 24/7 UTC daily bar closes at end of UTC day
        same_day_closed = np.zeros(len(out), dtype=bool)
    elif is_asia_india_ticker(ticker):
        # NSE/BSE closes 15:30 IST = 10:00 UTC (+15m settle buffer)
        is_settled = (ref_now.hour > 10) or (ref_now.hour == 10 and ref_now.minute >= 15)
        same_day_closed = (bar_dates == ref_date) & is_settled
    else:
        # US/Global daily close (~21:00 UTC standard / 20:00 UTC DST; 21:15 UTC covers both safely)
        is_settled = (ref_now.hour > 21) or (ref_now.hour == 21 and ref_now.minute >= 15)
        same_day_closed = (bar_dates == ref_date) & is_settled

    return out.loc[prior_mask | same_day_closed]


def _direction_from_verdict(verdict: str) -> int:
    v = verdict.upper()
    if "BULLISH" in v:
        return 1
    if "BEARISH" in v:
        return -1
    return 0


@dataclass
class EvaluationBatchResult:
    """Summary of a paper-ledger grading pass."""

    checked_open: int
    newly_matured: int
    still_open: int
    errors: Dict[str, str]


class PaperLedger:
    """Persistent forward paper-trading ledger backed by SQLite."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = Path(db_path or DEFAULT_LEDGER_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS paper_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    logged_at_utc TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    horizon_name TEXT NOT NULL,
                    forecast_horizon INTEGER NOT NULL,
                    as_of_timestamp TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    verdict TEXT NOT NULL,
                    direction INTEGER NOT NULL,
                    quant_expected_ret_pct REAL NOT NULL,
                    composite_expected_ret_pct REAL NOT NULL,
                    prob_up_pct REAL NOT NULL,
                    model_agreement_pct REAL NOT NULL,
                    suggested_allocation_pct REAL NOT NULL,
                    stop_loss_price REAL NOT NULL,
                    take_profit_price REAL NOT NULL,
                    garch_ann_vol_pct REAL NOT NULL,
                    regime_label TEXT NOT NULL,
                    news_sentiment REAL NOT NULL,
                    cost_hurdle_pct REAL NOT NULL,
                    model_preds_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'OPEN',
                    exit_timestamp TEXT,
                    exit_price REAL,
                    actual_asset_ret_pct REAL,
                    direction_correct INTEGER,
                    gross_strategy_ret_pct REAL,
                    net_strategy_ret_pct REAL,
                    risk_managed_ret_pct REAL,
                    allocated_hold_pnl_pct REAL,
                    allocated_risk_pnl_pct REAL,
                    barrier_outcome TEXT,
                    mfe_pct REAL,
                    mae_pct REAL,
                    evaluated_at_utc TEXT,
                    UNIQUE(ticker, horizon_name, as_of_timestamp)
                )
                """
            )

    def log_prediction(
        self,
        report: LivePredictionReport,
        horizon_cfg: Optional[HorizonConfig] = None,
        logged_at_utc: Optional[pd.Timestamp] = None,
    ) -> Tuple[bool, str]:
        """Insert a live prediction row idempotently per (ticker, horizon, as_of_timestamp)."""
        hcfg = horizon_cfg or HORIZON_CONFIGS[report.horizon_name]
        ts_now = (logged_at_utc or utc_now_naive()).strftime("%Y-%m-%d %H:%M:%S")
        direction = _direction_from_verdict(report.verdict)
        news_sent = float((report.news_summary or {}).get("composite_sentiment", 0.0))
        model_preds_json = json.dumps(report.model_breakdown)

        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO paper_predictions (
                    logged_at_utc, ticker, horizon_name, forecast_horizon,
                    as_of_timestamp, entry_price, verdict, direction,
                    quant_expected_ret_pct, composite_expected_ret_pct,
                    prob_up_pct, model_agreement_pct, suggested_allocation_pct,
                    stop_loss_price, take_profit_price, garch_ann_vol_pct,
                    regime_label, news_sentiment, cost_hurdle_pct, model_preds_json, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')
                """,
                (
                    ts_now,
                    report.ticker,
                    report.horizon_name,
                    int(hcfg.forecast_horizon),
                    str(report.as_of_timestamp),
                    float(report.last_close),
                    report.verdict,
                    int(direction),
                    float(report.quant_expected_ret_pct),
                    float(report.composite_expected_ret_pct),
                    float(report.prob_up_pct),
                    float(report.model_agreement_pct),
                    float(report.suggested_allocation_pct),
                    float(report.stop_loss_price),
                    float(report.take_profit_price),
                    float(report.garch_ann_vol_pct),
                    str(report.regime_label),
                    news_sent,
                    float(report.cost_hurdle_pct),
                    model_preds_json,
                ),
            )
            if cur.rowcount == 0:
                return False, f"Already logged for {report.ticker} ({report.horizon_name} @ {report.as_of_timestamp})"
        return True, f"Logged {report.ticker} ({report.horizon_name} @ {report.as_of_timestamp}): {report.verdict}"

    def evaluate_open_predictions(
        self,
        ohlcv_fetcher: Optional[Callable[[str, str, str], pd.DataFrame]] = None,
        now_utc: Optional[pd.Timestamp] = None,
    ) -> EvaluationBatchResult:
        """Evaluate matured OPEN predictions strictly against closed future bars."""
        ref_now = now_utc or utc_now_naive()
        with self._connect() as conn:
            open_rows = conn.execute(
                "SELECT * FROM paper_predictions WHERE status = 'OPEN' ORDER BY as_of_timestamp ASC"
            ).fetchall()

        if not open_rows:
            return EvaluationBatchResult(0, 0, 0, {})

        fetcher = ohlcv_fetcher or (
            lambda ticker, period, interval: fetch_ohlcv(
                ticker=ticker, period=period, interval=interval, use_cache=False
            )
        )

        # Fetch once per unique (ticker, horizon_name)
        bars_cache: Dict[Tuple[str, str], pd.DataFrame] = {}
        errors: Dict[str, str] = {}
        for row in open_rows:
            key = (row["ticker"], row["horizon_name"])
            if key in bars_cache:
                continue
            hcfg = HORIZON_CONFIGS[row["horizon_name"]]
            age_days = max(1, int((ref_now - pd.to_datetime(row["as_of_timestamp"])).total_seconds() // 86400))
            if hcfg.interval not in {"1d", "1wk", "1mo"}:
                fetch_period = "730d" if age_days > 45 else "60d"
            else:
                fetch_period = "2y" if age_days > 120 else "6mo"
            try:
                raw = fetcher(row["ticker"], fetch_period, hcfg.interval)
                bars_cache[key] = filter_closed_bars(raw, hcfg.interval, row["ticker"], now_utc=ref_now)
            except Exception as exc:
                bars_cache[key] = pd.DataFrame()
                errors[f"{key[0]}:{key[1]}"] = str(exc)

        newly_matured = 0
        still_open = 0
        eval_ts_str = ref_now.strftime("%Y-%m-%d %H:%M:%S")

        with self._connect() as conn:
            for row in open_rows:
                closed_df = bars_cache.get((row["ticker"], row["horizon_name"]), pd.DataFrame())
                if closed_df.empty:
                    still_open += 1
                    continue

                as_of_ts = pd.to_datetime(row["as_of_timestamp"])
                future_bars = closed_df[closed_df.index > as_of_ts].sort_index()
                horizon = int(row["forecast_horizon"])
                if len(future_bars) < horizon:
                    still_open += 1
                    continue

                window = future_bars.iloc[:horizon]
                entry_price = float(row["entry_price"])
                exit_price = float(window.iloc[-1]["Close"])
                exit_ts = str(window.index[-1])[:19]
                direction = int(row["direction"])
                cost_pct = float(row["cost_hurdle_pct"])
                alloc_frac = abs(float(row["suggested_allocation_pct"])) / 100.0
                sl_price = float(row["stop_loss_price"])
                tp_price = float(row["take_profit_price"])

                actual_asset_ret_pct = ((exit_price / entry_price) - 1.0) * 100.0

                if direction == 0:
                    dir_correct: Optional[int] = None
                    gross_hold = 0.0
                    net_hold = 0.0
                    risk_managed_ret = 0.0
                    alloc_hold_pnl = 0.0
                    alloc_risk_pnl = 0.0
                    barrier_outcome = "FLAT"
                    mfe_pct = 0.0
                    mae_pct = 0.0
                else:
                    dir_correct = 1 if (direction * actual_asset_ret_pct) > 0.0 else 0
                    gross_hold = direction * actual_asset_ret_pct
                    net_hold = gross_hold - cost_pct

                    # Conservative bar-by-bar barrier inspection (Stop-Loss takes priority if both hit same bar)
                    exit_rm = exit_price
                    barrier_outcome = "HOLD_WIN" if gross_hold > 0.0 else "HOLD_LOSS"
                    for _, bar in window.iterrows():
                        hi = float(bar["High"])
                        lo = float(bar["Low"])
                        if direction == 1:
                            if lo <= sl_price:
                                exit_rm = sl_price
                                barrier_outcome = "STOP_LOSS"
                                break
                            if hi >= tp_price:
                                exit_rm = tp_price
                                barrier_outcome = "TAKE_PROFIT"
                                break
                        else:
                            if hi >= sl_price:
                                exit_rm = sl_price
                                barrier_outcome = "STOP_LOSS"
                                break
                            if lo <= tp_price:
                                exit_rm = tp_price
                                barrier_outcome = "TAKE_PROFIT"
                                break

                    risk_managed_ret = direction * ((exit_rm / entry_price) - 1.0) * 100.0 - cost_pct
                    alloc_hold_pnl = alloc_frac * net_hold
                    alloc_risk_pnl = alloc_frac * risk_managed_ret

                    if direction == 1:
                        mfe_pct = ((float(window["High"].max()) / entry_price) - 1.0) * 100.0
                        mae_pct = ((float(window["Low"].min()) / entry_price) - 1.0) * 100.0
                    else:
                        mfe_pct = (1.0 - (float(window["Low"].min()) / entry_price)) * 100.0
                        mae_pct = (1.0 - (float(window["High"].max()) / entry_price)) * 100.0

                conn.execute(
                    """
                    UPDATE paper_predictions
                    SET status = 'MATURED',
                        exit_timestamp = ?,
                        exit_price = ?,
                        actual_asset_ret_pct = ?,
                        direction_correct = ?,
                        gross_strategy_ret_pct = ?,
                        net_strategy_ret_pct = ?,
                        risk_managed_ret_pct = ?,
                        allocated_hold_pnl_pct = ?,
                        allocated_risk_pnl_pct = ?,
                        barrier_outcome = ?,
                        mfe_pct = ?,
                        mae_pct = ?,
                        evaluated_at_utc = ?
                    WHERE id = ?
                    """,
                    (
                        exit_ts,
                        round(exit_price, 6),
                        round(actual_asset_ret_pct, 4),
                        dir_correct,
                        round(gross_hold, 4),
                        round(net_hold, 4),
                        round(risk_managed_ret, 4),
                        round(alloc_hold_pnl, 4),
                        round(alloc_risk_pnl, 4),
                        barrier_outcome,
                        round(mfe_pct, 4),
                        round(mae_pct, 4),
                        eval_ts_str,
                        int(row["id"]),
                    ),
                )
                newly_matured += 1

        return EvaluationBatchResult(
            checked_open=len(open_rows),
            newly_matured=newly_matured,
            still_open=still_open,
            errors=errors,
        )

    def load_all_predictions(self) -> pd.DataFrame:
        """Load complete prediction ledger as DataFrame."""
        with self._connect() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM paper_predictions ORDER BY as_of_timestamp DESC, ticker ASC",
                conn,
            )
        return df

    def get_open_mtm(
        self,
        ohlcv_fetcher: Optional[Callable[[str, str, str], pd.DataFrame]] = None,
        now_utc: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """Compute live unrealized mark-to-market return and bars elapsed for OPEN predictions."""
        df = self.load_all_predictions()
        open_df = df[df["status"] == "OPEN"].copy()
        if open_df.empty:
            return pd.DataFrame()

        ref_now = now_utc or utc_now_naive()
        fetcher = ohlcv_fetcher or (
            lambda ticker, period, interval: fetch_ohlcv(
                ticker=ticker, period=period, interval=interval, use_cache=True
            )
        )

        rows: List[Dict[str, Any]] = []
        price_cache: Dict[Tuple[str, str], pd.DataFrame] = {}
        for _, r in open_df.iterrows():
            key = (r["ticker"], r["horizon_name"])
            if key not in price_cache:
                hcfg = HORIZON_CONFIGS[r["horizon_name"]]
                period = "15d" if hcfg.interval not in {"1d", "1wk", "1mo"} else "3mo"
                try:
                    price_cache[key] = fetcher(r["ticker"], period, hcfg.interval)
                except Exception:
                    price_cache[key] = pd.DataFrame()

            px_df = price_cache[key]
            as_of_ts = pd.to_datetime(r["as_of_timestamp"])
            if px_df.empty:
                latest_px = float(r["entry_price"])
                closed_elapsed = 0
            else:
                latest_px = float(px_df["Close"].iloc[-1])
                hcfg = HORIZON_CONFIGS[r["horizon_name"]]
                closed_df = filter_closed_bars(px_df, hcfg.interval, r["ticker"], now_utc=ref_now)
                closed_elapsed = int((closed_df.index > as_of_ts).sum())

            entry_px = float(r["entry_price"])
            direction = int(r["direction"])
            alloc_frac = abs(float(r["suggested_allocation_pct"])) / 100.0
            asset_mtm = ((latest_px / entry_px) - 1.0) * 100.0
            dir_mtm = direction * asset_mtm if direction != 0 else 0.0
            rows.append(
                {
                    "ticker": r["ticker"],
                    "horizon": r["horizon_name"],
                    "as_of": r["as_of_timestamp"],
                    "bars_elapsed": f"{closed_elapsed}/{int(r['forecast_horizon'])}",
                    "verdict": r["verdict"],
                    "entry_price": round(entry_px, 4),
                    "latest_price": round(latest_px, 4),
                    "expected_ret_pct": round(float(r["composite_expected_ret_pct"]), 2),
                    "unrealized_dir_pct": round(dir_mtm, 2),
                    "unrealized_alloc_pct": round(alloc_frac * dir_mtm, 2),
                    "stop_loss": round(float(r["stop_loss_price"]), 4),
                    "take_profit": round(float(r["take_profit_price"]), 4),
                }
            )
        return pd.DataFrame(rows)

    def get_summary(self) -> Dict[str, Any]:
        """Aggregate headline live validation metrics over OPEN and MATURED records."""
        df = self.load_all_predictions()
        if df.empty:
            return {
                "total_logged": 0,
                "open_count": 0,
                "matured_count": 0,
                "active_matured_count": 0,
                "hit_rate_pct": None,
                "avg_net_ret_pct": None,
                "avg_risk_ret_pct": None,
                "cum_alloc_hold_pnl_pct": 0.0,
                "cum_alloc_risk_pnl_pct": 0.0,
                "live_spearman_ic": None,
                "tp_count": 0,
                "sl_count": 0,
            }

        matured = df[df["status"] == "MATURED"].copy()
        active_matured = matured[matured["direction"] != 0].copy()

        hit_rate = (
            round(float(active_matured["direction_correct"].dropna().mean() * 100.0), 2)
            if not active_matured.empty and active_matured["direction_correct"].notna().any()
            else None
        )
        avg_net = round(float(active_matured["net_strategy_ret_pct"].mean()), 3) if not active_matured.empty else None
        avg_risk = round(float(active_matured["risk_managed_ret_pct"].mean()), 3) if not active_matured.empty else None

        live_ic = None
        if len(matured) >= 5 and matured["composite_expected_ret_pct"].std() > 1e-8:
            c, _ = spearmanr(matured["actual_asset_ret_pct"], matured["composite_expected_ret_pct"])
            if np.isfinite(c):
                live_ic = round(float(c), 4)

        return {
            "total_logged": int(len(df)),
            "open_count": int((df["status"] == "OPEN").sum()),
            "matured_count": int(len(matured)),
            "active_matured_count": int(len(active_matured)),
            "hit_rate_pct": hit_rate,
            "avg_net_ret_pct": avg_net,
            "avg_risk_ret_pct": avg_risk,
            "cum_alloc_hold_pnl_pct": round(float(matured["allocated_hold_pnl_pct"].fillna(0.0).sum()), 3) if not matured.empty else 0.0,
            "cum_alloc_risk_pnl_pct": round(float(matured["allocated_risk_pnl_pct"].fillna(0.0).sum()), 3) if not matured.empty else 0.0,
            "live_spearman_ic": live_ic,
            "tp_count": int((matured["barrier_outcome"] == "TAKE_PROFIT").sum()) if not matured.empty else 0,
            "sl_count": int((matured["barrier_outcome"] == "STOP_LOSS").sum()) if not matured.empty else 0,
        }

    def get_model_accuracy_leaderboard(self) -> pd.DataFrame:
        """Grade every individual model family on matured live predictions."""
        df = self.load_all_predictions()
        matured = df[df["status"] == "MATURED"]
        if matured.empty:
            return pd.DataFrame()

        per_model: Dict[str, List[Tuple[float, float, float]]] = {}
        for _, row in matured.iterrows():
            actual = float(row["actual_asset_ret_pct"])
            try:
                preds = json.loads(row["model_preds_json"])
            except Exception:
                continue
            for m in preds:
                name = str(m.get("model", m.get("model_key", "unknown")))
                exp_r = float(m.get("expected_ret_pct", 0.0))
                p_vol = float(m.get("pred_vol_norm", exp_r))
                p_up = float(m.get("prob_up_pct", 50.0))
                per_model.setdefault(name, []).append((exp_r, p_vol, p_up, actual))

        rows = []
        for name, items in per_model.items():
            arr = np.asarray(items, dtype=float)
            exp_arr, pvol_arr, p_arr, act_arr = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
            non_flat = np.abs(pvol_arr) > 1e-6
            hit = float(((pvol_arr[non_flat] > 0) == (act_arr[non_flat] > 0)).mean() * 100.0) if non_flat.any() else 50.0
            ic = 0.0
            if len(items) >= 5 and np.std(pvol_arr) > 1e-8 and np.std(act_arr) > 1e-8:
                c, _ = spearmanr(act_arr, pvol_arr)
                ic = float(c) if np.isfinite(c) else 0.0
            mae = float(np.mean(np.abs(exp_arr - act_arr)))
            rows.append(
                {
                    "model": name,
                    "matured_calls": len(items),
                    "live_hit_rate_pct": round(hit, 2),
                    "live_spearman_ic": round(ic, 4),
                    "return_mae_pct": round(mae, 2),
                    "mean_prob_up_pct": round(float(np.mean(p_arr)), 1),
                }
            )
        return pd.DataFrame(rows).sort_values("live_spearman_ic", ascending=False).reset_index(drop=True)


def log_paper_signals(
    tickers: List[str],
    horizons: List[str],
    universe_name: str = "stepping_stone",
    use_cache: bool = False,
    ledger: Optional[PaperLedger] = None,
) -> List[Dict[str, Any]]:
    """Generate closed-bar live predictions and record them idempotently to the paper ledger."""
    from quant_alpha.config import GLOBAL_DRIVERS, EngineSettings
    from quant_alpha.data.macro_feed import load_all_macro_series
    from quant_alpha.data.market_feed import fetch_universe_ohlcv
    from quant_alpha.data.news_feed import fetch_all_news
    from quant_alpha.features.sentiment import score_news_dataframe
    from quant_alpha.predictor import ensure_horizon_bundle, predict_asset

    ledger = ledger or PaperLedger()
    settings = EngineSettings()
    macro_df = load_all_macro_series(use_cache=use_cache, settings=settings)
    scored_news = score_news_dataframe(fetch_all_news(tickers, use_cache=use_cache, settings=settings))
    all_fetch = list(dict.fromkeys(list(tickers) + GLOBAL_DRIVERS))

    events: List[Dict[str, Any]] = []
    for h_name in horizons:
        hcfg = HORIZON_CONFIGS[h_name]
        bundle = ensure_horizon_bundle(h_name, universe_name=universe_name, settings=settings)
        ohlcv_map, _ = fetch_universe_ohlcv(
            all_fetch,
            period=hcfg.period,
            interval=hcfg.interval,
            use_cache=use_cache,
            settings=settings,
        )
        for t in tickers:
            try:
                report = predict_asset(
                    ticker=t,
                    horizon_name=h_name,
                    universe_name=universe_name,
                    settings=settings,
                    ohlcv_map=ohlcv_map,
                    macro_df=macro_df,
                    scored_news=scored_news,
                    bundle=bundle,
                    use_cache=use_cache,
                    closed_bars_only=True,
                )
                logged, msg = ledger.log_prediction(report, horizon_cfg=hcfg)
                events.append(
                    {
                        "ticker": t,
                        "horizon": h_name,
                        "as_of": report.as_of_timestamp,
                        "verdict": report.verdict,
                        "entry_price": report.last_close,
                        "alloc_pct": report.suggested_allocation_pct,
                        "logged": logged,
                        "message": msg,
                    }
                )
            except Exception as exc:
                events.append(
                    {
                        "ticker": t,
                        "horizon": h_name,
                        "as_of": "N/A",
                        "verdict": "ERROR",
                        "entry_price": None,
                        "alloc_pct": 0.0,
                        "logged": False,
                        "message": str(exc),
                    }
                )
    return events
