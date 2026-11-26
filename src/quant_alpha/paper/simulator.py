"""Broker-grade paper trading simulator with real Indian delivery costs, holdings, and accuracy correlation."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from quant_alpha.config import is_crypto_ticker
from quant_alpha.paper.costs import DEFAULT_FEE_SCHEDULE, compute_order_fees, estimate_round_trip_breakeven_pct
from quant_alpha.paper.ledger import DEFAULT_LEDGER_PATH, PaperLedger
from quant_alpha.portfolio.construct import is_cash_long_only_asset


@dataclass
class OrderPreview:
    """Pre-trade execution check, cash check, and exact itemized fee breakdown."""

    valid: bool
    reason: str
    ticker: str
    side: str
    qty: float
    price: float
    notional: float
    fees: Dict[str, float]
    net_cash_delta: float
    breakeven_move_pct: float


class PaperBrokerSimulator:
    """Multi-account virtual broker simulating cash delivery execution and real market fees."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else DEFAULT_LEDGER_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
        self.ensure_account("default_student", "Student Research Desk", starting_cash=100000.0)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=15)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sim_accounts (
                    account_id TEXT PRIMARY KEY,
                    owner_name TEXT NOT NULL,
                    starting_cash REAL NOT NULL,
                    cash_balance REAL NOT NULL,
                    created_at_utc TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sim_positions (
                    account_id TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    qty REAL NOT NULL,
                    avg_entry_price REAL NOT NULL,
                    total_fees_paid REAL NOT NULL DEFAULT 0.0,
                    stop_loss REAL NOT NULL DEFAULT 0.0,
                    take_profit REAL NOT NULL DEFAULT 0.0,
                    opened_at_utc TEXT NOT NULL,
                    signal_verdict TEXT NOT NULL DEFAULT '',
                    signal_ret_pct REAL NOT NULL DEFAULT 0.0,
                    PRIMARY KEY (account_id, ticker)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sim_orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_id TEXT NOT NULL,
                    timestamp_utc TEXT NOT NULL,
                    bar_date TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    side TEXT NOT NULL,
                    qty REAL NOT NULL,
                    fill_price REAL NOT NULL,
                    notional REAL NOT NULL,
                    brokerage REAL NOT NULL,
                    stt REAL NOT NULL,
                    stamp_duty REAL NOT NULL,
                    exchange_sebi REAL NOT NULL,
                    gst REAL NOT NULL,
                    dp_charge REAL NOT NULL,
                    total_fee REAL NOT NULL,
                    realized_pnl REAL NOT NULL DEFAULT 0.0,
                    signal_verdict TEXT NOT NULL DEFAULT '',
                    notes TEXT NOT NULL DEFAULT ''
                )
                """
            )

    def ensure_account(self, account_id: str, owner_name: str, starting_cash: float = 100000.0) -> None:
        """Create a simulated portfolio account if absent."""
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO sim_accounts (account_id, owner_name, starting_cash, cash_balance, created_at_utc)
                VALUES (?, ?, ?, ?, ?)
                """,
                (account_id.strip(), owner_name.strip(), float(starting_cash), float(starting_cash), now_str),
            )

    def reset_account(self, account_id: str, starting_cash: float = 100000.0) -> None:
        """Reset an account's cash, open positions, and orders back to initial capital."""
        with self._connect() as conn:
            conn.execute("DELETE FROM sim_positions WHERE account_id = ?", (account_id,))
            conn.execute("DELETE FROM sim_orders WHERE account_id = ?", (account_id,))
            conn.execute(
                "UPDATE sim_accounts SET starting_cash = ?, cash_balance = ? WHERE account_id = ?",
                (float(starting_cash), float(starting_cash), account_id),
            )

    def list_accounts(self) -> List[Dict[str, Any]]:
        """List all virtual trading accounts."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM sim_accounts ORDER BY created_at_utc ASC").fetchall()
        return [dict(r) for r in rows]

    def get_account(self, account_id: str) -> Dict[str, Any]:
        """Fetch account balance and metadata."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM sim_accounts WHERE account_id = ?", (account_id,)).fetchone()
        if row is None:
            raise ValueError(f"Simulated account '{account_id}' does not exist.")
        return dict(row)

    def preview_order(
        self,
        account_id: str,
        ticker: str,
        side: str,
        qty: float,
        price: float,
    ) -> OrderPreview:
        """Validate execution rules (cash long-only, integer shares, available balance) & compute fees."""
        acc = self.get_account(account_id)
        side_u = side.upper()
        t_clean = ticker.strip().upper()
        p = float(price)
        q = float(qty)

        if side_u not in {"BUY", "SELL"}:
            return OrderPreview(False, "Side must be BUY or SELL.", t_clean, side_u, q, p, 0.0, {}, 0.0, 0.0)
        if p <= 0.0 or q <= 0.0:
            return OrderPreview(False, "Quantity and price must be positive.", t_clean, side_u, q, p, 0.0, {}, 0.0, 0.0)
        if not is_crypto_ticker(t_clean) and abs(q - round(q)) > 1e-6:
            return OrderPreview(
                False,
                "NSE/ETF cash delivery requires whole shares/units (fractional units only for crypto).",
                t_clean,
                side_u,
                q,
                p,
                0.0,
                {},
                0.0,
                0.0,
            )

        notional = round(q * p, 2)
        fees = compute_order_fees(t_clean, side_u, notional, DEFAULT_FEE_SCHEDULE)
        breakeven = estimate_round_trip_breakeven_pct(t_clean, notional, DEFAULT_FEE_SCHEDULE)

        if side_u == "BUY":
            total_debit = notional + fees["total_fee"]
            if total_debit > float(acc["cash_balance"]) + 1e-6:
                return OrderPreview(
                    False,
                    f"Insufficient cash: need {total_debit:,.2f}, available {acc['cash_balance']:,.2f}.",
                    t_clean,
                    side_u,
                    q,
                    p,
                    notional,
                    fees,
                    -total_debit,
                    breakeven,
                )
            return OrderPreview(True, "Ready to BUY", t_clean, side_u, q, p, notional, fees, -total_debit, breakeven)

        # SELL validation (enforce cash delivery long-only when shorting not held)
        with self._connect() as conn:
            pos = conn.execute(
                "SELECT * FROM sim_positions WHERE account_id = ? AND ticker = ?",
                (account_id, t_clean),
            ).fetchone()
        held_qty = float(pos["qty"]) if pos is not None else 0.0
        if is_cash_long_only_asset(t_clean) and q > held_qty + 1e-6:
            return OrderPreview(
                False,
                f"Cash delivery SELL capped at held quantity ({held_qty:g} units held). Delivery shorting is disabled.",
                t_clean,
                side_u,
                q,
                p,
                notional,
                fees,
                0.0,
                breakeven,
            )
        if q > held_qty + 1e-6:
            return OrderPreview(
                False,
                f"Cannot sell {q:g} units; only {held_qty:g} held in this cash simulator.",
                t_clean,
                side_u,
                q,
                p,
                notional,
                fees,
                0.0,
                breakeven,
            )

        net_credit = notional - fees["total_fee"]
        return OrderPreview(True, "Ready to SELL", t_clean, side_u, q, p, notional, fees, net_credit, breakeven)

    def execute_order(
        self,
        account_id: str,
        ticker: str,
        side: str,
        qty: float,
        fill_price: float,
        bar_date: str = "",
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
        signal_verdict: str = "",
        signal_ret_pct: float = 0.0,
        notes: str = "",
    ) -> Tuple[bool, str]:
        """Execute a validated virtual order with exact broker fees & realized P&L accounting."""
        prev = self.preview_order(account_id, ticker, side, qty, fill_price)
        if not prev.valid:
            return False, prev.reason

        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        b_date = bar_date or now_str[:10]
        f = prev.fees

        with self._connect() as conn:
            acc = conn.execute("SELECT cash_balance FROM sim_accounts WHERE account_id = ?", (account_id,)).fetchone()
            cash_now = float(acc["cash_balance"])
            pos = conn.execute(
                "SELECT * FROM sim_positions WHERE account_id = ? AND ticker = ?",
                (account_id, prev.ticker),
            ).fetchone()

            realized_pnl = 0.0
            if prev.side == "BUY":
                new_cash = cash_now + prev.net_cash_delta
                if pos is None:
                    conn.execute(
                        """
                        INSERT INTO sim_positions (
                            account_id, ticker, qty, avg_entry_price, total_fees_paid,
                            stop_loss, take_profit, opened_at_utc, signal_verdict, signal_ret_pct
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            account_id,
                            prev.ticker,
                            prev.qty,
                            prev.price,
                            f["total_fee"],
                            float(stop_loss),
                            float(take_profit),
                            now_str,
                            signal_verdict,
                            float(signal_ret_pct),
                        ),
                    )
                else:
                    old_q = float(pos["qty"])
                    old_p = float(pos["avg_entry_price"])
                    new_q = old_q + prev.qty
                    new_avg = ((old_q * old_p) + (prev.qty * prev.price)) / new_q
                    conn.execute(
                        """
                        UPDATE sim_positions
                        SET qty = ?, avg_entry_price = ?, total_fees_paid = total_fees_paid + ?,
                            stop_loss = ?, take_profit = ?
                        WHERE account_id = ? AND ticker = ?
                        """,
                        (
                            new_q,
                            new_avg,
                            f["total_fee"],
                            float(stop_loss or pos["stop_loss"]),
                            float(take_profit or pos["take_profit"]),
                            account_id,
                            prev.ticker,
                        ),
                    )
            else:
                new_cash = cash_now + prev.net_cash_delta
                old_q = float(pos["qty"])
                avg_entry = float(pos["avg_entry_price"])
                entry_fee_share = float(pos["total_fees_paid"]) * (prev.qty / max(old_q, 1e-8))
                realized_pnl = round(((prev.price - avg_entry) * prev.qty) - entry_fee_share - f["total_fee"], 2)
                rem_q = old_q - prev.qty
                if rem_q <= 1e-6:
                    conn.execute("DELETE FROM sim_positions WHERE account_id = ? AND ticker = ?", (account_id, prev.ticker))
                else:
                    conn.execute(
                        "UPDATE sim_positions SET qty = ?, total_fees_paid = total_fees_paid - ? WHERE account_id = ? AND ticker = ?",
                        (rem_q, entry_fee_share, account_id, prev.ticker),
                    )

            conn.execute("UPDATE sim_accounts SET cash_balance = ? WHERE account_id = ?", (round(new_cash, 2), account_id))
            conn.execute(
                """
                INSERT INTO sim_orders (
                    account_id, timestamp_utc, bar_date, ticker, side, qty, fill_price, notional,
                    brokerage, stt, stamp_duty, exchange_sebi, gst, dp_charge, total_fee,
                    realized_pnl, signal_verdict, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    account_id,
                    now_str,
                    b_date,
                    prev.ticker,
                    prev.side,
                    prev.qty,
                    prev.price,
                    prev.notional,
                    f["brokerage"],
                    f["stt"],
                    f["stamp_duty"],
                    f["exchange_sebi"],
                    f["gst"],
                    f["dp_charge"],
                    f["total_fee"],
                    realized_pnl,
                    signal_verdict,
                    notes,
                ),
            )

        return True, f"Executed {prev.side} {prev.qty:g} {prev.ticker} @ {prev.price:,.2f} (Fee: {f['total_fee']:.2f})"

    def get_portfolio_snapshot(
        self,
        account_id: str,
        price_map: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Compute live mark-to-market equity, unrealized/realized P&L, holdings, and orders."""
        acc = self.get_account(account_id)
        p_map = price_map or {}
        with self._connect() as conn:
            pos_rows = conn.execute(
                "SELECT * FROM sim_positions WHERE account_id = ? ORDER BY opened_at_utc DESC",
                (account_id,),
            ).fetchall()
            ord_rows = conn.execute(
                "SELECT * FROM sim_orders WHERE account_id = ? ORDER BY id DESC LIMIT 100",
                (account_id,),
            ).fetchall()

        pos_list = []
        market_val_sum = 0.0
        invested_cost_sum = 0.0
        for r in pos_rows:
            t = r["ticker"]
            qty = float(r["qty"])
            avg_p = float(r["avg_entry_price"])
            ltp = float(p_map.get(t, avg_p))
            cost = qty * avg_p
            mkt_val = qty * ltp
            unr_pnl = mkt_val - cost
            unr_pct = ((ltp / avg_p) - 1.0) * 100.0 if avg_p > 0 else 0.0
            market_val_sum += mkt_val
            invested_cost_sum += cost
            pos_list.append(
                {
                    "Ticker": t,
                    "Qty": qty,
                    "Avg Buy": round(avg_p, 2),
                    "Live Price": round(ltp, 2),
                    "Invested": round(cost, 2),
                    "Market Value": round(mkt_val, 2),
                    "Unrealized P&L": round(unr_pnl, 2),
                    "Return %": round(unr_pct, 2),
                    "Stop Loss": round(float(r["stop_loss"]), 2),
                    "Take Profit": round(float(r["take_profit"]), 2),
                    "Entry View": r["signal_verdict"],
                }
            )

        orders_df = pd.DataFrame([dict(r) for r in ord_rows]) if ord_rows else pd.DataFrame()
        realized_pnl = float(orders_df["realized_pnl"].sum()) if not orders_df.empty else 0.0
        total_fees = float(orders_df["total_fee"].sum()) if not orders_df.empty else 0.0
        cash = float(acc["cash_balance"])
        start_cash = float(acc["starting_cash"])
        total_equity = cash + market_val_sum
        net_return_pct = ((total_equity / max(start_cash, 1.0)) - 1.0) * 100.0

        return {
            "account": acc,
            "cash_balance": round(cash, 2),
            "invested_cost": round(invested_cost_sum, 2),
            "market_value": round(market_val_sum, 2),
            "total_equity": round(total_equity, 2),
            "unrealized_pnl": round(market_val_sum - invested_cost_sum, 2),
            "realized_pnl": round(realized_pnl, 2),
            "total_fees_paid": round(total_fees, 2),
            "net_return_pct": round(net_return_pct, 2),
            "positions_df": pd.DataFrame(pos_list),
            "orders_df": orders_df,
        }


def compute_prediction_market_correlation(ledger: PaperLedger) -> Dict[str, Any]:
    """Measure live correlation between closed-bar predicted returns and actual realized moves."""
    all_df = ledger.load_all_predictions()
    if all_df.empty:
        return {"n_matured": 0, "pearson_r": None, "spearman_ic": None, "hit_rate_pct": None, "always_long_pct": None, "df": pd.DataFrame()}

    matured = all_df[all_df["status"] == "MATURED"].copy()
    if matured.empty:
        return {"n_matured": 0, "pearson_r": None, "spearman_ic": None, "hit_rate_pct": None, "always_long_pct": None, "df": pd.DataFrame()}

    pred = pd.to_numeric(matured["composite_expected_ret_pct"], errors="coerce")
    act = pd.to_numeric(matured["actual_asset_ret_pct"], errors="coerce")
    valid = matured.loc[pred.notna() & act.notna()].copy()
    if len(valid) < 2:
        return {"n_matured": len(valid), "pearson_r": None, "spearman_ic": None, "hit_rate_pct": None, "always_long_pct": None, "df": valid}

    p_arr = pd.to_numeric(valid["composite_expected_ret_pct"]).to_numpy(dtype=float)
    a_arr = pd.to_numeric(valid["actual_asset_ret_pct"]).to_numpy(dtype=float)
    pearson_val = float(pearsonr(p_arr, a_arr)[0]) if np.std(p_arr) > 1e-9 and np.std(a_arr) > 1e-9 else 0.0
    spearman_val = float(spearmanr(p_arr, a_arr)[0]) if np.std(p_arr) > 1e-9 and np.std(a_arr) > 1e-9 else 0.0

    non_flat = np.abs(p_arr) > 1e-6
    hit = float(((p_arr[non_flat] > 0) == (a_arr[non_flat] > 0)).mean() * 100.0) if non_flat.any() else 50.0
    always_long = float((a_arr > 0).mean() * 100.0)

    return {
        "n_matured": len(valid),
        "pearson_r": round(pearson_val if np.isfinite(pearson_val) else 0.0, 4),
        "spearman_ic": round(spearman_val if np.isfinite(spearman_val) else 0.0, 4),
        "hit_rate_pct": round(hit, 2),
        "always_long_pct": round(always_long, 2),
        "df": valid,
    }
