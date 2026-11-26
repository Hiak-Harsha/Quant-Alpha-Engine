"""Real-world Indian equity/ETF delivery, US/global, and crypto transaction cost calculator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from quant_alpha.config import is_crypto_ticker

_INDIA_ETFS = {
    "GOLDBEES.NS",
    "HDFCGOLD.NS",
    "GOLDCASE.NS",
    "SILVERBEES.NS",
    "SILVER.NS",
    "SILVERIETF.NS",
    "HDFCSILVER.NS",
    "NIFTYBEES.NS",
    "BANKBEES.NS",
    "JUNIORBEES.NS",
}


@dataclass(frozen=True)
class FeeSchedule:
    """Fee parameters calibrated to real Indian discount broker delivery & exchange tariffs."""

    india_brokerage_rate: float = 0.0         # Zero brokerage equity/ETF delivery
    india_equity_stt_rate: float = 0.001      # 0.10% STT on buy & sell for cash equities
    india_etf_sell_stt_rate: float = 0.00001  # 0.001% STT on sell for ETFs
    india_stamp_duty_buy: float = 0.00015     # 0.015% stamp duty on delivery buy
    india_exchange_txn_rate: float = 0.0000297  # 0.00297% NSE CM transaction fee
    india_sebi_rate: float = 0.000001         # Rs 10 / crore SEBI turnover fee
    india_gst_rate: float = 0.18              # 18% GST on brokerage + txn + SEBI
    india_dp_sell_charge: float = 15.34       # Rs 13 + 18% GST fixed per delivery SELL execution
    crypto_fee_rate: float = 0.001            # 0.10% spot exchange maker/taker fee
    us_global_fee_rate: float = 0.0005        # 5 bps synthetic US/global spread + fee proxy


DEFAULT_FEE_SCHEDULE = FeeSchedule()


def classify_fee_instrument(ticker: str) -> str:
    """Classify ticker into INDIA_ETF, INDIA_EQUITY, CRYPTO, or US_GLOBAL."""
    up = ticker.upper()
    if is_crypto_ticker(up):
        return "CRYPTO"
    if up in _INDIA_ETFS:
        return "INDIA_ETF"
    if up.endswith((".NS", ".BO")) or up in {"^NSEI", "^NSEBANK", "^CNXIT"}:
        return "INDIA_EQUITY"
    return "US_GLOBAL"


def compute_order_fees(
    ticker: str,
    side: str,
    notional_value: float,
    schedule: FeeSchedule = DEFAULT_FEE_SCHEDULE,
) -> Dict[str, float]:
    """Return exact itemized brokerage, STT, stamp duty, exchange/SEBI, GST, DP, and total fees."""
    side_u = side.upper()
    val = max(0.0, float(notional_value))
    inst = classify_fee_instrument(ticker)

    if val <= 0.0:
        return {
            "instrument_type": 0.0,
            "brokerage": 0.0,
            "stt": 0.0,
            "stamp_duty": 0.0,
            "exchange_sebi": 0.0,
            "gst": 0.0,
            "dp_charge": 0.0,
            "total_fee": 0.0,
            "fee_bps": 0.0,
        }

    if inst in {"INDIA_EQUITY", "INDIA_ETF"}:
        brokerage = val * schedule.india_brokerage_rate
        if inst == "INDIA_EQUITY":
            stt = round(val * schedule.india_equity_stt_rate, 2)
        else:
            stt = round(val * schedule.india_etf_sell_stt_rate, 2) if side_u == "SELL" else 0.0
        stamp = val * schedule.india_stamp_duty_buy if side_u == "BUY" else 0.0
        exch_sebi = val * (schedule.india_exchange_txn_rate + schedule.india_sebi_rate)
        gst = (brokerage + exch_sebi) * schedule.india_gst_rate
        dp = schedule.india_dp_sell_charge if side_u == "SELL" else 0.0
        total = brokerage + stt + stamp + exch_sebi + gst + dp
    elif inst == "CRYPTO":
        brokerage = val * schedule.crypto_fee_rate
        stt = 0.0
        stamp = 0.0
        exch_sebi = 0.0
        gst = 0.0
        dp = 0.0
        total = brokerage
    else:
        brokerage = val * schedule.us_global_fee_rate
        stt = 0.0
        stamp = 0.0
        exch_sebi = 0.0
        gst = 0.0
        dp = 0.0
        total = brokerage

    return {
        "brokerage": round(brokerage, 2),
        "stt": round(stt, 2),
        "stamp_duty": round(stamp, 2),
        "exchange_sebi": round(exch_sebi, 2),
        "gst": round(gst, 2),
        "dp_charge": round(dp, 2),
        "total_fee": round(total, 2),
        "fee_bps": round((total / val) * 10000.0, 2),
    }


def estimate_round_trip_breakeven_pct(
    ticker: str,
    notional_value: float,
    schedule: FeeSchedule = DEFAULT_FEE_SCHEDULE,
) -> float:
    """Compute exact minimum price gain (%) required to break even after BUY + SELL costs."""
    val = max(1.0, float(notional_value))
    buy_fee = compute_order_fees(ticker, "BUY", val, schedule)["total_fee"]
    sell_fee = compute_order_fees(ticker, "SELL", val, schedule)["total_fee"]
    return round(((buy_fee + sell_fee) / val) * 100.0, 3)
