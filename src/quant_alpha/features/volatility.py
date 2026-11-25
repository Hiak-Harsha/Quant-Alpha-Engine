"""Causal realized, range-based (Parkinson, Garman-Klass), and rolling GARCH(1,1) volatility features."""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _rolling_causal_garch(
    returns: pd.Series,
    bars_per_year: int = 252,
    refit_every: int = 21,
    min_obs: int = 120,
) -> pd.Series:
    """Estimate strictly causal 1-step-ahead GARCH(1,1) annualized volatility.

    Refits GARCH(1,1) periodically on expanding history up to t and propagates
    conditional variance causally between refits, falling back cleanly to EWMA
    when history is short or optimizer fails.
    """
    ewma_vol = returns.ewm(span=21, min_periods=10).std() * np.sqrt(bars_per_year)
    if len(returns.dropna()) < min_obs:
        return ewma_vol

    try:
        from arch import arch_model
    except ImportError:
        return ewma_vol

    r_clean = returns.fillna(0.0).astype(float)
    r_pct = r_clean * 100.0
    n = len(r_pct)
    garch_ann = pd.Series(np.nan, index=returns.index, dtype=float)

    omega = 0.05
    alpha = 0.08
    beta = 0.88
    sigma2 = float(r_pct.iloc[:min_obs].var()) if r_pct.iloc[:min_obs].var() > 1e-8 else 1.0

    for i in range(min_obs, n):
        if (i - min_obs) % refit_every == 0:
            hist = r_pct.iloc[:i]
            if hist.std() > 1e-6:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        am = arch_model(hist, vol="Garch", p=1, q=1, mean="Zero", rescale=False)
                        res = am.fit(disp="off", show_warning=False)
                        params = res.params
                        o = float(params.get("omega", omega))
                        a = float(params.get("alpha[1]", alpha))
                        b = float(params.get("beta[1]", beta))
                        if o > 0 and a >= 0 and b >= 0 and (a + b) < 0.995:
                            omega, alpha, beta = o, a, b
                            cond_vol = res.conditional_volatility
                            if len(cond_vol) > 0 and np.isfinite(cond_vol.iloc[-1]):
                                sigma2 = float(cond_vol.iloc[-1] ** 2)
                except Exception:
                    pass

        r_prev = float(r_pct.iloc[i - 1])
        sigma2 = max(1e-8, omega + alpha * (r_prev ** 2) + beta * sigma2)
        daily_vol = np.sqrt(sigma2) / 100.0
        garch_ann.iloc[i] = daily_vol * np.sqrt(bars_per_year)

    return garch_ann.fillna(ewma_vol)


def compute_volatility_features(df: pd.DataFrame, bars_per_year: int = 252) -> pd.DataFrame:
    """Compute realized, Parkinson, Garman-Klass, downside, vol-of-vol, and GARCH(1,1) volatility."""
    if df is None or df.empty:
        return pd.DataFrame()

    open_p = df["Open"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)

    log_ret = np.log(close / close.shift(1).replace(0.0, np.nan))
    feat = pd.DataFrame(index=df.index)

    # Close-to-close realized volatility (annualized)
    for w in (5, 10, 21, 63):
        feat[f"rv_{w}"] = log_ret.rolling(w, min_periods=max(3, w // 2)).std() * np.sqrt(bars_per_year)

    # Parkinson range-based volatility
    log_hl = np.log((high / low.replace(0.0, np.nan)).clip(lower=1e-9))
    parkinson_var = (log_hl ** 2) / (4.0 * np.log(2.0))
    for w in (10, 21):
        feat[f"parkinson_vol_{w}"] = np.sqrt(
            parkinson_var.rolling(w, min_periods=5).mean().clip(lower=0.0) * bars_per_year
        )

    # Garman-Klass OHLC volatility
    log_co = np.log((close / open_p.replace(0.0, np.nan)).clip(lower=1e-9))
    gk_var = 0.5 * (log_hl ** 2) - (2.0 * np.log(2.0) - 1.0) * (log_co ** 2)
    for w in (10, 21):
        feat[f"gk_vol_{w}"] = np.sqrt(
            gk_var.rolling(w, min_periods=5).mean().clip(lower=0.0) * bars_per_year
        )

    # Downside vs upside semivolatility asymmetry
    neg_ret = log_ret.where(log_ret < 0.0, 0.0)
    pos_ret = log_ret.where(log_ret > 0.0, 0.0)
    down_vol = np.sqrt((neg_ret ** 2).rolling(21, min_periods=8).mean() * bars_per_year)
    up_vol = np.sqrt((pos_ret ** 2).rolling(21, min_periods=8).mean() * bars_per_year)
    feat["downside_vol_21"] = down_vol
    feat["vol_asymmetry_21"] = down_vol / up_vol.replace(0.0, np.nan)

    # Volatility term structure & vol-of-vol
    feat["vol_ratio_5_21"] = feat["rv_5"] / feat["rv_21"].replace(0.0, np.nan)
    feat["vol_ratio_21_63"] = feat["rv_21"] / feat["rv_63"].replace(0.0, np.nan)
    feat["vol_of_vol_21"] = feat["rv_5"].rolling(21, min_periods=8).std()
    feat["vol_zscore_63"] = (
        feat["rv_21"] - feat["rv_21"].rolling(63, min_periods=20).mean()
    ) / feat["rv_21"].rolling(63, min_periods=20).std().replace(0.0, np.nan)

    # Causal GARCH(1,1) conditional volatility forecast
    feat["garch_vol_forecast"] = _rolling_causal_garch(log_ret, bars_per_year=bars_per_year)
    feat["garch_vs_rv21"] = feat["garch_vol_forecast"] / feat["rv_21"].replace(0.0, np.nan)

    return feat.replace([np.inf, -np.inf], np.nan)
