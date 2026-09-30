"""Realized variance, the variance risk premium, and a walk-forward HAR forecast.

Conventions (annualized, in variance units unless a name says vol points):
- r_t = ln(S_t / S_{t-1}), daily S&P 500 log returns;
- forward realized variance over the next h = 21 trading days (about the 30 calendar
  days VIX covers): RV_t = (252 / h) * sum_{i=1..h} r_{t+i}^2;
- implied variance IV_t = (VIX_t / 100)^2, the 30-day variance-swap rate VIX replicates;
- variance risk premium VRP_t = IV_t - RV_t (positive: implied was too high).

HAR (Corsi 2009): RV_{t->t+h} = a + b_d RV^d_t + b_w RV^w_t + b_m RV^m_t, with RV^d the
day's squared return, RV^w and RV^m trailing 5- and 22-day averages (all annualized).
A forecast made at t is fitted only on samples whose target window has ended by t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

H = 21


def log_returns(spx: pd.Series) -> pd.Series:
    return np.log(spx).diff()


def forward_rv(r: pd.Series, h: int = H) -> pd.Series:
    """RV_t over days t+1..t+h (NaN for the last h days)."""
    sq = (r ** 2).shift(-1).rolling(h).sum().shift(-(h - 1))
    return 252.0 / h * sq


def har_features(r: pd.Series) -> pd.DataFrame:
    sq = 252.0 * r ** 2
    return pd.DataFrame({"rv_d": sq, "rv_w": sq.rolling(5).mean(), "rv_m": sq.rolling(22).mean()})


def har_forecast(r: pd.Series, h: int = H, min_train: int = 750, refit_every: int = 21) -> pd.Series:
    """Walk-forward HAR forecast of RV_{t->t+h}.

    At each refit date t the regression uses only samples s with s + h <= t (their target
    window is over), so nothing from the future enters a forecast."""
    X = har_features(r)
    y = forward_rv(r, h)
    idx = X.index
    out = pd.Series(np.nan, index=idx)
    beta = None
    for i in range(len(idx)):
        if i >= min_train + h and (beta is None or i % refit_every == 0):
            train = slice(22, i - h + 1)  # targets ending by day i
            Xt, yt = X.iloc[train], y.iloc[train]
            ok = Xt.notna().all(axis=1) & yt.notna()
            A = np.column_stack([np.ones(ok.sum()), Xt[ok].to_numpy()])
            beta, *_ = np.linalg.lstsq(A, yt[ok].to_numpy(), rcond=None)
        if beta is not None and X.iloc[i].notna().all():
            out.iloc[i] = max(float(beta[0] + X.iloc[i].to_numpy() @ beta[1:]), 1e-6)
    return out


def frame(df: pd.DataFrame, h: int = H) -> pd.DataFrame:
    """Daily frame: implied and realized variance/vol, the premium, the HAR forecast."""
    r = log_returns(df["spx"])
    out = pd.DataFrame(index=df.index)
    out["r"] = r
    out["iv2"] = (df["vix"] / 100.0) ** 2
    out["rv"] = forward_rv(r, h)
    out["vrp"] = out["iv2"] - out["rv"]
    out["vix"] = df["vix"]
    out["vix3m"] = df["vix3m"]
    out["rv_vol"] = 100.0 * np.sqrt(out["rv"])
    out["har"] = har_forecast(r, h)
    return out
