"""Black-Scholes-Merton pricing with a continuous dividend yield q.

Black & Scholes (1973), Merton (1973). For spot S, strike K, time to expiry T
(years), risk-free rate r and dividend yield q (both continuously compounded),
and volatility sigma:

    d1 = [ln(S/K) + (r - q + sigma^2/2) T] / (sigma sqrt T),   d2 = d1 - sigma sqrt T
    C  = S e^{-qT} N(d1) - K e^{-rT} N(d2)
    P  = K e^{-rT} N(-d2) - S e^{-qT} N(-d1)

Greeks (phi is the standard normal density):

    delta_C = e^{-qT} N(d1)                 delta_P = -e^{-qT} N(-d1)
    gamma   = e^{-qT} phi(d1) / (S sigma sqrt T)            (same for C and P)
    vega    = S e^{-qT} phi(d1) sqrt T      per 1.00 of vol (divide by 100 per vol point)
    theta_C = -S e^{-qT} phi(d1) sigma / (2 sqrt T) - r K e^{-rT} N(d2) + q S e^{-qT} N(d1)
    theta_P = -S e^{-qT} phi(d1) sigma / (2 sqrt T) + r K e^{-rT} N(-d2) - q S e^{-qT} N(-d1)
              (dV/dt per year of calendar time; divide by 365 for one day)
    rho_C   = K T e^{-rT} N(d2)             rho_P = -K T e^{-rT} N(-d2)   per 1.00 of rate

No-arbitrage bounds (European):
    max(S e^{-qT} - K e^{-rT}, 0) <= C <= S e^{-qT}
    max(K e^{-rT} - S e^{-qT}, 0) <= P <= K e^{-rT}

Every function broadcasts over NumPy arrays; `kind` is "call"/"put" (or "C"/"P").
"""

from __future__ import annotations

import math
from typing import Union

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr

ArrayLike = Union[float, np.ndarray]
_CALLS = ("call", "c", "C")
_PUTS = ("put", "p", "P")


def _is_call(kind) -> np.ndarray:
    k = np.asarray(kind)
    call = np.isin(k, _CALLS)
    if not np.all(call | np.isin(k, _PUTS)):
        raise ValueError(f"kind must be one of {_CALLS + _PUTS}")
    return call


def _pdf(x: ArrayLike) -> ArrayLike:
    return np.exp(-0.5 * np.square(x)) / math.sqrt(2.0 * math.pi)


def d1_d2(S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, q: ArrayLike,
          sigma: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
    """The two BSM standardized moneyness terms (requires T > 0, sigma > 0)."""
    v = np.multiply(sigma, np.sqrt(T))
    d1 = (np.log(np.divide(S, K)) + (np.subtract(r, q) + 0.5 * np.square(sigma)) * T) / v
    return d1, d1 - v


def price(S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, q: ArrayLike,
          sigma: ArrayLike, kind="call") -> ArrayLike:
    """European call or put price. sigma*sqrt(T) = 0 returns the discounted intrinsic value."""
    call = _is_call(kind)
    fwd_s = np.multiply(S, np.exp(-np.multiply(q, T)))      # S e^{-qT}
    pv_k = np.multiply(K, np.exp(-np.multiply(r, T)))       # K e^{-rT}
    v = np.multiply(sigma, np.sqrt(T))
    with np.errstate(divide="ignore", invalid="ignore"):
        d1, d2 = d1_d2(S, K, T, r, q, sigma)
        c = fwd_s * ndtr(d1) - pv_k * ndtr(d2)
        p = pv_k * ndtr(-d2) - fwd_s * ndtr(-d1)
    out = np.where(call, c, p)
    intrinsic = np.where(call, np.maximum(fwd_s - pv_k, 0.0), np.maximum(pv_k - fwd_s, 0.0))
    out = np.where(np.asarray(v) > 0, out, intrinsic)
    return out[()] if np.ndim(out) == 0 else out


def call_price(S, K, T, r, q, sigma):
    return price(S, K, T, r, q, sigma, "call")


def put_price(S, K, T, r, q, sigma):
    return price(S, K, T, r, q, sigma, "put")


def delta(S, K, T, r, q, sigma, kind="call"):
    """dV/dS."""
    d1, _ = d1_d2(S, K, T, r, q, sigma)
    dq = np.exp(-np.multiply(q, T))
    out = np.where(_is_call(kind), dq * ndtr(d1), -dq * ndtr(-d1))
    return out[()] if np.ndim(out) == 0 else out


def gamma(S, K, T, r, q, sigma):
    """d2V/dS2 (identical for calls and puts)."""
    d1, _ = d1_d2(S, K, T, r, q, sigma)
    return np.exp(-np.multiply(q, T)) * _pdf(d1) / (np.multiply(S, sigma) * np.sqrt(T))


def vega(S, K, T, r, q, sigma):
    """dV/dsigma per 1.00 of volatility (identical for calls and puts)."""
    d1, _ = d1_d2(S, K, T, r, q, sigma)
    return np.multiply(S, np.exp(-np.multiply(q, T))) * _pdf(d1) * np.sqrt(T)


def theta(S, K, T, r, q, sigma, kind="call"):
    """dV/dt per year of calendar time (= -dV/dT); negative for a long vanilla option
    except deep in-the-money puts (and calls when q is large)."""
    d1, d2 = d1_d2(S, K, T, r, q, sigma)
    fwd_s = np.multiply(S, np.exp(-np.multiply(q, T)))
    pv_k = np.multiply(K, np.exp(-np.multiply(r, T)))
    decay = -fwd_s * _pdf(d1) * np.multiply(sigma, 0.5) / np.sqrt(T)
    c = decay - np.multiply(r, pv_k) * ndtr(d2) + np.multiply(q, fwd_s) * ndtr(d1)
    p = decay + np.multiply(r, pv_k) * ndtr(-d2) - np.multiply(q, fwd_s) * ndtr(-d1)
    out = np.where(_is_call(kind), c, p)
    return out[()] if np.ndim(out) == 0 else out


def rho(S, K, T, r, q, sigma, kind="call"):
    """dV/dr per 1.00 of rate."""
    _, d2 = d1_d2(S, K, T, r, q, sigma)
    kt = np.multiply(K, T) * np.exp(-np.multiply(r, T))
    out = np.where(_is_call(kind), kt * ndtr(d2), -kt * ndtr(-d2))
    return out[()] if np.ndim(out) == 0 else out


def greeks(S, K, T, r, q, sigma, kind="call") -> dict:
    """All five Greeks plus the price."""
    return {"price": price(S, K, T, r, q, sigma, kind), "delta": delta(S, K, T, r, q, sigma, kind),
            "gamma": gamma(S, K, T, r, q, sigma), "vega": vega(S, K, T, r, q, sigma),
            "theta": theta(S, K, T, r, q, sigma, kind), "rho": rho(S, K, T, r, q, sigma, kind)}


def bounds(S, K, T, r, q, kind="call") -> tuple[ArrayLike, ArrayLike]:
    """European no-arbitrage (lower, upper) bounds on the option price."""
    fwd_s = np.multiply(S, np.exp(-np.multiply(q, T)))
    pv_k = np.multiply(K, np.exp(-np.multiply(r, T)))
    call = _is_call(kind)
    lower = np.where(call, np.maximum(fwd_s - pv_k, 0.0), np.maximum(pv_k - fwd_s, 0.0))
    upper = np.where(call, fwd_s, pv_k)
    return lower, upper


def implied_vol(target: float, S: float, K: float, T: float, r: float, q: float,
                kind: str = "call", lo: float = 1e-6, hi: float = 10.0, xtol: float = 1e-12) -> float:
    """The sigma in [lo, hi] with price(sigma) = target, by Brent's method (Brent 1973).

    The BSM price is strictly increasing in sigma (vega > 0), rising from the discounted
    intrinsic value at sigma -> 0 to S e^{-qT} (call) or K e^{-rT} (put) as sigma -> inf.
    If `target` is not strictly inside [price(lo), price(hi)], for example a price below
    intrinsic value, no volatility reproduces it and the result is NaN.
    """
    if not (np.isfinite(target) and T > 0 and S > 0 and K > 0):
        return float("nan")

    def f(sig: float) -> float:
        return float(price(S, K, T, r, q, sig, kind)) - target

    f_lo, f_hi = f(lo), f(hi)
    if not (f_lo < 0.0 < f_hi):
        return float("nan")
    return float(brentq(f, lo, hi, xtol=xtol, rtol=4 * np.finfo(float).eps, maxiter=200))


def implied_vol_array(targets, S, K, T, r, q, kinds, **kw) -> np.ndarray:
    """Elementwise `implied_vol` over broadcast arrays."""
    b = np.broadcast_arrays(np.asarray(targets, float), np.asarray(S, float), np.asarray(K, float),
                            np.asarray(T, float), np.asarray(r, float), np.asarray(q, float),
                            np.asarray(kinds))
    out = np.full(b[0].shape, np.nan)
    for idx in np.ndindex(out.shape):
        t, s, k, tt, rr, qq, kd = (x[idx] for x in b)
        out[idx] = implied_vol(float(t), float(s), float(k), float(tt), float(rr), float(qq), str(kd), **kw)
    return out
