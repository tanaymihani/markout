"""Black-Scholes-Merton: parity, Greeks vs finite differences, IV round-trip, bounds."""

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from markout.options import bs

spot = st.floats(10.0, 500.0)
strike = st.floats(10.0, 500.0)
expiry = st.floats(0.01, 3.0)
rate = st.floats(-0.01, 0.10)
yield_ = st.floats(0.0, 0.08)
vol = st.floats(0.05, 1.5)


def test_textbook_value():
    # Hull, Options, Futures and Other Derivatives, Example 15.6: c = 4.76, p = 0.81
    c = bs.call_price(42.0, 40.0, 0.5, 0.10, 0.0, 0.20)
    p = bs.put_price(42.0, 40.0, 0.5, 0.10, 0.0, 0.20)
    assert c == pytest.approx(4.7594, abs=1e-4)
    assert p == pytest.approx(0.8086, abs=1e-4)


@settings(max_examples=300, deadline=None)
@given(S=spot, K=strike, T=expiry, r=rate, q=yield_, sigma=vol)
def test_put_call_parity(S, K, T, r, q, sigma):
    lhs = bs.call_price(S, K, T, r, q, sigma) - bs.put_price(S, K, T, r, q, sigma)
    rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
    assert lhs == pytest.approx(rhs, abs=1e-9 * max(S, K))


@settings(max_examples=300, deadline=None)
@given(S=spot, K=strike, T=expiry, r=rate, q=yield_, sigma=vol)
def test_prices_respect_no_arbitrage_bounds(S, K, T, r, q, sigma):
    for kind in ("call", "put"):
        lo, hi = bs.bounds(S, K, T, r, q, kind)
        v = bs.price(S, K, T, r, q, sigma, kind)
        tol = 1e-9 * max(S, K)
        assert lo - tol <= v <= hi + tol


CASES = [  # (S, K, T, r, q, sigma): ATM, ITM, OTM, short and long dated, with carry
    (100.0, 100.0, 0.25, 0.03, 0.01, 0.20),
    (100.0, 80.0, 1.00, 0.05, 0.00, 0.30),
    (100.0, 125.0, 0.10, 0.02, 0.02, 0.45),
    (764.0, 700.0, 0.08, 0.041, 0.012, 0.15),
    (50.0, 55.0, 2.00, 0.00, 0.03, 0.60),
]


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("S,K,T,r,q,sigma", CASES)
def test_greeks_match_finite_differences(S, K, T, r, q, sigma, kind):
    V = lambda **kw: bs.price(**{**dict(S=S, K=K, T=T, r=r, q=q, sigma=sigma), **kw}, kind=kind)  # noqa: E731
    hS, hv, hT, hr = 1e-4 * S, 1e-5, 1e-6, 1e-6
    fd = {
        "delta": (V(S=S + hS) - V(S=S - hS)) / (2 * hS),
        "gamma": (V(S=S + hS) - 2 * V() + V(S=S - hS)) / hS**2,
        "vega": (V(sigma=sigma + hv) - V(sigma=sigma - hv)) / (2 * hv),
        "theta": -(V(T=T + hT) - V(T=T - hT)) / (2 * hT),   # dV/dt = -dV/dT
        "rho": (V(r=r + hr) - V(r=r - hr)) / (2 * hr),
    }
    g = bs.greeks(S, K, T, r, q, sigma, kind)
    for name, approx_value in fd.items():
        scale = max(abs(g[name]), 1e-3 * S if name != "gamma" else 1e-3 / S)
        assert abs(g[name] - approx_value) / scale < 1e-4, (name, g[name], approx_value)


def test_iv_round_trip_across_strikes_and_maturities():
    S, r, q = 100.0, 0.03, 0.01
    worst = 0.0
    for K in np.linspace(60.0, 160.0, 21):
        for T in (1 / 52, 0.1, 0.5, 1.0, 2.0):
            for sigma in (0.08, 0.25, 0.6, 1.2):
                for kind in ("call", "put"):
                    p = bs.price(S, K, T, r, q, sigma, kind)
                    lo, _ = bs.bounds(S, K, T, r, q, kind)
                    # when the time value is below ~1e-10 the vol is not identifiable
                    if p - lo < 1e-10:
                        continue
                    iv = bs.implied_vol(p, S, K, T, r, q, kind)
                    assert math.isfinite(iv)
                    assert bs.price(S, K, T, r, q, iv, kind) == pytest.approx(p, abs=1e-10)
                    if bs.vega(S, K, T, r, q, sigma) > 1e-4:
                        worst = max(worst, abs(iv - sigma))
    assert worst < 1e-7


def test_iv_is_nan_when_no_vol_fits():
    S, K, T, r, q = 100.0, 90.0, 0.5, 0.04, 0.01
    lo_c, hi_c = bs.bounds(S, K, T, r, q, "call")
    lo_p, hi_p = bs.bounds(S, 110.0, T, r, q, "put")
    assert math.isnan(bs.implied_vol(lo_c - 0.01, S, K, T, r, q, "call"))       # below intrinsic
    assert math.isnan(bs.implied_vol(hi_c + 0.01, S, K, T, r, q, "call"))       # above S e^{-qT}
    assert math.isnan(bs.implied_vol(lo_p - 0.01, S, 110.0, T, r, q, "put"))
    assert math.isnan(bs.implied_vol(hi_p + 0.01, S, 110.0, T, r, q, "put"))
    assert math.isnan(bs.implied_vol(float("nan"), S, K, T, r, q, "call"))
    assert math.isnan(bs.implied_vol(1.0, S, K, 0.0, r, q, "call"))


def test_static_arbitrage_shape_in_strike():
    S, T, r, q, sigma = 100.0, 0.5, 0.03, 0.01, 0.25
    K = np.linspace(50.0, 150.0, 101)
    c = bs.price(S, K, T, r, q, sigma, "call")
    p = bs.price(S, K, T, r, q, sigma, "put")
    assert np.all(np.diff(c) < 0) and np.all(np.diff(p) > 0)          # monotone in K
    assert np.all(c[:-2] - 2 * c[1:-1] + c[2:] > 0)                    # butterflies cost money
    assert np.all(-np.diff(c) <= np.diff(K) * math.exp(-r * T) + 1e-12)  # call spread <= PV(K2 - K1)
    sigmas = np.linspace(0.05, 1.0, 20)
    assert np.all(np.diff(bs.price(S, 100.0, T, r, q, sigmas, "call")) > 0)  # vega > 0


def test_vectorised_matches_scalar_and_iv_array():
    S, T, r, q = 100.0, 0.3, 0.02, 0.0
    K = np.array([80.0, 100.0, 120.0])
    kinds = np.array(["put", "call", "call"])
    sig = np.array([0.3, 0.2, 0.25])
    vec = bs.price(S, K, T, r, q, sig, kinds)
    for i in range(3):
        assert vec[i] == pytest.approx(bs.price(S, K[i], T, r, q, sig[i], kinds[i]))
    assert np.allclose(bs.implied_vol_array(vec, S, K, T, r, q, kinds), sig, atol=1e-9)
    with pytest.raises(ValueError):
        bs.price(S, 100.0, T, r, q, 0.2, "straddle")
