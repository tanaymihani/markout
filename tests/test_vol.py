"""Volatility risk premium: the math, the walk-forward forecast, sizing."""

import numpy as np
import pandas as pd
import pytest

from markout.vol import strategy as S
from markout.vol import vrp


def returns(n=1200, sigma=0.01, seed=0):
    idx = pd.bdate_range("2000-01-03", periods=n)
    return pd.Series(np.random.default_rng(seed).normal(0, sigma, n), index=idx)


def test_forward_realized_variance_uses_the_next_h_days_only():
    r = pd.Series(np.arange(1, 31, dtype=float) / 100, index=pd.bdate_range("2020-01-01", periods=30))
    rv = vrp.forward_rv(r, h=5)
    assert rv.iloc[0] == pytest.approx(252 / 5 * sum((np.arange(2, 7) / 100) ** 2))
    assert rv.iloc[-5:].isna().all()


def test_short_variance_payoff():
    assert S.short_var_pnl(20.0, 20.0) == pytest.approx(0.0)
    assert S.short_var_pnl(20.0, 0.0) == pytest.approx(10.0)  # capped at K/2
    assert S.short_var_pnl(20.0, 40.0) == pytest.approx((400 - 1600) / 40)  # unbounded below
    assert S.short_var_pnl(20.0, 20.0, cost_vol=0.5) == pytest.approx(-0.5)


def test_har_forecast_never_sees_the_future():
    r = returns()
    f = vrp.har_forecast(r, h=21, min_train=500)
    t = 900
    r2 = r.copy()
    r2.iloc[t + 1:] = np.random.default_rng(9).normal(0, 0.05, len(r) - t - 1)  # a different future
    f2 = vrp.har_forecast(r2, h=21, min_train=500)
    np.testing.assert_allclose(f.iloc[:t + 1].to_numpy(), f2.iloc[:t + 1].to_numpy(), equal_nan=True)


def test_har_recovers_a_constant_variance():
    r = returns(sigma=0.01)
    f = vrp.har_forecast(r, h=21, min_train=500).dropna()
    assert f.mean() == pytest.approx(252 * 0.01 ** 2, rel=0.1)


def test_rules_use_the_term_structure_and_the_forecast():
    row = lambda **k: pd.Series({"vix": 20.0, "vix3m": 22.0, "iv2": 0.04, "har": 0.03, **k})  # noqa: E731
    assert S.Rule("c", use_contango=True).trade(row())
    assert not S.Rule("c", use_contango=True).trade(row(vix3m=18.0))
    assert S.Rule("h", har_tau=0.0).trade(row())
    assert not S.Rule("h", har_tau=0.3).trade(row())  # premium 25% < 30%
    assert not S.Rule("h", har_tau=0.0).trade(row(har=float("nan")))


def test_exact_kelly_matches_the_binary_bet_and_mu_sigma2_overbets_fat_tails():
    # win 1 with prob 0.6, lose 1 with prob 0.4: Kelly f* = 0.6 - 0.4 = 0.2
    pnl = np.array([1.0] * 60 + [-1.0] * 40)
    assert S.kelly_exact(pnl)["v_star"] == pytest.approx(0.2, abs=0.002)
    # many small gains, one large loss: mean/variance asks for far more than the exact optimum
    tail = np.array([1.0] * 99 + [-60.0])
    k = S.kelly_exact(tail)
    assert k["v_continuous"] > k["v_star"] and k["v_star"] < k["v_max"] <= 1 / 60


def test_ruin_is_absorbing():
    w = S.wealth_path(np.array([1.0, -200.0, -200.0, 5.0]), v=0.01)
    assert w[1] == 0.0 and w[2] == 0.0 and w[3] == 0.0  # two losses do not multiply back to positive
    assert S.max_drawdown(w) == -1.0
