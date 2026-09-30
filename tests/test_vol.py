"""Volatility risk premium: the math, the walk-forward forecast, sizing."""

import numpy as np
import pandas as pd
import pytest

from markout.options import bs
from markout.vol import straddle as ST
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


# ---- the delta-hedged straddle -------------------------------------------------------


def straddle_value(sigma, T=21 / 252):
    return bs.price(1.0, 1.0, T, 0.0, 0.0, sigma, "call") + bs.price(1.0, 1.0, T, 0.0, 0.0, sigma, "put")


def test_straddle_keeps_the_premium_when_nothing_moves():
    m = ST.hedge_month(np.zeros(21), 0.20)
    assert m["gross"] == pytest.approx(m["premium_per_vega"])  # no payoff, no hedge PnL
    assert m["premium_per_vega"] == pytest.approx(20.0, rel=0.01)  # an ATM straddle costs about vega x sigma
    assert m["eff_vol"] == 0.0
    assert 1.5 < m["exposure"] < 2.0  # pinned at the strike: about twice an average month's dollar gamma


def test_hedged_straddle_earns_the_price_difference_on_average():
    # zero-drift paths at 15% vol, straddle sold at 20%: whatever the hedge does, the mean PnL is V(20%) - V(15%)
    rng = np.random.default_rng(0)
    real, implied, n = 0.15, 0.20, 4000
    paths = rng.normal(-0.5 * real ** 2 / 252, real / np.sqrt(252), size=(n, 21))
    pnl = np.array([ST.hedge_month(p, implied)["gross"] for p in paths])
    vega_pt = 2 * bs.vega(1.0, 1.0, 21 / 252, 0.0, 0.0, implied) / 100
    expected = (straddle_value(implied) - straddle_value(real)) / vega_pt
    assert pnl.mean() == pytest.approx(expected, abs=3 * pnl.std() / np.sqrt(n))


def test_gamma_theta_attribution_tracks_the_hedged_pnl():
    rng = np.random.default_rng(1)
    months = [ST.hedge_month(rng.normal(0, 0.20 / np.sqrt(252), 21), 0.20) for _ in range(500)]
    gross, attribution = (np.array([m[k] for m in months]) for k in ("gross", "attribution"))
    assert np.corrcoef(gross, attribution)[0, 1] > 0.9


def test_straddle_months_line_up_with_the_variance_swap():
    idx = pd.bdate_range("2000-01-03", periods=300)
    spx = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(2).normal(0, 0.01, 300))), index=idx)
    fr = vrp.frame(pd.DataFrame({"spx": spx, "vix": 20.0, "vix3m": np.nan}))
    res = ST.run(fr, "2000-01-03", cost_vol=0.0, hedge_bps=0.0)
    assert list(res.index) == list(S.rolls(fr, "2000-01-03").index)
    d = res.index[1]
    p = fr.index.get_loc(d)
    month = fr["r"].to_numpy()[p + 1:p + 22]  # the returns behind the swap's realized vol
    assert 100 * np.sqrt(252 / 21 * np.sum(month ** 2)) == pytest.approx(res.loc[d, "rv_vol"])
    assert res.loc[d, "straddle"] == pytest.approx(ST.hedge_month(month, 0.20)["pnl"])
    assert res.loc[d, "var_swap"] == pytest.approx(float(S.short_var_pnl(20.0, res.loc[d, "rv_vol"])))
    costly = ST.run(fr, "2000-01-03", cost_vol=0.0, hedge_bps=1.0)
    assert (costly["straddle"] < res["straddle"]).all()  # every month trades the hedge
    cheap = ST.run(fr, "2000-01-03", cost_vol=0.0, hedge_bps=0.0, vol_shift=-2.0)
    assert cheap["straddle"].mean() < res["straddle"].mean()
