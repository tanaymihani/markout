"""Delta-hedging simulation: zero-cost sanity checks and the continuous limit."""

import math

import numpy as np
import pytest

from markout.options import bs
from markout.options import hedging as hg

# small grid: 20 days x 16 bars, so intervals 16/4/1 give N = 20, 80, 320 rebalances
BASE = hg.HedgeConfig(days=20, bars_per_day=16, n_paths=6000, seed=7)
INTERVALS = (16, 4, 1)


@pytest.fixture(scope="module")
def fair():
    """sigma_real = sigma_imp, with carry, so every frequency should break even."""
    return hg.simulate(BASE.with_(r=0.05, q=0.02), INTERVALS)


def test_fair_vol_gives_zero_mean_pnl(fair):
    for res in fair.values():
        s = hg.summarize(res)
        assert abs(s["mean"]) < 4 * s["se"], s


def test_hedging_error_std_shrinks_like_one_over_sqrt_n(fair):
    stds = [hg.summarize(fair[h])["std"] for h in INTERVALS]
    assert stds[0] > stds[1] > stds[2]
    # each step quadruples N, so the std should roughly halve
    for a, b in zip(stds, stds[1:]):
        assert 1.6 < a / b < 2.4
    n = [fair[h].n_rebalances for h in INTERVALS]
    assert hg.loglog_slope(n, stds) == pytest.approx(-0.5, abs=0.1)


def test_turnover_grows_with_frequency_and_costs_scale_linearly(fair):
    turnover = [fair[h].turnover.mean() for h in INTERVALS]
    assert turnover[0] < turnover[1] < turnover[2]
    res = fair[1]
    assert np.allclose(res.net(0.0), res.pnl)
    assert np.allclose(res.pnl - res.net(5e-4), 5e-4 * res.turnover)


@pytest.mark.parametrize("sigma_real", [0.12, 0.28])
def test_short_gamma_earns_when_realized_is_below_implied(sigma_real):
    cfg = BASE.with_(sigma_real=sigma_real, r=0.03)
    res = hg.simulate(cfg, (1,))[1]
    s, th = hg.summarize(res), hg.expected_pnl(cfg)
    # sign: short gamma is short realized vol
    assert np.sign(s["mean"]) == np.sign(cfg.sigma_imp - sigma_real)
    # size: the continuous-limit mean C(sigma_imp) - C(sigma_real)
    assert s["mean"] == pytest.approx(th["exact"], abs=4 * s["se"] + 0.02 * abs(th["exact"]))
    # and the first-order vega approximation is close to it for a 8-vol-point gap
    assert th["first_order"] == pytest.approx(th["exact"], rel=0.05)


def test_attribution_tracks_simulated_pnl():
    res = hg.simulate(BASE.with_(sigma_real=0.15), INTERVALS)
    fits = [hg.summarize(res[h]) for h in INTERVALS]
    assert fits[-1]["attr_corr"] > 0.99
    assert [f["resid_std_ratio"] for f in fits] == sorted((f["resid_std_ratio"] for f in fits), reverse=True)


def test_premium_and_initial_hedge_are_black_scholes():
    cfg = BASE.with_(n_paths=10)
    th = hg.expected_pnl(cfg)
    assert th["premium"] == pytest.approx(bs.call_price(cfg.S0, cfg.K, cfg.T, cfg.r, cfg.q, cfg.sigma_imp))
    assert th["vega"] == pytest.approx(bs.vega(cfg.S0, cfg.K, cfg.T, cfg.r, cfg.q, cfg.sigma_imp))
    with pytest.raises(ValueError):
        hg.simulate(cfg, (7,))          # 7 does not divide 320 steps


def test_frontier_picks_finer_hedging_when_costs_are_lower(fair):
    _, choices = hg.frontier(fair, kappas=[0.0, 1e-3], risk_weights=[1.0])
    by_kappa = {c["kappa"]: c for c in choices}
    assert by_kappa[0.0]["interval"] == min(INTERVALS)            # free hedging: hedge as often as possible
    assert by_kappa[1e-3]["n_rebalances"] <= by_kappa[0.0]["n_rebalances"]


def test_interval_labels():
    assert hg.interval_label(1, 78) == "5 min"
    assert hg.interval_label(12, 78) == "1 hour"
    assert hg.interval_label(78, 78) == "1 day"
    assert hg.interval_label(390, 78) == "1 week"
    assert math.isclose(BASE.T, 20 / 252)
