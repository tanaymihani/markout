"""Kyle (1985): closed form, best-response iteration, Monte Carlo, bridge to module D."""

import math

import numpy as np
import pytest

from markout.games import kyle


@pytest.mark.parametrize("s0, su", [(4.0, 1.0), (0.25, 3.0), (9.0, 0.5)])
def test_closed_form(s0, su):
    k = kyle.KyleParams(100.0, s0, su)
    eq = kyle.equilibrium(k)
    assert eq.beta == pytest.approx(su / math.sqrt(s0))
    assert eq.lam == pytest.approx(math.sqrt(s0) / (2 * su))
    assert eq.lam * eq.beta == pytest.approx(0.5)
    assert eq.insider_profit == pytest.approx(0.5 * su * math.sqrt(s0))
    assert eq.posterior_var == pytest.approx(s0 / 2)
    # the closed form is a mutual best response
    assert kyle.insider_best_response(eq.lam) == pytest.approx(eq.beta)
    assert kyle.maker_best_response(eq.beta, k) == pytest.approx(eq.lam)
    assert kyle.expected_insider_profit(eq.beta, eq.lam, k) == pytest.approx(eq.insider_profit)


@pytest.mark.parametrize("start", [1e-3, 0.1, 0.5, 0.99, 1.01, 2.0, 10.0, 1e3])
def test_best_response_iteration_converges(start):
    k = kyle.KyleParams(100.0, 4.0, 1.5)
    lam_star = kyle.equilibrium(k).lam
    path = kyle.lambda_iteration(k, start * lam_star, n_iter=60)
    assert abs(path[-1] - lam_star) < 1e-12 * lam_star
    # after the first step the sequence never overshoots λ* and rises monotonically
    assert np.all(path[1:] <= lam_star * (1 + 1e-12))
    assert np.all(np.diff(path[1:]) >= -1e-15)


def test_monte_carlo_profit_peaks_at_beta_star():
    k = kyle.KyleParams(100.0, 4.0, 1.0)
    eq = kyle.equilibrium(k)
    rel = np.linspace(0.2, 2.0, 91)            # grid step 0.02·β*
    mean, se = kyle.mc_insider_profit(rel * eq.beta, eq.lam, k, n=400_000,
                                      rng=np.random.default_rng(7))
    best = rel[np.argmax(mean)]
    assert abs(best - 1.0) <= 0.021
    i_star = int(np.argmin(np.abs(rel - 1.0)))
    assert abs(mean[i_star] - eq.insider_profit) < 3 * se[i_star]
    # the analytic curve β(1 − λβ)Σ₀ lies inside the Monte Carlo band everywhere
    analytic = kyle.expected_insider_profit(rel * eq.beta, eq.lam, k)
    assert np.all(np.abs(mean - analytic) < 4 * se)


def test_market_maker_regression_recovers_lambda_and_posterior_variance():
    k = kyle.KyleParams(100.0, 4.0, 1.0)
    out = kyle.mc_market_maker_check(k, n=400_000, rng=np.random.default_rng(11))
    assert out["lambda_hat"] == pytest.approx(out["lambda_star"], rel=0.01)
    assert out["posterior_var_hat"] == pytest.approx(out["posterior_var_star"], rel=0.01)


def test_bridge_without_module_d():
    out = kyle.kyle_bridge({"something_else": 1})
    assert out["available"] is False and "module D" in out["note"]
    assert kyle.kyle_bridge({})["available"] is False


@pytest.mark.parametrize("block", [
    {"AAPL": 0.02, "INTC": 0.9},
    {"AAPL": {"beta": 0.02, "r2": 0.4}, "INTC": {"beta": 0.9, "r2": 0.7}},
    [{"ticker": "AAPL", "beta_ticks_per_1000": 0.02}, {"ticker": "INTC", "beta_ticks_per_1000": 0.9}],
    {"rows": [{"stock": "AAPL", "cks_beta": 0.02}, {"stock": "INTC", "cks_beta": 0.9}]},
])
def test_bridge_reads_several_shapes(block):
    out = kyle.kyle_bridge({"kyle_bridge": block})
    assert out["available"] is True
    got = {r["ticker"]: r["cks_beta"] for r in out["rows"]}
    assert got == {"AAPL": 0.02, "INTC": 0.9}
    assert out["has_kyle_lambda"] is False


def test_bridge_computes_kyle_lambda_when_dispersions_are_given():
    block = {"MSFT": {"beta": 0.5, "sd_dmid_ticks": 2.0, "sd_trade_imbalance_k": 4.0}}
    row = kyle.kyle_bridge({"kyle_bridge": block})["rows"][0]
    assert row["kyle_lambda_trades"] == pytest.approx(0.5)     # sd(Δp)/sd(y)
    assert row["ratio_cks_to_kyle"] == pytest.approx(1.0)


def test_bridge_reads_module_d_schema():
    """The layout module D's report writes (one dict of fields per ticker)."""
    block = {"INTC": {"beta_ticks_per_1000_shares": 0.25, "r2": 0.64, "mean_depth_best": 2000.0,
                      "mean_price_usd": 27.3, "beta_usd_per_share": 2.5e-6,
                      "beta_bps_per_1000_shares": 0.9}}
    out = kyle.kyle_bridge({"kyle_bridge": block})
    row = out["rows"][0]
    assert out["available"] and row["cks_beta"] == 0.25
    assert row["kyle_ratio"] == pytest.approx(0.25 / 0.8)       # β/√R²
    assert row["beta_times_depth"] == pytest.approx(0.5)        # β × depth in ticks
    assert row["price"] == 27.3 and out["has_kyle_lambda"] is False
