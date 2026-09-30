"""Market-making tournament: quote tables, accounting, and the three headline results."""

import numpy as np
import pytest

from markout.games import arena as ar
from markout.games import glosten_milgrom as gm
from markout.games.stats import ci_contains

MKT = ar.Market()
N = 400   # seeds per configuration in these tests (the report uses more)


# ---------------------------------------------------------------------------
# Quote tables against closed forms
# ---------------------------------------------------------------------------

def test_zero_profit_quote_matches_closed_form_at_half():
    a, b = ar.zero_profit_edges(np.array([0.5]), MKT)
    h = ar.zero_profit_half_spread_at_half(MKT.mu, MKT.c_max, MKT.dv)
    assert a[0] == pytest.approx(h, abs=1e-10) and b[0] == pytest.approx(h, abs=1e-10)


def test_inelastic_noise_gives_the_glosten_milgrom_spread():
    m = ar.Market(c_max=1e12)
    pi = np.linspace(0.02, 0.98, 49)
    a, b = ar.zero_profit_edges(pi, m)
    np.testing.assert_allclose(a + b, gm.spread(pi, m.mu, m.v_low, m.v_high), rtol=1e-9)


def test_zero_profit_root_is_the_least_fixed_point():
    """Iterating e ← k/(wμ + n(e)) from 0 rises monotonically to the least fixed
    point; it must land on the closed-form root."""
    pi = np.linspace(0.01, 0.99, 99)
    for side, w in (("ask", pi), ("bid", 1 - pi)):
        k = MKT.dv * pi * (1 - pi) * MKT.mu
        e = np.zeros_like(pi)
        for _ in range(500):
            e = k / (w * MKT.mu + MKT.noise_rate(e))
        np.testing.assert_allclose(e, ar.zero_profit_edge(pi, MKT, side), rtol=1e-10)


def test_zero_profit_edges_earn_exactly_nothing():
    pi = np.linspace(0.01, 0.99, 99)
    a, b = ar.zero_profit_edges(pi, MKT)
    np.testing.assert_allclose(ar.side_profit(a, pi, MKT, "ask"), 0.0, atol=1e-12)
    np.testing.assert_allclose(ar.side_profit(b, pi, MKT, "bid"), 0.0, atol=1e-12)
    # and any tighter quote loses money in expectation
    assert np.all(ar.side_profit(0.9 * a, pi, MKT, "ask") < 0)


def test_monopoly_quote_matches_first_order_condition_at_half():
    ma, _ = ar.monopoly_edge(np.array([0.5]), MKT, "ask")
    # d/da [n(a)·a + (μ/2)(a − Δ/2)] = 0  →  a = c_max / (2(1 − μ))
    assert ma[0] == pytest.approx(MKT.c_max / (2 * (1 - MKT.mu)), abs=1e-3)
    tab = MKT.tables
    assert np.all(tab.mono_ask >= tab.be_ask) and np.all(tab.mono_profit > 0)


def test_market_breakdown_is_detected():
    with pytest.raises(ValueError, match="breaks down"):
        ar.QuoteTables.build(ar.Market(mu=0.6, c_max=0.3))


# ---------------------------------------------------------------------------
# Engine mechanics
# ---------------------------------------------------------------------------

def test_min_excluding_self_and_tie_picks():
    x = np.array([[3, 1, 5], [1, 1, 5], [2, 4, 5]])
    np.testing.assert_array_equal(ar._min_excluding_self(x),
                                  [[1, 1, 5], [2, 1, 5], [1, 1, 5]])
    at = np.array([[True, False], [True, True], [False, True]])
    np.testing.assert_array_equal(ar._pick(at, np.array([0.0, 0.99])), [0, 2])
    np.testing.assert_array_equal(ar._pick(at, np.array([0.6, 0.2])), [1, 1])


@pytest.fixture(scope="module")
def ffa():
    return ar.simulate(MKT, ar.default_agents(), n_seeds=N, seed=0)


def test_accounting_identities(ffa):
    ms = ffa.market_summary()
    assert ms["zero_sum_max_error"] < 1e-8                    # makers + traders = 0
    np.testing.assert_allclose(ffa.pnl, ffa.edge - ffa.adverse, atol=1e-8)
    np.testing.assert_array_equal(ffa.fills.sum(axis=0), ffa.trades)
    assert np.all(ffa.informed_fills <= ffa.fills)


def test_common_random_numbers_across_lineups(ffa):
    other = ar.simulate(MKT, [ar.GMQuoter()], n_seeds=N, seed=0)
    np.testing.assert_array_equal(other.value_high, ffa.value_high)


def test_public_belief_learns_the_value_and_is_calibrated():
    run = ar.simulate(MKT, ar.default_agents(), n_seeds=4000, seed=1)
    path = run.belief_truth_path
    assert path[0] == pytest.approx(MKT.prior)
    assert path[100] > path[50] > path[10] > path[0] and path[100] > 0.95
    # a correct Bayesian filter is calibrated: P(V = V_H | belief in a bin) ≈ mean belief there
    pi, hit = run.final_belief, run.value_high.astype(float)
    edges = np.array([0.0, 0.02, 0.1, 0.3, 0.7, 0.9, 0.98, 1.0])
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (pi >= lo) & (pi <= hi)
        if m.sum() < 30:
            continue
        freq, avg = hit[m].mean(), pi[m].mean()
        se = np.sqrt(max(avg * (1 - avg), 1e-4) / m.sum())
        assert abs(freq - avg) < 4 * se + 1e-3, (lo, hi, freq, avg, m.sum())


# ---------------------------------------------------------------------------
# The three headline results
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def k_sweep():
    return {k: ar.simulate(MKT, [ar.clone(ar.Undercutter(), f"u{i}") for i in range(k)],
                           n_seeds=N, seed=0) for k in (1, 2, 4)}


def test_monopolist_earns_rents_with_a_wide_spread(k_sweep):
    mono = k_sweep[1].market_summary()
    assert mono["maker_pnl"]["lo"] > 0
    assert mono["quoted_spread"]["mean"] > 5 * mono["zero_profit_spread"]["mean"]


def test_competition_compresses_spreads_to_zero_profit(k_sweep):
    for k in (2, 4):
        ms = k_sweep[k].market_summary()
        # quotes are rounded away from the mid, so at most one tick above zero profit per side
        excess = ms["excess_spread"]["mean"]
        assert 0 <= excess <= 2 * MKT.tick + 1e-12
        # maker profit per trade is no more than the rounding rent (< 1 tick)
        assert ms["maker_pnl_per_trade"]["hi"] < MKT.tick
        assert ms["maker_pnl_per_trade"]["lo"] > -MKT.tick
        # noise traders are better off than under the monopolist
        assert (ms["noise_welfare_per_period"]["lo"]
                > k_sweep[1].market_summary()["noise_welfare_per_period"]["hi"])


def test_identical_makers_split_the_flow(k_sweep):
    run = k_sweep[4]
    for i in range(4):
        assert ci_contains(run.agent_summary(i)["fill_share"], 0.25)


def test_naive_tight_quoting_loses_to_informed_flow():
    run = ar.simulate(MKT, [ar.FixedSpread("tight", 0.05)], n_seeds=N, seed=0)
    s = run.agent_summary(0)
    assert s["pnl"]["hi"] < 0
    assert s["adverse_per_fill"]["mean"] > s["edge_per_fill"]["mean"]


def test_gm_quoter_earns_only_the_rounding_rent():
    s = ar.simulate(MKT, [ar.GMQuoter()], n_seeds=N, seed=0).agent_summary(0)
    assert -MKT.tick < s["pnl_per_fill"]["lo"] and s["pnl_per_fill"]["hi"] < MKT.tick


def test_undercutter_takes_the_wide_quoters_flow():
    run = ar.simulate(MKT, [ar.FixedSpread("wide", 0.60), ar.Undercutter()], n_seeds=N, seed=0)
    h = ar.head_to_head(run)
    assert h["winner"] == "undercut"
    assert run.agent_summary(1)["fill_share"]["mean"] > 0.99


# ---------------------------------------------------------------------------
# Replicator dynamics
# ---------------------------------------------------------------------------

def test_replicator_eliminates_a_dominated_strategy():
    payoff = np.array([[1.0, 1.0], [0.0, 0.0]])   # row 0 strictly dominates row 1
    path = ar.replicator(payoff, dt=0.1, n_steps=500)
    np.testing.assert_allclose(path.sum(axis=1), 1.0)
    assert path[-1, 0] > 0.999
