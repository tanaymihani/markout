"""Kuhn poker: exact evaluation, exact best responses, CFR convergence, exploitation."""

from itertools import product

import numpy as np
import pytest

from markout.games import kuhn_cfr as kc


def _pure_strategies(infosets):
    for bits in product((0, 1), repeat=len(infosets)):
        yield {k: np.eye(2)[b] for k, b in zip(infosets, bits)}


def _random_profile(rng):
    return {k: np.array([p, 1 - p]) for k, p in zip(kc.INFOSETS, rng.random(len(kc.INFOSETS)))}


@pytest.mark.parametrize("alpha", [0.0, 1 / 6, 1 / 3])
def test_analytic_equilibrium_family(alpha):
    eq = kc.analytic_equilibrium(alpha)
    assert kc.expected_value(eq) == pytest.approx(kc.GAME_VALUE, abs=1e-14)
    assert kc.exploitability(eq) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_best_response_matches_brute_force(seed):
    """A best response is a pure strategy, so the exact BR value must equal the
    best of all 2⁶ pure strategies of that player."""
    prof = _random_profile(np.random.default_rng(seed))
    v1, br1 = kc.best_response(prof, 0)
    brute1 = max(kc.expected_value(kc.combine(s, prof)) for s in _pure_strategies(kc.P1_INFOSETS))
    assert v1 == pytest.approx(brute1, abs=1e-12)
    assert kc.expected_value(kc.combine(br1, prof)) == pytest.approx(v1, abs=1e-12)
    v2, br2 = kc.best_response(prof, 1)
    brute2 = max(-kc.expected_value(kc.combine(prof, s)) for s in _pure_strategies(kc.P2_INFOSETS))
    assert v2 == pytest.approx(brute2, abs=1e-12)
    assert kc.exploitability(prof) > 0


@pytest.fixture(scope="module")
def cfr_avg():
    avg, hist = kc.solve(n_iter=50_000, checkpoints=[100, 1_000, 10_000, 50_000])
    return avg, hist


def test_cfr_value_and_exploitability(cfr_avg):
    avg, hist = cfr_avg
    assert kc.expected_value(avg) == pytest.approx(-1 / 18, abs=2e-3)
    assert kc.exploitability(avg) < 5e-3
    expl = [h["exploitability"] for h in hist]
    assert all(b < a for a, b in zip(expl, expl[1:]))       # falls at every checkpoint


def test_cfr_lands_in_the_equilibrium_family(cfr_avg):
    fam = kc.equilibrium_family_check(cfr_avg[0])
    assert -0.01 <= fam["alpha"] <= 1 / 3 + 0.01
    assert fam["king_bet"] == pytest.approx(3 * fam["alpha"], abs=0.03)
    assert fam["queen_bet"] < 0.01
    assert fam["queen_call"] == pytest.approx(fam["alpha"] + 1 / 3, abs=0.03)
    assert fam["p2_jack_bluff"] == pytest.approx(1 / 3, abs=0.02)
    assert fam["p2_queen_call"] == pytest.approx(1 / 3, abs=0.02)


def test_cfr_plus_converges_faster():
    vanilla, _ = kc.solve(n_iter=2_000)
    plus, _ = kc.solve(n_iter=2_000, plus=True)
    assert kc.exploitability(plus) < kc.exploitability(vanilla) / 5


def test_mixture_is_realization_equivalent():
    """mix_p1(eq, br, λ) is the mixed strategy 'play br w.p. λ': its EV against any
    opponent is linear in λ."""
    rng = np.random.default_rng(3)
    eq = kc.analytic_equilibrium(0.2)
    br = {k: np.eye(2)[rng.integers(2)] for k in kc.P1_INFOSETS}
    opp = _random_profile(rng)
    e0 = kc.expected_value(kc.combine(eq, opp))
    e1 = kc.expected_value(kc.combine(br, opp))
    for lam in (0.25, 0.5, 0.9):
        m = kc.mix_p1(eq, kc.combine(br, eq), lam)
        assert kc.expected_value(kc.combine(m, opp)) == pytest.approx(lam * e1 + (1 - lam) * e0,
                                                                      abs=1e-12)


@pytest.mark.parametrize("leak", ["never_bluff_jack", "always_call_queen"])
def test_exploit_versus_protect(leak):
    eq = kc.analytic_equilibrium(0.2)
    st = kc.exploitation_study(eq, leak)
    # the leak sits at an infoset where the equilibrium makes P2 indifferent, so
    # equilibrium play neither gains nor loses from it
    assert st["ev_eq"] == pytest.approx(kc.GAME_VALUE, abs=1e-12)
    assert st["gain"] > 0.05
    # exploiting opens P1 up: against a best-responding P2 it does worse than −1/18
    assert st["eq_worst"] == pytest.approx(kc.GAME_VALUE, abs=1e-12)
    assert st["br_worst"] < st["eq_worst"] - 0.05
    f = st["frontier"]
    assert f[0]["ev_vs_leak"] == pytest.approx(st["ev_eq"]) and f[-1]["ev_vs_leak"] == pytest.approx(st["ev_br"])
    assert all(b["ev_vs_leak"] >= a["ev_vs_leak"] - 1e-12 for a, b in zip(f, f[1:]))
    assert all(b["worst_case"] <= a["worst_case"] + 1e-12 for a, b in zip(f, f[1:]))
