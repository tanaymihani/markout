"""Kelly sizing: the optimum, the no-edge case, NO-side symmetry, fractional Kelly."""

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.optimize import minimize_scalar

from markout.decision import kelly

prob = st.floats(0.02, 0.98)


@settings(max_examples=200, deadline=None)
@given(p=prob, q=prob)
def test_kelly_fraction_maximises_growth(p, q):
    side, f_star = kelly.best_side(p, q)
    res = minimize_scalar(lambda f: -kelly.growth_rate(f, p, q, side), bounds=(0.0, 1 - 1e-9), method="bounded",
                          options={"xatol": 1e-10})
    assert f_star == pytest.approx(res.x, abs=1e-5)
    assert kelly.growth_rate(f_star, p, q, side) >= kelly.growth_rate(res.x, p, q, side) - 1e-12
    assert kelly.growth_rate(f_star, p, q, side) >= 0.0          # never worse than not betting


@pytest.mark.parametrize("p", [0.1, 0.37, 0.5, 0.9])
def test_no_edge_means_no_bet(p):
    assert kelly.kelly_yes(p, p) == 0.0
    assert kelly.kelly_no(p, p) == 0.0
    assert kelly.expected_return(p, p, "YES") == pytest.approx(0.0)


@settings(max_examples=200, deadline=None)
@given(p=prob, q=prob, f=st.floats(0.0, 0.95))
def test_no_side_is_the_yes_side_of_the_complement(p, q, f):
    assert kelly.kelly_no(p, q) == pytest.approx(kelly.kelly_yes(1 - p, 1 - q), abs=1e-12)
    assert kelly.growth_rate(f, p, q, "NO") == pytest.approx(kelly.growth_rate(f, 1 - p, 1 - q, "YES"), abs=1e-12)


def test_worked_example_and_wrong_side():
    # belief 60% on a contract at 50c: f* = (0.6 - 0.5)/(1 - 0.5) = 0.2 of wealth on YES
    assert kelly.kelly_fraction(0.6, 0.5, "YES") == pytest.approx(0.2)
    assert kelly.kelly_fraction(0.6, 0.5, "no") == 0.0            # your belief says the other side
    assert kelly.best_side(0.3, 0.4) == ("NO", pytest.approx(0.25))
    with pytest.raises(ValueError):
        kelly.kelly_fraction(0.6, 0.5, "maybe")


def test_fractional_kelly_growth_is_two_c_minus_c_squared():
    p, q = 0.52, 0.50
    f = kelly.kelly_yes(p, q)
    g_full = kelly.growth_rate(f, p, q)
    for c in (0.25, 0.5, 1.0, 1.5):
        assert kelly.growth_rate(kelly.fractional(f, c), p, q) == pytest.approx((2 * c - c * c) * g_full, rel=0.02)
    assert kelly.growth_rate(kelly.fractional(f, 2.0), p, q) == pytest.approx(0.0, abs=0.02 * g_full)
    assert kelly.fractional(0.8, 3.0) == 1.0                       # capped: no leverage


def test_all_in_is_ruin_when_a_loss_is_possible():
    assert kelly.growth_rate(1.0, 0.9, 0.5) == -math.inf
    assert kelly.growth_rate(1.0, 1.0, 0.5) == pytest.approx(math.log(2.0))


def test_continuous_kelly_is_mu_over_sigma_squared():
    mu, sigma = 0.01, 0.1
    f_star = kelly.kelly_continuous(mu, sigma**2)
    z = np.random.default_rng(0).standard_normal(200_000)
    growth = lambda f: np.mean(np.log1p(f * (mu + sigma * z)))  # noqa: E731
    res = minimize_scalar(lambda f: -growth(f), bounds=(0.0, 3.0), method="bounded")
    assert f_star == pytest.approx(1.0)
    assert res.x == pytest.approx(f_star, rel=0.1)                # exact only as returns -> 0


def test_vectorised_helpers_match_scalars():
    p = np.array([0.7, 0.2, 0.5])
    q = np.array([0.5, 0.4, 0.5])
    yes, f = kelly.kelly_binary(p, q)
    assert list(yes) == [True, False, True]
    assert np.allclose(f, [0.4, 0.5, 0.0])
    pay = kelly.payoff_per_unit(yes, q, np.array([1, 1, 0]))
    assert np.allclose(pay, [1.0, -1.0, -1.0])
