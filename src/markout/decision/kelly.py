"""Kelly sizing for binary contracts and for continuous returns.

Kelly (1956), "A New Interpretation of Information Rate"; Thorp (2006), "The Kelly
Criterion in Blackjack, Sports Betting and the Stock Market".

Binary contract: a YES share costs q (0 < q < 1) and pays 1 if the event happens;
a NO share costs 1 - q and pays 1 if it does not. With belief p = P(event) and a
fraction f of wealth spent on YES, wealth is multiplied by 1 + f (1 - q)/q with
probability p and by 1 - f with probability 1 - p, so the expected log growth is

    g(f) = p ln(1 + f (1 - q)/q) + (1 - p) ln(1 - f).

Setting g'(f) = 0 gives the Kelly fraction

    YES:  f* = (p - q) / (1 - q)       (bet only when p > q)
    NO:   f* = (q - p) / q             (bet only when p < q)

The NO side is the YES formula applied to the complementary event, (1 - p, 1 - q).
Fractional Kelly bets c * f*. For small edges g(c f*) ~= (2c - c^2) g(f*), so half
Kelly keeps about three quarters of the growth at half the volatility, and 2x Kelly
has about zero growth.

Continuous returns with mean mu and variance sigma^2 per period (small):
g(f) ~= f mu - f^2 sigma^2 / 2, maximised at f* = mu / sigma^2.

These formulas take p as the truth. When p is itself a noisy estimate and you bet
where p differs most from the market, the edge you act on is biased upward (the
optimizer's curse, Smith & Winkler 2006). Fractional Kelly partly corrects for that.
"""

from __future__ import annotations

from typing import Union

import numpy as np

ArrayLike = Union[float, np.ndarray]
SIDES = ("YES", "NO")


def _side(side: str) -> str:
    s = side.strip().upper()
    if s not in SIDES:
        raise ValueError(f"side must be YES or NO, got {side!r}")
    return s


def kelly_yes(p: ArrayLike, q: ArrayLike) -> ArrayLike:
    """f* = (p - q)/(1 - q) for buying YES at price q, floored at 0 (no bet)."""
    return np.maximum((np.asarray(p, float) - q) / (1.0 - np.asarray(q, float)), 0.0)[()]


def kelly_no(p: ArrayLike, q: ArrayLike) -> ArrayLike:
    """f* = (q - p)/q for buying NO (price 1 - q), floored at 0 (no bet)."""
    return np.maximum((np.asarray(q, float) - p) / np.asarray(q, float), 0.0)[()]


def kelly_fraction(p: ArrayLike, q: ArrayLike, side: str) -> ArrayLike:
    """Kelly fraction of wealth for the given side; 0 when the belief favours the other side."""
    return kelly_yes(p, q) if _side(side) == "YES" else kelly_no(p, q)


def best_side(p: float, q: float) -> tuple[str, float]:
    """The side your belief favours and its Kelly fraction ("YES", 0.0 when p == q)."""
    return ("YES", float(kelly_yes(p, q))) if p >= q else ("NO", float(kelly_no(p, q)))


def kelly_binary(p: ArrayLike, q: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised: (buy_yes, f*) for the favoured side of every (p, q)."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    yes = p >= q
    return yes, np.where(yes, kelly_yes(p, q), kelly_no(p, q))


def fractional(f: ArrayLike, c: float, cap: float = 1.0) -> ArrayLike:
    """c times the Kelly fraction, capped at `cap` of wealth (no leverage by default)."""
    return np.minimum(np.multiply(c, f), cap)[()]


def payoff_per_unit(buy_yes: ArrayLike, q: ArrayLike, outcome: ArrayLike) -> ArrayLike:
    """Profit per unit staked: YES pays 1/q - 1 or -1; NO pays 1/(1 - q) - 1 or -1."""
    y = np.asarray(outcome, float)
    return np.where(buy_yes, y / q - 1.0, (1.0 - y) / (1.0 - np.asarray(q, float)) - 1.0)[()]


def expected_return(p: ArrayLike, q: ArrayLike, side: str) -> ArrayLike:
    """Expected profit per unit staked under belief p: (p - q)/q for YES, (q - p)/(1 - q) for NO."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    return ((p - q) / q if _side(side) == "YES" else (q - p) / (1.0 - q))[()]


def growth_rate(f: ArrayLike, p: float, q: float, side: str = "YES") -> ArrayLike:
    """Expected log growth per bet, g(f); -inf when a loss would wipe you out (f >= 1)."""
    f = np.asarray(f, float)
    if _side(side) == "NO":
        p, q = 1.0 - p, 1.0 - q
    with np.errstate(divide="ignore", invalid="ignore"):
        win = np.log1p(f * (1.0 - q) / q)
        lose = np.where(f < 1.0, np.log1p(-np.minimum(f, 1.0)), -np.inf)
        g = p * win + (1.0 - p) * lose
    return np.where((p == 1.0) & (f >= 1.0), win, g)[()]


def kelly_continuous(mu: ArrayLike, sigma2: ArrayLike) -> ArrayLike:
    """Thorp's continuous approximation f* = mu / sigma^2."""
    return (np.asarray(mu, float) / np.asarray(sigma2, float))[()]
