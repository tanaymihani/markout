"""Glosten–Milgrom (1985) sequential-trade model.

Glosten, L. and Milgrom, P. (1985), "Bid, ask and transaction prices in a
specialist market with heterogeneously informed traders", *Journal of Financial
Economics* 14(1), 71–100.

Setup
-----
The asset pays V ∈ {V_L, V_H}; the market maker's belief is π = P(V = V_H).
One trader arrives per period and trades one unit:

- with probability μ it is *informed*: it buys if V = V_H and sells if V = V_L;
- otherwise it is a *noise* trader that buys or sells with probability ½ each.

So P(buy | V_H) = (1 + μ)/2 and P(buy | V_L) = (1 − μ)/2. A competitive,
risk-neutral maker quotes the conditional expectations (zero expected profit on
every trade, "regret-free" quotes):

    ask = E[V | buy]  = V_L + (V_H − V_L)·π⁺,   π⁺ = π(1 + μ) / (1 + μ(2π − 1))
    bid = E[V | sell] = V_L + (V_H − V_L)·π⁻,   π⁻ = π(1 − μ) / (1 − μ(2π − 1))

and after the trade the belief becomes π⁺ or π⁻ (Bayes' rule). The spread is

    ask − bid = (V_H − V_L) · 4π(1 − π)μ / (1 − μ²(2π − 1)²),

which equals μ(V_H − V_L) at π = ½, rises with μ, and is concave in π. Because π
is a martingale under the maker's information and the spread is concave in π,
the expected spread can only fall as trades reveal V (Jensen's inequality).

The closed forms below use only +, −, ×, ÷, so they accept floats, NumPy arrays
or :class:`fractions.Fraction` (the last gives exact rational answers).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GMParams:
    """Model parameters: the two values, the informed share μ and the prior π₀."""

    v_low: float = 0.0
    v_high: float = 1.0
    mu: float = 0.3
    prior: float = 0.5

    def __post_init__(self) -> None:
        if not self.v_high > self.v_low:
            raise ValueError("need v_high > v_low")
        if not 0 <= self.mu < 1:
            raise ValueError("need 0 <= mu < 1 (mu = 1 means no noise trader ever arrives)")
        if not 0 < self.prior < 1:
            raise ValueError("need 0 < prior < 1")


# ---------------------------------------------------------------------------
# Closed forms
# ---------------------------------------------------------------------------

def posterior_after_buy(pi, mu):
    """P(V = V_H | buy) = π(1 + μ) / (1 + μ(2π − 1))."""
    return pi * (1 + mu) / (1 + mu * (2 * pi - 1))


def posterior_after_sell(pi, mu):
    """P(V = V_H | sell) = π(1 − μ) / (1 − μ(2π − 1))."""
    return pi * (1 - mu) / (1 - mu * (2 * pi - 1))


def quotes(pi, mu, v_low=0.0, v_high=1.0):
    """Zero-profit (bid, ask) = (E[V | sell], E[V | buy]) at belief π."""
    dv = v_high - v_low
    return (v_low + dv * posterior_after_sell(pi, mu),
            v_low + dv * posterior_after_buy(pi, mu))


def spread(pi, mu, v_low=0.0, v_high=1.0):
    """ask − bid = (V_H − V_L)·4π(1 − π)μ / (1 − μ²(2π − 1)²)."""
    d = 2 * pi - 1
    return (v_high - v_low) * 4 * pi * (1 - pi) * mu / (1 - mu * mu * d * d)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

@dataclass
class GMPaths:
    """Simulated paths, one row per run.

    ``belief[:, t]`` is π before trade t (the last column is the belief after
    the final trade). ``side`` is the trader's direction (+1 buy, −1 sell).
    PnL is marked to the realized V; the per-run totals satisfy
    ``maker + insider + noise == 0`` exactly (trading is zero-sum).
    """

    params: GMParams
    value_high: np.ndarray   # (n_runs,) bool, True when V = V_H
    belief: np.ndarray       # (n_runs, n_steps + 1)
    bid: np.ndarray          # (n_runs, n_steps)
    ask: np.ndarray          # (n_runs, n_steps)
    side: np.ndarray         # (n_runs, n_steps) int8, +1 buy / −1 sell
    informed: np.ndarray     # (n_runs, n_steps) bool
    maker_pnl: np.ndarray    # (n_runs,)
    insider_pnl: np.ndarray  # (n_runs,)
    noise_pnl: np.ndarray    # (n_runs,)

    @property
    def value(self) -> np.ndarray:
        p = self.params
        return np.where(self.value_high, p.v_high, p.v_low)

    @property
    def spread(self) -> np.ndarray:
        return self.ask - self.bid

    @property
    def belief_in_truth(self) -> np.ndarray:
        """P(V = realized V) under the maker's belief, shape (n_runs, n_steps + 1)."""
        return np.where(self.value_high[:, None], self.belief, 1.0 - self.belief)


def simulate(params: GMParams, n_steps: int, n_runs: int,
             rng: np.random.Generator) -> GMPaths:
    """Simulate ``n_runs`` independent GM markets of ``n_steps`` trades each.

    Each run draws V from the prior, then each period a trader arrives
    (informed with probability μ) and trades one unit at the maker's quote.
    Vectorized across runs.
    """
    p = params
    value_high = rng.random(n_runs) < p.prior
    informed = rng.random((n_runs, n_steps)) < p.mu
    noise_buy = rng.random((n_runs, n_steps)) < 0.5
    buy = np.where(informed, value_high[:, None], noise_buy)
    side = np.where(buy, 1, -1).astype(np.int8)

    belief = np.empty((n_runs, n_steps + 1))
    belief[:, 0] = p.prior
    bid = np.empty((n_runs, n_steps))
    ask = np.empty((n_runs, n_steps))
    for t in range(n_steps):
        pi = belief[:, t]
        bid[:, t], ask[:, t] = quotes(pi, p.mu, p.v_low, p.v_high)
        belief[:, t + 1] = np.where(buy[:, t], posterior_after_buy(pi, p.mu),
                                    posterior_after_sell(pi, p.mu))

    v = np.where(value_high, p.v_high, p.v_low)[:, None]
    price = np.where(buy, ask, bid)
    # the trader gains side·(V − price); the maker is the other side of every trade
    trader_gain = side * (v - price)
    insider = np.where(informed, trader_gain, 0.0).sum(axis=1)
    noise = np.where(informed, 0.0, trader_gain).sum(axis=1)
    maker = -trader_gain.sum(axis=1)
    return GMPaths(p, value_high, belief, bid, ask, side, informed, maker, insider, noise)


def spread_by_trade(paths: GMPaths) -> np.ndarray:
    """Mean quoted spread across runs before each trade, shape (n_steps,)."""
    return paths.spread.mean(axis=0)


def belief_after_net_flow(k, params: GMParams):
    """Belief after trades with net order flow k = #buys − #sells.

    Each buy multiplies the odds π/(1 − π) by r = (1 + μ)/(1 − μ) and each sell
    divides it by r, so the order of trades does not matter:
    π_k = 1 / (1 + (1 − π₀)/π₀ · r^(−k)).
    """
    r = (1 + params.mu) / (1 - params.mu)
    odds0 = params.prior / (1 - params.prior)
    return 1.0 / (1.0 + 1.0 / (odds0 * np.power(r, k, dtype=float)))


def expected_paths(params: GMParams, n_steps: int) -> dict[str, np.ndarray]:
    """Exact expected spread and belief in the truth before trade t = 0..n_steps−1.

    No Monte Carlo: given V_H the number of buys after t trades is
    Binomial(t, (1 + μ)/2) (Binomial(t, (1 − μ)/2) given V_L), and the belief
    depends only on k = 2·#buys − t, so each expectation is a finite sum
    Σ_V P(V) Σ_b P(b | V, t)·f(π_(2b−t)).
    """
    from scipy.stats import binom

    p = params
    spread_path = np.empty(n_steps)
    truth_path = np.empty(n_steps)
    for t in range(n_steps):
        b = np.arange(t + 1)
        pi = belief_after_net_flow(2 * b - t, p)
        s = spread(pi, p.mu, p.v_low, p.v_high)
        w_h = binom.pmf(b, t, (1 + p.mu) / 2)
        w_l = binom.pmf(b, t, (1 - p.mu) / 2)
        spread_path[t] = p.prior * (w_h @ s) + (1 - p.prior) * (w_l @ s)
        truth_path[t] = p.prior * (w_h @ pi) + (1 - p.prior) * (w_l @ (1 - pi))
    return {"spread": spread_path, "belief_in_truth": truth_path}


def expected_spread_path(params: GMParams, n_steps: int) -> np.ndarray:
    """Exact E[spread before trade t], t = 0..n_steps−1 (see :func:`expected_paths`)."""
    return expected_paths(params, n_steps)["spread"]
