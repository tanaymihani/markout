"""Market-making tournament: Glosten–Milgrom informed flow, K competing makers.

This is module E's "simulated competition". It extends the Glosten & Milgrom
(1985) market (see :mod:`markout.games.glosten_milgrom`) to several makers who
compete on price, and to noise traders whose demand is *elastic*.

Market (one episode = one seed)
-------------------------------
- **Value.** V ∈ {V_L, V_H}, drawn once per episode from the prior π₀. Binary V
  keeps the Bayesian benchmark exact: the public belief and the zero-profit
  quotes have closed forms, so "did competition drive spreads to the zero-profit
  level?" can be checked against an exact number every period. It also makes an
  episode one information event, as in GM and in the PIN model of Easley et al.
  (1996). The cost is that V is learned within the episode, so adverse selection
  is concentrated early; T is chosen so learning takes a large part of it.
- **Arrivals.** One trader per period. With probability μ it is informed: it
  knows V and buys (sells) one unit if the best ask is below (best bid above) V.
  Otherwise it is a noise trader with a coin-flip direction and a private
  urgency c ~ U[0, c_max]. A noise buyer trades only if the half-spread it pays,
  ask − m_t, is below c (a seller: m_t − bid), which happens with probability
  q(e) = 1 − e/c_max for an edge e. For quotes centred on the mid this is exactly
  "the half-spread is below c". Measuring each side from the mid is what
  Avellaneda & Stoikov's arrival intensities do; it also means a maker cannot
  tax one side for free by skewing. Elastic demand gives a monopolist a finite
  optimal spread.
- **Quotes and matching.** K makers each post a bid and an ask on a tick grid.
  The trader hits the best price; ties are split uniformly at random.
- **Information.** Quotes and trades are public. Every maker sees the same tape
  and shares one Bayesian belief π_t = P(V = V_H | tape), computed with the true
  μ and c_max; the public mid is m_t = V_L + π_t(V_H − V_L). Makers therefore
  differ only in how they turn that belief (and their own inventory) into
  quotes, which isolates quoting policy from forecasting skill.
- **Accounting.** Cash plus inventory marked to V at the end of the episode.
  Each fill's PnL splits exactly into edge − adverse selection:
  d·(price − V) = d·(price − m_t) − d·(V − m_t), with d = +1 when the trader
  buys. The second term is the fill's markout to the episode-end value.

Zero-profit and monopoly quotes
-------------------------------
Write a = ask − m and b = m − bid for the edges, Δ = V_H − V_L, and
n(e) = (1 − μ)q(e)/2 for the probability of a noise buy at edge e. A buy then
comes from an informed trader with probability πμ / (πμ + n(a)), and the
zero-profit (GM) edges are the least fixed points of

    a = Δπ(1 − π)μ / (πμ + n(a)),   b = Δπ(1 − π)μ / ((1 − π)μ + n(b)).

With inelastic noise (c_max → ∞, so n ≡ (1 − μ)/2) this is exactly the GM
spread. A single maker's expected profit per period separates by side:

    n(a)·a + πμ·min(0, a − (1 − π)Δ)   +   n(b)·b + (1 − π)μ·min(0, b − πΔ),

noise flow pays the edge, and informed flow costs the distance to V whenever the
quote is inside V. The (myopic) monopoly quote maximizes each side. Both are
tabulated on a grid of beliefs and interpolated.

Agents
------
- ``GMQuoter``: posts the zero-profit quotes at the public belief (Glosten &
  Milgrom 1985 with elastic noise). It earns no rents and loses nothing on
  average.
- ``FixedSpread``: mid ± a fixed half-spread ("tight" and "wide").
- ``InventorySkew``: Avellaneda & Stoikov (2008) reservation price
  r = m − q·γ·σ²·(T − t) with a fixed half-spread around r; σ² = Δ²/(4T) spreads
  the prior variance of V evenly over the episode.
- ``Undercutter``: quotes one tick inside the best rival's last quote on each
  side while that stays at or above its zero-profit quote, and never wider than
  the monopoly quote (which is what it posts when it has no rival).
- ``MarkoutQuoter`` (this project's design): the undercutter's competitive logic,
  but its floor is the zero-profit quote plus the Avellaneda–Stoikov
  indifference adjustment with the Bayesian conditional variance
  Var_t(V) = π(1 − π)Δ² in place of σ²(T − t): it sells no lower than
  E[V | buy] + (1 − 2q)γVar_t(V)/2 and buys no higher than
  E[V | sell] − (1 + 2q)γVar_t(V)/2. Alone, it posts the monopoly edges around
  its reservation price m − qγVar_t(V).

Quote revisions are fast relative to arrivals: before each arrival the reactive
makers (undercutter, markout) revise simultaneously, each responding to its
rivals' previous-round quotes, until nobody moves. The response rule is monotone
and its fixed point is unique, so the outcome is the Bertrand price on the tick
grid given everyone's floors and caps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Protocol, Sequence

import numpy as np

from markout.games.stats import mean_ci, ratio_ci

BIG = np.int64(1) << 40   # "no rival" sentinel for the undercutting rule (in ticks)


# ---------------------------------------------------------------------------
# Market and the belief-indexed quote tables
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Market:
    """Parameters of one tournament market (prices in dollars)."""

    v_low: float = 99.0
    v_high: float = 101.0
    prior: float = 0.5
    mu: float = 0.2            # probability an arrival is informed
    c_max: float = 1.2         # noise urgency c ~ U[0, c_max]
    tick: float = 0.01
    n_periods: int = 100       # T, arrivals per episode

    def __post_init__(self) -> None:
        if not self.v_high > self.v_low:
            raise ValueError("need v_high > v_low")
        if not 0 < self.mu < 1 or not 0 < self.prior < 1:
            raise ValueError("need 0 < mu < 1 and 0 < prior < 1")
        if not self.c_max > 0 or not self.tick > 0 or self.n_periods < 1:
            raise ValueError("need c_max > 0, tick > 0 and n_periods >= 1")

    @property
    def dv(self) -> float:
        return self.v_high - self.v_low

    def noise_trade_prob(self, e):
        """q(e) = P(c > e) for c ~ U[0, c_max] (1 for a negative edge)."""
        return np.clip(1.0 - np.asarray(e, dtype=float) / self.c_max, 0.0, 1.0)

    def noise_rate(self, e):
        """n(e) = (1 − μ)q(e)/2: probability that a period brings a noise buy at ask edge e."""
        return 0.5 * (1 - self.mu) * self.noise_trade_prob(e)

    @cached_property
    def tables(self) -> "QuoteTables":
        return QuoteTables.build(self)


def zero_profit_edge(pi, mkt: Market, side: str = "ask") -> np.ndarray:
    """Zero-profit edge on one side at beliefs ``pi`` (NaN where none exists).

    With w = π for the ask (1 − π for the bid) and k = Δπ(1 − π)μ, the condition
    e = k/(wμ + n(e)) with linear demand n(e) = (1 − μ)(1 − e/c_max)/2 is the
    quadratic A·e² − B·e + k = 0, A = (1 − μ)/(2c_max), B = wμ + (1 − μ)/2. Its
    smaller root is the competitive (GM) quote, the least fixed point. If the
    discriminant is negative, or the root is not below c_max (no noise trader
    would trade there), the side breaks down and NaN is returned.
    """
    pi = np.asarray(pi, dtype=float)
    w = pi if side == "ask" else 1 - pi
    k = mkt.dv * pi * (1 - pi) * mkt.mu
    a_ = (1 - mkt.mu) / (2 * mkt.c_max)
    b_ = w * mkt.mu + (1 - mkt.mu) / 2
    disc = b_ * b_ - 4 * a_ * k
    with np.errstate(invalid="ignore"):
        # 2k/(B + √disc) is the smaller root, written to avoid cancellation when k is small
        e = np.where(disc >= 0, 2 * k / (b_ + np.sqrt(np.maximum(disc, 0.0))), np.nan)
    return np.where(e < mkt.c_max, e, np.nan)


def zero_profit_edges(pi, mkt: Market) -> tuple[np.ndarray, np.ndarray]:
    """Zero-profit (ask edge, bid edge) at beliefs ``pi``."""
    return zero_profit_edge(pi, mkt, "ask"), zero_profit_edge(pi, mkt, "bid")


def zero_profit_half_spread_at_half(mu: float, c_max: float, dv: float) -> float:
    """Closed form of the zero-profit half-spread at π = ½ with elastic noise.

    At π = ½, h(2μ + 2(1 − μ)(1 − h/c_max)) = μΔ, whose smaller root is
    h* = c_max·(1 − √(1 − 2μ(1 − μ)Δ/c_max)) / (2(1 − μ)). As c_max → ∞ this tends
    to μΔ/2, the GM half-spread. Returns NaN if no root exists (market breakdown).
    """
    disc = 1 - 2 * mu * (1 - mu) * dv / c_max
    if disc < 0:
        return float("nan")
    h = c_max * (1 - np.sqrt(disc)) / (2 * (1 - mu))
    return float(h) if h < c_max else float("nan")


def side_profit(e, pi, mkt: Market, side: str = "ask"):
    """Expected profit per period from one side of a lone maker's quote.

    Ask: n(a)·a + πμ·min(0, a − (1 − π)Δ); bid: n(b)·b + (1 − π)μ·min(0, b − πΔ).
    """
    w = pi if side == "ask" else 1 - pi
    return mkt.noise_rate(e) * e + w * mkt.mu * np.minimum(0.0, e - (1 - w) * mkt.dv)


def one_period_profit(a, b, pi, mkt: Market):
    """Expected profit per period of a lone maker quoting edges (a, b) at belief π."""
    return side_profit(a, pi, mkt, "ask") + side_profit(b, pi, mkt, "bid")


def monopoly_edge(pi, mkt: Market, side: str = "ask", de: float = 0.0005,
                  chunk: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Myopic monopoly edge on one side and its one-period profit (grid search)."""
    pi = np.atleast_1d(np.asarray(pi, dtype=float))
    es = np.arange(0.0, mkt.c_max + de, de)
    best_e = np.empty_like(pi)
    best_v = np.empty_like(pi)
    for s in range(0, pi.size, chunk):
        v = side_profit(es[None, :], pi[s:s + chunk, None], mkt, side)
        j = v.argmax(axis=1)
        best_e[s:s + chunk] = es[j]
        best_v[s:s + chunk] = v[np.arange(j.size), j]
    return best_e, best_v


@dataclass(frozen=True)
class QuoteTables:
    """Zero-profit and monopoly edges tabulated on a belief grid."""

    grid: np.ndarray
    be_ask: np.ndarray
    be_bid: np.ndarray
    mono_ask: np.ndarray
    mono_bid: np.ndarray
    mono_profit: np.ndarray   # one-period profit of the monopoly quote

    @classmethod
    def build(cls, mkt: Market, n_grid: int = 2001) -> "QuoteTables":
        grid = np.linspace(0.0, 1.0, n_grid)
        a, b = zero_profit_edges(grid, mkt)
        if np.isnan(a).any() or np.isnan(b).any():
            bad = grid[np.isnan(a) | np.isnan(b)]
            raise ValueError(f"no zero-profit quote for beliefs in [{bad.min():.3f}, "
                             f"{bad.max():.3f}]: the market breaks down; raise c_max or lower mu")
        ma, va = monopoly_edge(grid, mkt, "ask")
        mb, vb = monopoly_edge(grid, mkt, "bid")
        return cls(grid, a, b, ma, mb, va + vb)

    def lookup(self, pi: np.ndarray) -> tuple[np.ndarray, ...]:
        f = lambda y: np.interp(pi, self.grid, y)  # noqa: E731
        return f(self.be_ask), f(self.be_bid), f(self.mono_ask), f(self.mono_bid)


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

@dataclass
class Context:
    """What a maker sees when it quotes in period t (arrays over seeds)."""

    mkt: Market
    t: int
    pi: np.ndarray
    mid: np.ndarray
    q: np.ndarray          # this maker's inventory
    be_ask: np.ndarray     # zero-profit edges at the public belief
    be_bid: np.ndarray
    mono_ask: np.ndarray   # monopoly edges at the public belief
    mono_bid: np.ndarray

    @property
    def var_v(self) -> np.ndarray:
        """Var(V | tape) = π(1 − π)Δ²."""
        return self.pi * (1 - self.pi) * self.mkt.dv ** 2

    def ask_ticks(self, price) -> np.ndarray:
        """Round an ask up to the tick grid (away from the mid)."""
        return np.ceil(np.asarray(price) / self.mkt.tick - 1e-7).astype(np.int64)

    def bid_ticks(self, price) -> np.ndarray:
        """Round a bid down to the tick grid (away from the mid)."""
        return np.floor(np.asarray(price) / self.mkt.tick + 1e-7).astype(np.int64)


class Agent(Protocol):
    """A tournament entrant.

    Non-reactive agents (``reactive = False``) implement ``quote(ctx)``, which
    returns (ask, bid) in ticks. Reactive agents implement ``bounds(ctx)``,
    which returns (ask floor, ask cap, bid cap, bid floor) in ticks; the engine
    then applies the undercutting rule between those bounds.
    """

    name: str
    reactive: bool


@dataclass
class FixedSpread:
    """Quote mid ± h, whatever the belief or the flow."""

    name: str
    half_spread: float
    reactive: bool = field(default=False, init=False)

    def quote(self, c: Context) -> tuple[np.ndarray, np.ndarray]:
        return c.ask_ticks(c.mid + self.half_spread), c.bid_ticks(c.mid - self.half_spread)


@dataclass
class GMQuoter:
    """Zero-profit quotes ask = E[V | buy], bid = E[V | sell] (Glosten–Milgrom)."""

    name: str = "gm"
    reactive: bool = field(default=False, init=False)

    def quote(self, c: Context) -> tuple[np.ndarray, np.ndarray]:
        return c.ask_ticks(c.mid + c.be_ask), c.bid_ticks(c.mid - c.be_bid)


@dataclass
class InventorySkew:
    """Avellaneda–Stoikov reservation price r = m − q·γ·σ²·(T − t), mid ± h around r."""

    name: str = "inventory"
    half_spread: float = 0.25
    gamma: float = 0.02
    sigma2: float | None = None   # None → Δ²/(4T): prior Var(V) spread evenly over T
    reactive: bool = field(default=False, init=False)

    def quote(self, c: Context) -> tuple[np.ndarray, np.ndarray]:
        T = c.mkt.n_periods
        s2 = self.sigma2 if self.sigma2 is not None else c.mkt.dv ** 2 / (4 * T)
        r = c.mid - c.q * self.gamma * s2 * (T - c.t)
        return c.ask_ticks(r + self.half_spread), c.bid_ticks(r - self.half_spread)


@dataclass
class Undercutter:
    """One tick inside the best rival, floored at zero profit, capped at monopoly.

    ``bounds`` returns (ask floor, ask cap, bid cap, bid floor) in ticks, where
    the floors are the tightest acceptable quotes (+EV under the belief, rounded
    away from the mid) and the caps are the monopoly quotes.
    """

    name: str = "undercut"
    reactive: bool = field(default=True, init=False)

    def bounds(self, c: Context) -> tuple[np.ndarray, ...]:
        return (c.ask_ticks(c.mid + c.be_ask), c.ask_ticks(c.mid + c.mono_ask),
                c.bid_ticks(c.mid - c.mono_bid), c.bid_ticks(c.mid - c.be_bid))


@dataclass
class MarkoutQuoter:
    """Bayesian (GM) floors + Avellaneda–Stoikov inventory terms + Bertrand undercutting.

    Uses Var_t(V) = π(1 − π)Δ² from the public belief as the inventory risk, so
    the skew fades as the tape reveals V rather than linearly in time.
    """

    name: str = "markout"
    gamma: float = 0.02
    reactive: bool = field(default=True, init=False)

    def bounds(self, c: Context) -> tuple[np.ndarray, ...]:
        g = self.gamma * c.var_v
        r = c.mid - c.q * g                                     # reservation price
        ask_floor = c.mid + c.be_ask + (1 - 2 * c.q) * g / 2    # E[V | buy] + AS premium
        bid_floor = c.mid - c.be_bid - (1 + 2 * c.q) * g / 2    # E[V | sell] − AS premium
        return (c.ask_ticks(ask_floor), c.ask_ticks(r + c.mono_ask),
                c.bid_ticks(r - c.mono_bid), c.bid_ticks(bid_floor))


def default_agents() -> list[Agent]:
    """The six tournament entrants with the parameters used in the report."""
    return [GMQuoter("gm"), FixedSpread("tight", 0.05), FixedSpread("wide", 0.60),
            InventorySkew("inventory", half_spread=0.25, gamma=0.02),
            Undercutter("undercut"), MarkoutQuoter("markout", gamma=0.02)]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

@dataclass
class ArenaRun:
    """Per-seed outcomes of one tournament configuration."""

    mkt: Market
    names: list[str]
    value_high: np.ndarray        # (S,)
    pnl: np.ndarray               # (K, S) marked to V
    fills: np.ndarray             # (K, S)
    informed_fills: np.ndarray    # (K, S)
    edge: np.ndarray              # (K, S) Σ d·(price − m_t) over fills
    adverse: np.ndarray           # (K, S) Σ d·(V − m_t) over fills (markout to V)
    q2: np.ndarray                # (K, S) time-average of inventory²
    q_final: np.ndarray           # (K, S)
    trades: np.ndarray            # (S,)
    informed_trades: np.ndarray   # (S,)
    noise_trades: np.ndarray      # (S,)
    noise_arrivals: np.ndarray    # (S,)
    noise_pnl: np.ndarray         # (S,) noise traders' trading PnL marked to V
    noise_surplus: np.ndarray     # (S,) Σ (urgency c + trading PnL) over noise trades
    informed_pnl: np.ndarray      # (S,)
    quoted_spread: np.ndarray     # (S,) time-average inside spread
    zero_profit_spread: np.ndarray  # (S,) time-average of a* + b* at the public belief
    eff_half: np.ndarray          # (S,) Σ d·(price − m_t) over all trades
    spread_path: np.ndarray       # (T,) mean inside spread across seeds
    zero_profit_path: np.ndarray  # (T,) mean zero-profit spread across seeds
    belief_truth_path: np.ndarray  # (T+1,) mean P(V = realized V) across seeds
    final_belief: np.ndarray      # (S,) public P(V = V_H) after the last period
    fill_path: np.ndarray         # (K, T) mean fills per seed in each period
    informed_fill_path: np.ndarray  # (K, T) mean informed fills per seed in each period
    edge_path: np.ndarray         # (K, T) mean Σ d·(price − m_t) per seed in each period
    adverse_path: np.ndarray      # (K, T) mean Σ d·(V − m_t) per seed in each period
    rounds_max: int               # most revision rounds any period needed

    @property
    def n_seeds(self) -> int:
        return self.pnl.shape[1]

    def agent_summary(self, i: int) -> dict:
        """PnL, fill share, per-fill economics and inventory risk of maker i."""
        return {
            "name": self.names[i],
            "pnl": mean_ci(self.pnl[i]),
            "pnl_sd": float(self.pnl[i].std(ddof=1)),
            "fill_share": ratio_ci(self.fills[i], self.trades),
            "fills_per_episode": float(self.fills[i].mean()),
            "pnl_per_fill": ratio_ci(self.pnl[i], self.fills[i]),
            "edge_per_fill": ratio_ci(self.edge[i], self.fills[i]),
            "adverse_per_fill": ratio_ci(self.adverse[i], self.fills[i]),
            "informed_share_of_fills": ratio_ci(self.informed_fills[i], self.fills[i]),
            "inventory_sq": mean_ci(self.q2[i]),
            "final_inventory_sd": float(self.q_final[i].std(ddof=1)),
        }

    def market_summary(self) -> dict:
        """Spreads, maker profit per trade and noise-trader welfare for the market."""
        T = self.mkt.n_periods
        maker = self.pnl.sum(axis=0)
        return {
            "quoted_spread": mean_ci(self.quoted_spread),
            "effective_spread": ratio_ci(2 * self.eff_half, self.trades),
            "zero_profit_spread": mean_ci(self.zero_profit_spread),
            "excess_spread": mean_ci(self.quoted_spread - self.zero_profit_spread),
            "maker_pnl": mean_ci(maker),
            "maker_pnl_per_trade": ratio_ci(maker, self.trades),
            "trades_per_episode": mean_ci(self.trades),
            "noise_participation": ratio_ci(self.noise_trades, self.noise_arrivals),
            "noise_welfare_per_period": mean_ci(self.noise_surplus / T),
            "noise_cost_per_trade": ratio_ci(-self.noise_pnl, self.noise_trades),
            "informed_pnl": mean_ci(self.informed_pnl),
            "informed_share_of_trades": ratio_ci(self.informed_trades, self.trades),
            "zero_sum_max_error": float(np.abs(maker + self.noise_pnl + self.informed_pnl).max()),
        }


def _min_excluding_self(x: np.ndarray) -> np.ndarray:
    """For each row i, the column-wise minimum over the other rows (BIG if none)."""
    k = x.shape[0]
    if k == 1:
        return np.full_like(x, BIG)
    m1 = x.min(axis=0)
    at_min = x == m1
    unique_min = at_min.sum(axis=0) == 1
    m2 = np.where(at_min, BIG, x).min(axis=0)
    return np.where(at_min & unique_min, m2, m1)


def _pick(at_best: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Index of a uniformly chosen True row in each column (ties split at random)."""
    n = at_best.sum(axis=0)
    k = np.floor(u * n).astype(np.int64)
    return np.argmax(np.cumsum(at_best, axis=0) > k[None, :], axis=0)


def _informed_side(v: float | np.ndarray, ask: np.ndarray, bid: np.ndarray):
    """Whether an informed trader who knows V buys / sells at the best quotes."""
    gain_buy = v - ask
    gain_sell = bid - v
    buys = (gain_buy > 0) & (gain_buy >= gain_sell)
    sells = (gain_sell > 0) & (gain_sell > gain_buy)
    return buys, sells


def simulate(mkt: Market, agents: Sequence[Agent], n_seeds: int = 1000, seed: int = 0,
             max_rounds: int = 5000) -> ArenaRun:
    """Run one tournament configuration over ``n_seeds`` independent episodes.

    The exogenous randomness (V, who arrives, noise direction and urgency, tie
    breaks) depends only on ``seed``, so different line-ups face the same flow
    (common random numbers), which sharpens paired comparisons.
    """
    S, K, T = n_seeds, len(agents), mkt.n_periods
    if len({a.name for a in agents}) != K:
        raise ValueError("agent names must be unique")
    rng = np.random.default_rng(seed)
    value_high = rng.random(S) < mkt.prior
    informed = rng.random((T, S)) < mkt.mu
    noise_buy = rng.random((T, S)) < 0.5
    urgency = rng.random((T, S)) * mkt.c_max
    tie_u = rng.random((T, S))

    V = np.where(value_high, mkt.v_high, mkt.v_low)
    tab = mkt.tables
    reactive = [i for i, a in enumerate(agents) if a.reactive]

    pi = np.full(S, mkt.prior)
    q = np.zeros((K, S))
    cash = np.zeros((K, S))
    fills = np.zeros((K, S))
    inf_fills = np.zeros((K, S))
    edge = np.zeros((K, S))
    adverse = np.zeros((K, S))
    q2 = np.zeros((K, S))
    trades = np.zeros(S)
    inf_trades = np.zeros(S)
    noise_trades = np.zeros(S)
    noise_arr = np.zeros(S)
    noise_pnl = np.zeros(S)
    noise_surplus = np.zeros(S)
    inf_pnl = np.zeros(S)
    quoted = np.zeros(S)
    zp = np.zeros(S)
    eff_half = np.zeros(S)
    spread_path = np.zeros(T)
    zp_path = np.zeros(T)
    fill_path = np.zeros((K, T))
    inf_fill_path = np.zeros((K, T))
    edge_path = np.zeros((K, T))
    adverse_path = np.zeros((K, T))
    truth_path = np.zeros(T + 1)
    truth_path[0] = np.where(value_high, pi, 1 - pi).mean()
    prev_ask_edge = np.zeros((K, S))
    prev_bid_edge = np.zeros((K, S))
    rounds_max = 0
    rows = np.arange(K)[:, None]

    for t in range(T):
        mid = mkt.v_low + pi * mkt.dv
        be_a, be_b, mo_a, mo_b = tab.lookup(pi)
        asks = np.empty((K, S), dtype=np.int64)
        bids = np.empty((K, S), dtype=np.int64)
        lo_a = np.zeros((K, S), dtype=np.int64)
        hi_a = np.zeros((K, S), dtype=np.int64)
        lo_b = np.zeros((K, S), dtype=np.int64)
        hi_b = np.zeros((K, S), dtype=np.int64)
        for i, ag in enumerate(agents):
            c = Context(mkt, t, pi, mid, q[i], be_a, be_b, mo_a, mo_b)
            if ag.reactive:
                fa, ca, cb, fb = ag.bounds(c)
                lo_a[i], hi_a[i] = fa, np.maximum(fa, ca)
                lo_b[i], hi_b[i] = np.minimum(cb, fb), fb
                if t == 0:   # no last quotes yet: start from the monopoly quote
                    asks[i], bids[i] = hi_a[i], lo_b[i]
                else:        # start from the last quote, re-centred on the new mid
                    asks[i] = np.clip(c.ask_ticks(mid + prev_ask_edge[i]), lo_a[i], hi_a[i])
                    bids[i] = np.clip(c.bid_ticks(mid - prev_bid_edge[i]), lo_b[i], hi_b[i])
            else:
                asks[i], bids[i] = ag.quote(c)

        # simultaneous best-response revisions until nobody moves
        if reactive:
            for r in range(max_rounds):
                rival_ask = _min_excluding_self(asks)
                rival_bid = -_min_excluding_self(-bids)
                new_a = np.clip(rival_ask[reactive] - 1, lo_a[reactive], hi_a[reactive])
                new_b = np.clip(rival_bid[reactive] + 1, lo_b[reactive], hi_b[reactive])
                if np.array_equal(new_a, asks[reactive]) and np.array_equal(new_b, bids[reactive]):
                    rounds_max = max(rounds_max, r)
                    break
                asks[reactive], bids[reactive] = new_a, new_b
            else:
                raise RuntimeError("quote revisions did not converge")

        best_ask = asks.min(axis=0)
        best_bid = bids.max(axis=0)
        ask_p = best_ask * mkt.tick
        bid_p = best_bid * mkt.tick
        ask_edge = ask_p - mid          # what a noise buyer pays over the public mid
        bid_edge = mid - bid_p

        # the arriving trader
        inf = informed[t]
        ib, isl = _informed_side(V, ask_p, bid_p)
        noise_go = ~inf & (urgency[t] > np.where(noise_buy[t], ask_edge, bid_edge))
        buy = (inf & ib) | (noise_go & noise_buy[t])
        sell = (inf & isl) | (noise_go & ~noise_buy[t])
        trade = buy | sell
        d = np.where(buy, 1.0, np.where(sell, -1.0, 0.0))   # trader's direction
        price = np.where(buy, ask_p, bid_p)

        win_ask = _pick(asks == best_ask[None, :], tie_u[t])
        win_bid = _pick(bids == best_bid[None, :], tie_u[t])
        winner = np.where(buy, win_ask, win_bid)
        hit = (rows == winner[None, :]) & trade[None, :]      # (K, S) one-hot fill
        q -= hit * d
        cash += hit * (d * price)
        fills += hit
        inf_fills += hit & inf[None, :]
        fill_edge = hit * (d * (price - mid))
        fill_adverse = hit * (d * (V - mid))
        edge += fill_edge
        adverse += fill_adverse
        fill_path[:, t] = hit.mean(axis=1)
        inf_fill_path[:, t] = (hit & inf[None, :]).mean(axis=1)
        edge_path[:, t] = fill_edge.mean(axis=1)
        adverse_path[:, t] = fill_adverse.mean(axis=1)

        trader_gain = d * (V - price)
        trades += trade
        inf_trades += trade & inf
        noise_trades += noise_go
        noise_arr += ~inf
        noise_pnl += np.where(noise_go, trader_gain, 0.0)
        noise_surplus += np.where(noise_go, urgency[t] + trader_gain, 0.0)
        inf_pnl += np.where(inf & trade, trader_gain, 0.0)
        quoted += ask_p - bid_p
        zp += be_a + be_b
        eff_half += d * (price - mid)
        spread_path[t] = (ask_p - bid_p).mean()
        zp_path[t] = (be_a + be_b).mean()

        # public Bayesian update on the observed outcome (buy, sell or no trade)
        n_buy = mkt.noise_rate(ask_edge)
        n_sell = mkt.noise_rate(bid_edge)
        hb, hs = _informed_side(mkt.v_high, ask_p, bid_p)
        lb, ls = _informed_side(mkt.v_low, ask_p, bid_p)
        none_h = 1 - mkt.mu * (hb | hs) - n_buy - n_sell
        none_l = 1 - mkt.mu * (lb | ls) - n_buy - n_sell
        like_h = np.where(buy, mkt.mu * hb + n_buy, np.where(sell, mkt.mu * hs + n_sell, none_h))
        like_l = np.where(buy, mkt.mu * lb + n_buy, np.where(sell, mkt.mu * ls + n_sell, none_l))
        pi = pi * like_h / (pi * like_h + (1 - pi) * like_l)
        truth_path[t + 1] = np.where(value_high, pi, 1 - pi).mean()

        prev_ask_edge = asks * mkt.tick - mid[None, :]
        prev_bid_edge = mid[None, :] - bids * mkt.tick
        q2 += q * q

    pnl = cash + q * V[None, :]
    return ArenaRun(mkt, [a.name for a in agents], value_high, pnl, fills, inf_fills,
                    edge, adverse, q2 / T, q.copy(), trades, inf_trades, noise_trades,
                    noise_arr, noise_pnl, noise_surplus, inf_pnl, quoted / T, zp / T,
                    eff_half, spread_path, zp_path, truth_path, pi.copy(), fill_path,
                    inf_fill_path, edge_path, adverse_path, rounds_max)


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def clone(agent, name: str):
    """A copy of ``agent`` under a new name (for line-ups of identical makers)."""
    import copy

    a = copy.copy(agent)
    a.name = name
    return a


def head_to_head(run: ArenaRun) -> dict:
    """Paired PnL difference of the first two makers (same seeds, same flow)."""
    diff = run.pnl[0] - run.pnl[1]
    ci = mean_ci(diff)
    if ci["lo"] > 0:
        winner = run.names[0]
    elif ci["hi"] < 0:
        winner = run.names[1]
    else:
        winner = "draw"
    return {"a": run.names[0], "b": run.names[1], "diff": ci, "winner": winner}


def replicator(payoff: np.ndarray, x0: np.ndarray | None = None, dt: float = 0.05,
               n_steps: int = 4000) -> np.ndarray:
    """Discrete-time replicator dynamics ẋ_i = x_i((Ax)_i − xᵀAx).

    ``payoff[i, j]`` is strategy i's mean payoff against j. Uses the exact
    exponential-weights update x_i ← x_i·exp(dt·(Ax)_i)/Z, which stays on the
    simplex and is the replicator equation in the limit dt → 0. Returns the path
    of shares, shape (n_steps + 1, n).
    """
    n = payoff.shape[0]
    x = np.full(n, 1.0 / n) if x0 is None else np.asarray(x0, dtype=float)
    path = np.empty((n_steps + 1, n))
    path[0] = x
    for s in range(n_steps):
        f = payoff @ x
        w = x * np.exp(dt * (f - f.max()))
        x = w / w.sum()
        path[s + 1] = x
    return path
