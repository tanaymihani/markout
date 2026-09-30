"""Kuhn poker solved by counterfactual regret minimization (CFR).

References
----------
- Kuhn, H. W. (1950), "A simplified two-person poker", *Contributions to the
  Theory of Games* I, 97–103 (the game and its equilibrium family).
- Zinkevich, M., Johanson, M., Bowling, M. and Piccione, C. (2007), "Regret
  minimization in games with incomplete information", *NIPS* 20 (CFR).
- Neller, T. W. and Lanctot, M. (2013), *An Introduction to Counterfactual
  Regret Minimization* (the tutorial this implementation follows).
- Tammelin, O. (2014), "Solving large imperfect information games using CFR+",
  arXiv:1407.5042 (CFR+, used only as a comparison curve).

The game
--------
A deck of three cards, J < Q < K. Each player antes 1 and is dealt one card.
Player 1 (P1) checks ("p", pass) or bets 1 ("b"). After a check, P2 checks (show
down for 1) or bets; P1 then folds ("p") or calls ("b", show down for 2). After a
bet, P2 folds or calls (show down for 2). Terminal histories and P1's payoff:

    pp   ±1 (higher card wins)      bp   +1 (P2 folds)
    bb   ±2                        pbp  −1 (P1 folds)
    pbb  ±2

An information set is the player's card plus the public history, written like
"Jpb" (P1 holding the Jack after check–bet). There are 12. A strategy maps each
information set to the probabilities of (pass, bet).

Equilibrium (Kuhn 1950): the game value is −1/18 for P1. P1 bets the Jack with
some α ∈ [0, 1/3], always checks the Queen, bets the King with 3α, and after
check–bet calls with the Queen with α + 1/3 (folds J, calls K). P2's strategy is
unique: bet K, bluff J 1/3 of the time after a check, check Q; call a bet with K,
with Q 1/3 of the time, never with J.

Vanilla CFR
-----------
Each iteration t computes the current strategy σᵗ by regret matching (play each
action in proportion to its positive cumulative regret; uniform if none), then
walks the full game tree for all six deals with σᵗ fixed, adding to every
information set I the counterfactual regret

    r(I, a) = π₋ᵢ(h)·(u(h·a) − u(h)),

weighted by the opponent's (and chance's) reach π₋ᵢ, and adding σᵗ(I) weighted
by the player's own reach to the strategy sum. The *average* strategy converges
to a Nash equilibrium; the current one need not.

Exploitability
--------------
With an exact best-response value BRᵢ(σ₋ᵢ) for each player,
exploitability(σ) = BR₁(σ₂) + BR₂(σ₁) ≥ 0, with equality exactly at an
equilibrium (it is the sum of both players' best-response gains; some papers
report half of it).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CARDS = "JQK"
ACTIONS = "pb"
DEALS = [(i, j) for i in range(3) for j in range(3) if i != j]
TERMINAL = frozenset({"pp", "bp", "bb", "pbp", "pbb"})
P1_INFOSETS = [c + h for h in ("", "pb") for c in CARDS]
P2_INFOSETS = [c + h for h in ("p", "b") for c in CARDS]
INFOSETS = P1_INFOSETS + P2_INFOSETS
GAME_VALUE = -1.0 / 18.0

Strategy = dict[str, np.ndarray]   # infoset -> [P(pass), P(bet)]


def payoff_p1(c1: int, c2: int, h: str) -> float:
    """Player 1's payoff at terminal history ``h`` with cards c1 (P1) and c2 (P2)."""
    if h == "bp":
        return 1.0
    if h == "pbp":
        return -1.0
    stake = 2.0 if h in ("bb", "pbb") else 1.0
    return stake if c1 > c2 else -stake


def to_act(h: str) -> int:
    """0 if P1 acts at history h, 1 if P2 does."""
    return len(h) % 2


def uniform_strategy() -> Strategy:
    return {k: np.array([0.5, 0.5]) for k in INFOSETS}


def analytic_equilibrium(alpha: float = 1 / 6) -> Strategy:
    """A member of Kuhn's equilibrium family, indexed by P1's Jack-bet rate α ∈ [0, 1/3]."""
    if not 0 <= alpha <= 1 / 3:
        raise ValueError("alpha must lie in [0, 1/3]")
    bet = {"J": alpha, "Q": 0.0, "K": 3 * alpha,
           "Jpb": 0.0, "Qpb": alpha + 1 / 3, "Kpb": 1.0,
           "Jp": 1 / 3, "Qp": 0.0, "Kp": 1.0,
           "Jb": 0.0, "Qb": 1 / 3, "Kb": 1.0}
    return {k: np.array([1 - v, v]) for k, v in bet.items()}


def combine(p1_from: Strategy, p2_from: Strategy) -> Strategy:
    """Profile that plays P1's infosets from ``p1_from`` and P2's from ``p2_from``."""
    out = {k: np.asarray(p1_from[k], dtype=float) for k in P1_INFOSETS}
    out.update({k: np.asarray(p2_from[k], dtype=float) for k in P2_INFOSETS})
    return out


# ---------------------------------------------------------------------------
# Evaluation: expected value and exact best responses
# ---------------------------------------------------------------------------

def _ev(c1: int, c2: int, h: str, s: Strategy) -> float:
    if h in TERMINAL:
        return payoff_p1(c1, c2, h)
    p = s[CARDS[(c1, c2)[to_act(h)]] + h]
    return p[0] * _ev(c1, c2, h + "p", s) + p[1] * _ev(c1, c2, h + "b", s)


def expected_value(s: Strategy) -> float:
    """Player 1's expected payoff under profile ``s`` (averaged over the 6 deals)."""
    return sum(_ev(c1, c2, "", s) for c1, c2 in DEALS) / len(DEALS)


def _br(card: int, h: str, w: dict[int, float], s: Strategy, player: int,
        br: Strategy, default: Strategy | None, tol: float) -> float:
    """Best-response value from history h for ``player`` holding ``card``.

    ``w[c']`` is P(opponent holds c' | card) times the opponent's probability of
    having played its part of h with c'. The best responder's information set at
    h is (card, h), so choosing the better action here, against the w-weighted
    continuation values, is exactly a best response. When the two actions are
    worth the same (within ``tol``, which includes infosets the opponent never
    lets us reach) and a ``default`` strategy is given, the default mix is kept.
    """
    if h in TERMINAL:
        if player == 0:
            return sum(wc * payoff_p1(card, c, h) for c, wc in w.items())
        return sum(-wc * payoff_p1(c, card, h) for c, wc in w.items())
    if to_act(h) == player:
        vals = [_br(card, h + a, w, s, player, br, default, tol) for a in ACTIONS]
        key = CARDS[card] + h
        if default is not None and abs(vals[1] - vals[0]) <= tol:
            mix = np.asarray(default[key], dtype=float)
            br[key] = mix.copy()
            return float(mix[0] * vals[0] + mix[1] * vals[1])
        best = int(vals[1] > vals[0])          # exact ties go to pass
        br[key] = np.eye(2)[best]
        return vals[best]
    total = 0.0
    for i, a in enumerate(ACTIONS):
        w2 = {c: wc * s[CARDS[c] + h][i] for c, wc in w.items()}
        total += _br(card, h + a, w2, s, player, br, default, tol)
    return total


def best_response(s: Strategy, player: int, default: Strategy | None = None,
                  tol: float = 1e-12) -> tuple[float, Strategy]:
    """Exact best response of ``player`` (0 = P1, 1 = P2) to the other side of ``s``.

    Returns (value to ``player``, a best-response strategy for its six
    information sets). Best responses are not unique; by default ties go to
    "pass", and with ``default`` given, ties keep the default's mix, which gives
    the best response that departs from ``default`` only where that pays.
    """
    br: Strategy = {}
    total = 0.0
    for card in range(3):
        w = {c: 0.5 for c in range(3) if c != card}
        total += _br(card, "", w, s, player, br, default, tol) / 3
    return total, br


def exploitability(s: Strategy) -> float:
    """BR₁(σ₂) + BR₂(σ₁): zero exactly at a Nash equilibrium."""
    return best_response(s, 0)[0] + best_response(s, 1)[0]


# ---------------------------------------------------------------------------
# CFR
# ---------------------------------------------------------------------------

def regret_matching(r: np.ndarray) -> np.ndarray:
    """σ(a) ∝ max(R(a), 0); uniform when no regret is positive."""
    pos = np.maximum(r, 0.0)
    tot = pos.sum()
    return pos / tot if tot > 0 else np.full(r.size, 1.0 / r.size)


@dataclass
class KuhnCFR:
    """Vanilla CFR (default) or CFR+ on Kuhn poker, with full-tree traversal.

    ``plus=True`` switches to CFR+ (Tammelin 2014): regrets are floored at zero
    after every update, players update alternately, and the average strategy
    weights iteration t by t.
    """

    plus: bool = False
    regret: dict = field(default_factory=lambda: {k: np.zeros(2) for k in INFOSETS})
    strategy_sum: dict = field(default_factory=lambda: {k: np.zeros(2) for k in INFOSETS})
    iterations: int = 0

    def current_strategy(self) -> Strategy:
        return {k: regret_matching(r) for k, r in self.regret.items()}

    def average_strategy(self) -> Strategy:
        out = {}
        for k, ssum in self.strategy_sum.items():
            tot = ssum.sum()
            out[k] = ssum / tot if tot > 0 else np.array([0.5, 0.5])
        return out

    def _walk(self, cards: tuple[int, int], h: str, reach: tuple[float, float],
              sigma: Strategy, update: tuple[int, ...], d_regret: dict, d_ssum: dict,
              weight: float) -> float:
        """Return P1's expected utility below h; accumulate regret and strategy sums."""
        if h in TERMINAL:
            return payoff_p1(cards[0], cards[1], h)
        p = to_act(h)
        key = CARDS[cards[p]] + h
        s = sigma[key]
        u = [0.0, 0.0]
        for i, a in enumerate(ACTIONS):
            r = (reach[0] * s[i], reach[1]) if p == 0 else (reach[0], reach[1] * s[i])
            u[i] = self._walk(cards, h + a, r, sigma, update, d_regret, d_ssum, weight)
        node = s[0] * u[0] + s[1] * u[1]
        if p in update:
            sign = 1.0 if p == 0 else -1.0             # P2's utility is −u
            cf = reach[1 - p] / len(DEALS)             # opponent reach × chance
            d_regret[key] += cf * sign * np.array([u[0] - node, u[1] - node])
            d_ssum[key] += weight * reach[p] * s
        return node

    def _pass(self, update: tuple[int, ...], weight: float) -> None:
        sigma = self.current_strategy()
        d_regret = {k: np.zeros(2) for k in INFOSETS}
        d_ssum = {k: np.zeros(2) for k in INFOSETS}
        for cards in DEALS:
            self._walk(cards, "", (1.0, 1.0), sigma, update, d_regret, d_ssum, weight)
        for k in INFOSETS:
            self.regret[k] += d_regret[k]
            if self.plus:
                np.maximum(self.regret[k], 0.0, out=self.regret[k])
            self.strategy_sum[k] += d_ssum[k]

    def iterate(self) -> None:
        """One CFR iteration (both players' regrets updated)."""
        self.iterations += 1
        if self.plus:
            for player in (0, 1):
                self._pass((player,), float(self.iterations))
        else:
            self._pass((0, 1), 1.0)

    def run(self, n_iter: int, checkpoints: list[int] | None = None) -> list[dict]:
        """Run ``n_iter`` iterations; at each checkpoint record value and exploitability."""
        marks = set(checkpoints or [])
        history = []
        for _ in range(n_iter):
            self.iterate()
            if self.iterations in marks:
                avg = self.average_strategy()
                history.append({"iteration": self.iterations,
                                "value": expected_value(avg),
                                "exploitability": exploitability(avg)})
        return history


def solve(n_iter: int = 20_000, checkpoints: list[int] | None = None,
          plus: bool = False) -> tuple[Strategy, list[dict]]:
    """Run CFR and return (average strategy, checkpoint history)."""
    cfr = KuhnCFR(plus=plus)
    hist = cfr.run(n_iter, checkpoints)
    return cfr.average_strategy(), hist


def log_checkpoints(n_iter: int, per_decade: int = 8) -> list[int]:
    """Roughly log-spaced iteration counts from 1 to n_iter."""
    pts = np.unique(np.round(np.logspace(0, np.log10(n_iter), per_decade *
                                         int(np.ceil(np.log10(n_iter))) + 1)).astype(int))
    return [int(x) for x in pts if 1 <= x <= n_iter]


# ---------------------------------------------------------------------------
# Equilibrium diagnostics and the exploitation study
# ---------------------------------------------------------------------------

def bet_probs(s: Strategy) -> dict[str, float]:
    """P(bet or call) at each information set."""
    return {k: float(s[k][1]) for k in INFOSETS}


def equilibrium_family_check(s: Strategy) -> dict[str, float]:
    """How close ``s`` is to Kuhn's family: α, the K/J ratio and the other pins."""
    b = bet_probs(s)
    alpha = b["J"]
    return {"alpha": alpha, "king_bet": b["K"], "king_minus_3alpha": b["K"] - 3 * alpha,
            "queen_bet": b["Q"], "queen_call": b["Qpb"],
            "queen_call_minus_alpha_third": b["Qpb"] - (alpha + 1 / 3),
            "p2_jack_bluff": b["Jp"], "p2_queen_call": b["Qb"]}


def own_reach(s: Strategy, key: str) -> float:
    """P1's own probability of reaching infoset ``key`` (1 at the root; P(check) at 'pb')."""
    if key.endswith("pb"):
        return float(s[key[0]][0])
    return 1.0


def mix_p1(eq: Strategy, br: Strategy, lam: float) -> Strategy:
    """Behaviour strategy equivalent to playing ``br`` w.p. λ and ``eq`` otherwise.

    Mixing at each infoset must weight each component by its own reach
    probability (Kuhn's theorem / sequence form); a naive per-infoset average
    would not be the same mixed strategy.
    """
    out = {}
    for k in P1_INFOSETS:
        xb, xe = lam * own_reach(br, k), (1 - lam) * own_reach(eq, k)
        tot = xb + xe
        out[k] = (xb * br[k] + xe * eq[k]) / tot if tot > 0 else eq[k].copy()
    return out


def leaky_opponent(base: Strategy, leak: str) -> Strategy:
    """P2 as in ``base`` but with one leak.

    ``"never_bluff_jack"``: after a check P2 never bets the Jack.
    ``"always_call_queen"``: P2 always calls a bet with the Queen.
    """
    s = {k: np.asarray(v, dtype=float).copy() for k, v in base.items()}
    if leak == "never_bluff_jack":
        s["Jp"] = np.array([1.0, 0.0])
    elif leak == "always_call_queen":
        s["Qb"] = np.array([0.0, 1.0])
    else:
        raise ValueError(f"unknown leak {leak!r}")
    return s


def exploitation_study(eq: Strategy, leak: str, lams: np.ndarray | None = None,
                       tol: float = 1e-9) -> dict:
    """Exploit-vs-protect numbers for P1 against a leaky P2.

    ``eq`` should be an exact equilibrium profile (e.g. :func:`analytic_equilibrium`
    at the α CFR found), so that "indifferent" is well defined. P1's exploit is
    the best response to the leaky P2 that keeps ``eq``'s mix wherever deviating
    gains nothing (within ``tol``); it deviates only where the leak pays.

    - ``ev_eq``: the equilibrium strategy against the leak;
    - ``ev_br``: the best response against the leak;
    - ``br_worst``: the best response against *its own* best response (what it
      earns once P2 adapts), to compare with ``eq_worst`` (= −1/18 at equilibrium);
    - ``frontier``: for mixtures λ·BR + (1 − λ)·eq (mixed at the start of each
      hand), the EV against the leak and the worst-case EV.
    """
    opp = leaky_opponent(eq, leak)
    ev_eq = expected_value(combine(eq, opp))
    ev_br, br = best_response(combine(eq, opp), 0, default=eq, tol=tol)
    br_full = combine(br, eq)
    br_worst = -best_response(br_full, 1)[0]
    eq_worst = -best_response(eq, 1)[0]
    lams = np.linspace(0, 1, 21) if lams is None else lams
    frontier = []
    for lam in lams:
        m = mix_p1(eq, br_full, float(lam))
        frontier.append({"lam": float(lam),
                         "ev_vs_leak": expected_value(combine(m, opp)),
                         "worst_case": -best_response(combine(m, eq), 1)[0]})
    changed = [k for k in P1_INFOSETS if not np.allclose(br_full[k], eq[k])]
    return {"leak": leak, "ev_eq": ev_eq, "ev_br": ev_br, "gain": ev_br - ev_eq,
            "br_worst": br_worst, "eq_worst": eq_worst,
            "br_exploitability_loss": eq_worst - br_worst,
            "deviations": {k: [float(eq[k][1]), float(br_full[k][1])] for k in changed},
            "frontier": frontier}
