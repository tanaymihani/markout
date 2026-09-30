"""Card market-making game: interview practice with a Bayes-optimal benchmark.

Run ``python -m markout.games.cardgame`` to play, or add ``--auto`` to watch the
Bayes quoter play itself.

The game
--------
N cards are dealt face down from a standard 52-card deck (A = 1, …, K = 13, four
suits); you make markets on their sum S. Each round you quote a two-sided market
(bid < ask, width at most ``max_width``) and a few bots trade one unit each
against it:

- an *informed* bot (probability ``p_informed``) knows one of the cards still
  face down (uniformly chosen). Its value is E[S | the revealed cards and its
  card]; it buys at your ask if the ask is below that value, sells at your bid if
  the bid is above it, and otherwise passes. It trades only when your quote is
  wrong relative to its information;
- a *noise* bot buys or sells with probability ½ each.

Then one card is turned face up. After the last round every card is face up and
each trade settles at S: buying at the bid earns S − bid, selling at the ask
earns ask − S.

The benchmark
-------------
Before each round the game computes the Bayes-optimal mid,
E[S | revealed cards, all trades observed so far]. Trades are informative
because an informed bot's action depends on its card. The posterior is computed
by importance sampling: draw the face-down cards from the remaining deck and
weight each draw by the likelihood of every observed action,

    P(action | cards) = p_inf·(1/n_h)·Σ_{hidden j} 1{bot with card j acts so}
                        + (1 − p_inf)·P_noise(action),

where n_h is the number of cards that were face down when the bot traded. An
exact enumeration (:func:`exact_posterior_mean`) is provided for small decks and
is used by the tests to validate the sampler.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

STANDARD_DECK = tuple(int(r) for r in np.repeat(np.arange(1, 14), 4))
RANK_NAMES = {1: "A", 11: "J", 12: "Q", 13: "K"}


def card_name(v: int) -> str:
    return RANK_NAMES.get(v, str(v))


@dataclass(frozen=True)
class Config:
    n_cards: int = 5                 # hidden cards whose sum is traded
    bots_per_round: int = 3
    p_informed: float = 0.4
    max_width: float = 4.0
    deck: tuple[int, ...] = STANDARD_DECK

    def __post_init__(self) -> None:
        if not 1 <= self.n_cards <= len(self.deck):
            raise ValueError("n_cards must be between 1 and the deck size")
        if not 0 <= self.p_informed <= 1:
            raise ValueError("p_informed must be a probability")


@dataclass(frozen=True)
class Bot:
    informed: bool
    slot_u: float      # uniform draw that picks which face-down card an informed bot sees
    side: int          # noise direction, +1 buy / −1 sell


@dataclass(frozen=True)
class Deal:
    """Everything random about one game, fixed in advance (so quoters can be compared)."""

    cards: tuple[int, ...]                 # in the order they will be revealed
    bots: tuple[tuple[Bot, ...], ...]      # one tuple of bots per round


def deal(cfg: Config, rng: np.random.Generator) -> Deal:
    cards = tuple(int(c) for c in rng.permutation(np.array(cfg.deck))[:cfg.n_cards])
    bots = tuple(tuple(Bot(bool(rng.random() < cfg.p_informed), float(rng.random()),
                           1 if rng.random() < 0.5 else -1)
                       for _ in range(cfg.bots_per_round))
                 for _ in range(cfg.n_cards))
    return Deal(cards, bots)


def remaining_mean(cfg: Config, known: Sequence[int]) -> float:
    """Mean rank of the deck after removing the ``known`` cards."""
    rest = list(cfg.deck)
    for c in known:
        rest.remove(c)
    return float(np.mean(rest))


def bot_value(cfg: Config, revealed: Sequence[int], own: int, n_hidden: int) -> float:
    """An informed bot's E[S]: revealed + its card + (n_hidden − 1)·mean of the rest."""
    return sum(revealed) + own + (n_hidden - 1) * remaining_mean(cfg, [*revealed, own])


def bot_action(value: float, bid: float, ask: float) -> int:
    """+1 buys at the ask, −1 sells at the bid, 0 passes."""
    if ask < value:
        return 1
    if bid > value:
        return -1
    return 0


@dataclass(frozen=True)
class Event:
    """One bot's action against a quote, as the quoter sees it."""

    round: int      # cards revealed before this round = round
    bid: float
    ask: float
    action: int     # +1 bot bought at the ask, −1 sold at the bid, 0 passed


@dataclass
class Game:
    """The game engine: quote, see the bots' actions, reveal a card."""

    cfg: Config
    deal: Deal
    round: int = 0
    events: list[Event] = field(default_factory=list)
    quotes: list[tuple[float, float]] = field(default_factory=list)

    @property
    def revealed(self) -> list[int]:
        return list(self.deal.cards[:self.round])

    @property
    def done(self) -> bool:
        return self.round >= self.cfg.n_cards

    def validate(self, bid: float, ask: float) -> None:
        if not (math.isfinite(bid) and math.isfinite(ask)):
            raise ValueError("bid and ask must be numbers")
        if not bid < ask:
            raise ValueError("bid must be below ask")
        if ask - bid > self.cfg.max_width + 1e-9:
            raise ValueError(f"width {ask - bid:g} exceeds the maximum {self.cfg.max_width:g}")

    def play_round(self, bid: float, ask: float) -> tuple[list[Event], int]:
        """Post (bid, ask), let this round's bots trade, reveal one card.

        Returns the round's events and the card that was turned face up.
        """
        if self.done:
            raise RuntimeError("game over")
        self.validate(bid, ask)
        r = self.round
        hidden = list(self.deal.cards[r:])
        out = []
        for bot in self.deal.bots[r]:
            if bot.informed:
                own = hidden[min(int(bot.slot_u * len(hidden)), len(hidden) - 1)]
                act = bot_action(bot_value(self.cfg, self.revealed, own, len(hidden)), bid, ask)
            else:
                act = bot.side
            out.append(Event(r, bid, ask, act))
        self.events.extend(out)
        self.quotes.append((bid, ask))
        self.round += 1
        return out, self.deal.cards[r]

    def pnl_by_round(self) -> list[float]:
        """Your PnL from each round's trades, marked to the final sum S."""
        s = sum(self.deal.cards)
        per = [0.0] * self.cfg.n_cards
        for e in self.events:
            if e.action == 1:        # you sold at the ask
                per[e.round] += e.ask - s
            elif e.action == -1:     # you bought at the bid
                per[e.round] += s - e.bid
        return per


# ---------------------------------------------------------------------------
# Bayes benchmark
# ---------------------------------------------------------------------------

def _event_likelihood(cfg: Config, e: Event, cards: np.ndarray) -> np.ndarray:
    """P(event | cards) for each row of ``cards`` (full deals in reveal order)."""
    r = e.round
    n_hidden = cfg.n_cards - r
    revealed = cards[:, :r]
    rev_sum = revealed.sum(axis=1)
    deck = np.array(cfg.deck, dtype=float)
    # mean of the deck after removing the revealed cards and the bot's own card
    deck_sum, n_deck = deck.sum(), deck.size
    inf = np.zeros(cards.shape[0])
    for j in range(r, cfg.n_cards):
        own = cards[:, j]
        rest_mean = (deck_sum - rev_sum - own) / (n_deck - r - 1)
        value = rev_sum + own + (n_hidden - 1) * rest_mean
        act = np.where(e.ask < value, 1, np.where(e.bid > value, -1, 0))
        inf += act == e.action
    inf /= n_hidden
    noise = 0.5 if e.action != 0 else 0.0
    return cfg.p_informed * inf + (1 - cfg.p_informed) * noise


def posterior_mean(cfg: Config, revealed: Sequence[int], events: Sequence[Event],
                   n_samples: int = 20_000, rng: np.random.Generator | None = None
                   ) -> dict[str, float]:
    """E[S | revealed cards, events] by importance sampling over the face-down cards.

    Returns the posterior mean, its Monte Carlo standard error, and the
    effective sample size of the importance weights.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    r = len(revealed)
    n_hidden = cfg.n_cards - r
    rest = list(cfg.deck)
    for c in revealed:
        rest.remove(c)
    rest_arr = np.array(rest)
    if n_hidden == 0:
        return {"mean": float(sum(revealed)), "se": 0.0, "ess": float(n_samples)}
    # uniform draws without replacement of the face-down cards
    keys = rng.random((n_samples, rest_arr.size))
    idx = np.argpartition(keys, n_hidden - 1, axis=1)[:, :n_hidden]
    hidden = rest_arr[idx]
    cards = np.hstack([np.tile(np.array(revealed, dtype=float), (n_samples, 1)), hidden])
    logw = np.zeros(n_samples)
    for e in events:
        lik = _event_likelihood(cfg, e, cards)
        with np.errstate(divide="ignore"):
            logw += np.log(lik)
    if not np.isfinite(logw).any():
        raise ValueError("the observed actions are impossible under the model")
    w = np.exp(logw - logw[np.isfinite(logw)].max())
    w /= w.sum()
    s = cards.sum(axis=1)
    mean = float(w @ s)
    ess = float(1.0 / np.sum(w * w))
    se = float(np.sqrt(w @ (s - mean) ** 2 / ess))
    return {"mean": mean, "se": se, "ess": ess}


def exact_posterior_mean(cfg: Config, revealed: Sequence[int], events: Sequence[Event]) -> float:
    """E[S | revealed, events] by enumerating every ordered set of face-down cards.

    Feasible only for small decks; used to validate :func:`posterior_mean`.
    """
    from itertools import permutations

    rest = list(cfg.deck)
    for c in revealed:
        rest.remove(c)
    n_hidden = cfg.n_cards - len(revealed)
    combos = np.array(list(permutations(rest, n_hidden)), dtype=float).reshape(-1, n_hidden)
    cards = np.hstack([np.tile(np.array(revealed, dtype=float), (combos.shape[0], 1)), combos])
    w = np.ones(cards.shape[0])
    for e in events:
        w *= _event_likelihood(cfg, e, cards)
    return float(w @ cards.sum(axis=1) / w.sum())


def prior_mean(cfg: Config, revealed: Sequence[int]) -> float:
    """E[S | revealed] = Σ revealed + n_hidden · mean(remaining deck)."""
    n_hidden = cfg.n_cards - len(revealed)
    return sum(revealed) + (n_hidden * remaining_mean(cfg, revealed) if n_hidden else 0.0)


# ---------------------------------------------------------------------------
# Quoters and a full game
# ---------------------------------------------------------------------------

Quoter = Callable[[Game], tuple[float, float]]


def bayes_quoter(n_samples: int = 20_000, seed: int = 0) -> Quoter:
    """Quote E[S | revealed, trades] ± max_width/2."""
    rng = np.random.default_rng(seed)

    def quote(g: Game) -> tuple[float, float]:
        m = posterior_mean(g.cfg, g.revealed, g.events, n_samples, rng)["mean"]
        return m - g.cfg.max_width / 2, m + g.cfg.max_width / 2
    return quote


def prior_quoter(g: Game) -> tuple[float, float]:
    """Ignore the trades: quote E[S | revealed] ± max_width/2."""
    m = prior_mean(g.cfg, g.revealed)
    return m - g.cfg.max_width / 2, m + g.cfg.max_width / 2


@dataclass
class RoundRecord:
    round: int
    bid: float
    ask: float
    bayes_mid: float
    actions: list[int]
    revealed_card: int
    pnl: float


def play(cfg: Config, d: Deal, quoter: Quoter, n_samples: int = 20_000,
         seed: int = 0, benchmark: bool = True) -> tuple[Game, list[RoundRecord]]:
    """Play a full game with ``quoter``; record the Bayes mid before each round.

    With ``benchmark=False`` the Bayes mid is not computed (recorded as NaN).
    """
    g = Game(cfg, d)
    rng = np.random.default_rng(seed)
    recs = []
    while not g.done:
        bm = (posterior_mean(cfg, g.revealed, g.events, n_samples, rng)["mean"]
              if benchmark else float("nan"))
        bid, ask = quoter(g)
        ev, card = g.play_round(bid, ask)
        recs.append(RoundRecord(g.round - 1, bid, ask, bm, [e.action for e in ev], card, 0.0))
    for rec, p in zip(recs, g.pnl_by_round()):
        rec.pnl = p
    return g, recs


def summary_table(recs: Sequence[RoundRecord], total_sum: int) -> str:
    """Plain-text table of a finished game."""
    act = {1: "buy", -1: "sell", 0: "pass"}
    lines = [f"{'rnd':>3} {'bid':>7} {'ask':>7} {'your mid':>8} {'Bayes mid':>9} "
             f"{'bots':<16} {'card':>4} {'PnL':>7}"]
    for r in recs:
        mid = 0.5 * (r.bid + r.ask)
        bots = ",".join(act[a] for a in r.actions)
        lines.append(f"{r.round + 1:>3} {r.bid:>7.2f} {r.ask:>7.2f} {mid:>8.2f} {r.bayes_mid:>9.2f} "
                     f"{bots:<16} {card_name(r.revealed_card):>4} {r.pnl:>+7.2f}")
    lines.append(f"final sum S = {total_sum}; total PnL {sum(r.pnl for r in recs):+.2f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def describe_round(g: Game, r: int) -> str:
    """One line per finished round: each bot's action and the card turned up."""
    evs = [e for e in g.events if e.round == r]
    parts = []
    for k, e in enumerate(evs, 1):
        if e.action == 1:
            parts.append(f"bot {k} BUYS at {e.ask:g}")
        elif e.action == -1:
            parts.append(f"bot {k} SELLS at {e.bid:g}")
        else:
            parts.append(f"bot {k} passes")
    return "  " + "; ".join(parts) + f". Card turned up: {card_name(g.deal.cards[r])}."


def _human_quoter(read: Callable[[str], str], write: Callable[[str], None]) -> Quoter:
    def quote(g: Game) -> tuple[float, float]:
        if g.round > 0:
            write(describe_round(g, g.round - 1))
        shown = " ".join(card_name(c) for c in g.revealed) or "none"
        write(f"\nRound {g.round + 1}/{g.cfg.n_cards}. Face up: {shown}. "
              f"Face down: {g.cfg.n_cards - g.round}.")
        while True:
            raw = read(f"Your market 'bid ask' (max width {g.cfg.max_width:g}): ").strip()
            try:
                bid, ask = (float(x) for x in raw.replace(",", " ").split())
                g.validate(bid, ask)
                return bid, ask
            except (ValueError, TypeError) as exc:
                write(f"  invalid market ({exc}); try again, e.g. '33 39'")
    return quote


def main(argv: Sequence[str] | None = None, read: Callable[[str], str] = input,
         write: Callable[[str], None] = print) -> int:
    ap = argparse.ArgumentParser(description="Make markets on the sum of hidden cards.")
    ap.add_argument("--auto", action="store_true", help="let the Bayes quoter play itself")
    ap.add_argument("--seed", type=int, default=None, help="deal seed (random if omitted)")
    ap.add_argument("--cards", type=int, default=Config.n_cards)
    ap.add_argument("--bots", type=int, default=Config.bots_per_round)
    ap.add_argument("--p-informed", type=float, default=Config.p_informed)
    ap.add_argument("--width", type=float, default=Config.max_width)
    ap.add_argument("--samples", type=int, default=20_000)
    args = ap.parse_args(argv)
    cfg = Config(args.cards, args.bots, args.p_informed, args.width)
    seed = args.seed if args.seed is not None else int(np.random.SeedSequence().entropy % 2**32)
    d = deal(cfg, np.random.default_rng(seed))
    write(f"Card market game (seed {seed}): {cfg.n_cards} cards face down, sum S. "
          f"{cfg.bots_per_round} bots per round, each informed with p = {cfg.p_informed:g}. "
          f"E[S] before any card = {prior_mean(cfg, []):.2f}.")
    quoter = bayes_quoter(args.samples, seed) if args.auto else _human_quoter(read, write)
    try:
        g, recs = play(cfg, d, quoter, args.samples, seed)
    except (EOFError, KeyboardInterrupt):
        write("\nGame abandoned.")
        return 1
    if not args.auto:
        write(describe_round(g, cfg.n_cards - 1))
    write("\n" + summary_table(recs, sum(d.cards)))
    if not args.auto:
        _, bayes_recs = play(cfg, d, bayes_quoter(args.samples, seed), args.samples, seed)
        write(f"The Bayes quoter on the same deal and bots: total PnL "
              f"{sum(r.pnl for r in bayes_recs):+.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
