"""A tiny price-time-priority exchange that writes LOBSTER-format messages and books.

Used by the tests: hand-built scenarios with known answers for the fill models, and
random but *valid* event streams for the property tests of the C++ port. Every event
is recorded the way LOBSTER would record it (types 1-5, direction = side of the resting
limit order, prices in dollars x 10^4) together with the 10-level book after it, so the
output passes `lobster.check_consistency` by construction.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from markout.lob.lobster import DUMMY_ASK, DUMMY_BID, LEVELS, Book, Messages, MsgType


@dataclass
class SyntheticExchange:
    levels: int = LEVELS
    orders: dict[int, list] = field(default_factory=dict)        # id -> [side, price, size]
    queues: dict[int, dict[int, deque]] = field(default_factory=lambda: {1: {}, -1: {}})
    rows: list[tuple] = field(default_factory=list)              # message rows
    books: list[np.ndarray] = field(default_factory=list)        # (4, levels) per row
    next_id: int = 1

    # ------------------------------------------------------------------ book state
    def best(self, side: int) -> int | None:
        q = self.queues[side]
        if not q:
            return None
        return max(q) if side > 0 else min(q)

    def depth(self, side: int, price: int) -> int:
        return sum(self.orders[i][2] for i in self.queues[side].get(price, ()))

    def snapshot(self) -> np.ndarray:
        """Rows: ask price, ask size, bid price, bid size (levels columns)."""
        out = np.zeros((4, self.levels), dtype=np.int64)
        out[0, :], out[2, :] = DUMMY_ASK, DUMMY_BID
        for side, (pr, sz) in ((-1, (0, 1)), (1, (2, 3))):
            prices = sorted(self.queues[side], reverse=side > 0)[: self.levels]
            for k, p in enumerate(prices):
                out[pr, k] = p
                out[sz, k] = self.depth(side, p)
        return out

    def _emit(self, t: float, typ: int, oid: int, size: int, price: int, direction: int) -> None:
        self.rows.append((float(t), typ, oid, size, price, direction))
        self.books.append(self.snapshot())

    def _remove(self, oid: int, size: int) -> None:
        side, price, left = self.orders[oid]
        left -= size
        if left > 0:
            self.orders[oid][2] = left
            return
        del self.orders[oid]
        q = self.queues[side][price]
        q.remove(oid)
        if not q:
            del self.queues[side][price]

    # ------------------------------------------------------------------ events
    def submit(self, t: float, side: int, price: int, size: int, oid: int | None = None) -> int:
        """Add a resting limit order (must not cross the opposite best)."""
        opp = self.best(-side)
        if opp is not None and (price >= opp if side > 0 else price <= opp):
            raise ValueError("submit would cross the book; use market() for aggressive flow")
        if oid is None:
            oid = self.next_id
        self.next_id = max(self.next_id, oid) + 1
        self.orders[oid] = [side, price, size]
        self.queues[side].setdefault(price, deque()).append(oid)
        self._emit(t, MsgType.SUBMIT, oid, size, price, side)
        return oid

    def cancel(self, t: float, oid: int, size: int) -> None:
        """Partial cancellation (type 2); `size` must be less than what is left."""
        side, price, left = self.orders[oid]
        if not 0 < size < left:
            raise ValueError("partial cancel must leave shares")
        self._remove(oid, size)
        self._emit(t, MsgType.CANCEL, oid, size, price, side)

    def delete(self, t: float, oid: int) -> None:
        side, price, left = self.orders[oid]
        self._remove(oid, left)
        self._emit(t, MsgType.DELETE, oid, left, price, side)

    def execute(self, t: float, oid: int, size: int) -> None:
        """Visible execution of a resting order (type 4)."""
        side, price, left = self.orders[oid]
        if not 0 < size <= left:
            raise ValueError("execution size must be in (0, remaining]")
        self._remove(oid, size)
        self._emit(t, MsgType.EXECUTE, oid, size, price, side)

    def market(self, t: float, side: int, size: int) -> int:
        """Aggressive order of `side` (+1 buy): walks the opposite book in price-time
        priority, one type-4 message per resting order hit. Returns shares filled."""
        done = 0
        while done < size and self.best(-side) is not None:
            price = self.best(-side)
            oid = self.queues[-side][price][0]
            take = min(size - done, self.orders[oid][2])
            self.execute(t, oid, take)
            done += take
        return done

    def hidden(self, t: float, price: int, size: int, direction: int) -> None:
        """Execution of a hidden order (type 5): no change to displayed depth."""
        self._emit(t, MsgType.HIDDEN, 0, size, price, direction)

    def cross_trade(self, t: float, price: int, size: int) -> None:
        """Auction cross (type 6): printed, but the continuous book does not change."""
        self._emit(t, MsgType.CROSS, 0, size, price, -1)

    def halt(self, t: float, code: int = -1) -> None:
        """Trading-halt indicator (type 7): price -1 halt, 0 quote, 1 resume; the book row
        repeats the previous state, as in LOBSTER."""
        self._emit(t, MsgType.HALT, 0, 0, code, -1)

    # ------------------------------------------------------------------ output
    def build(self) -> tuple[Messages, Book]:
        arr = np.array(self.rows, dtype=np.float64) if self.rows else np.zeros((0, 6))
        messages = Messages(
            time=arr[:, 0].astype(np.float64),
            type=arr[:, 1].astype(np.int8),
            order_id=np.array([r[2] for r in self.rows], dtype=np.int64),
            size=np.array([r[3] for r in self.rows], dtype=np.int64),
            price=np.array([r[4] for r in self.rows], dtype=np.int64),
            direction=arr[:, 5].astype(np.int8),
        )
        b = np.stack(self.books) if self.books else np.zeros((0, 4, self.levels), np.int64)
        book = Book(*(np.ascontiguousarray(b[:, k, :]) for k in range(4)))
        return messages, book


# One random action: (kind, side, price offset in ticks, size, pick, time step).
KINDS = ("submit", "cancel", "delete", "market", "hidden", "cross", "halt")
SIZES = (1, 50, 100, 100, 200, 300, 500)


def apply_action(ex: SyntheticExchange, t: float, kind: str, side: int, offset: int, size: int,
                 pick: int, tick: int = 100, mid: int = 1_000_000) -> None:
    """Turn one action into a valid event. Actions that cannot apply to the current book
    (a cancel with no resting orders, a one-sided book...) fall back to a submission, so
    every action list maps to a well-formed LOBSTER stream."""
    own, opp = ex.best(side), ex.best(-side)
    if kind == "submit" or own is None or opp is None or not ex.orders:
        ref = own if own is not None else (opp - side * tick if opp is not None else mid)
        price = ref - side * tick * offset                    # offset -1 improves the quote
        if opp is not None and (price >= opp if side > 0 else price <= opp):
            price = opp - side * tick
        ex.submit(t, side, price, size)
    elif kind == "cancel":
        oid = sorted(ex.orders)[pick % len(ex.orders)]
        left = ex.orders[oid][2]
        if left > 1:
            ex.cancel(t, oid, 1 + pick % (left - 1))
        else:
            ex.delete(t, oid)
    elif kind == "delete":
        ex.delete(t, sorted(ex.orders)[pick % len(ex.orders)])
    elif kind == "market":
        ex.market(t, side, size)
    elif kind == "hidden":                   # usually inside the spread, sometimes at the touch
        bb, ba = ex.best(1), ex.best(-1)
        steps = (ba - bb) // tick
        ex.hidden(t, bb + tick * (pick % (steps + 1)), size, side)
    elif kind == "cross":
        ex.cross_trade(t, own, size)
    else:
        ex.halt(t, (-1, 0, 1)[pick % 3])


def seeded_exchange(levels: int, mid: int = 1_000_000, tick: int = 100, depth: int = 3,
                    size: int = 100) -> SyntheticExchange:
    """A two-sided book with `depth` levels per side around `mid`."""
    ex = SyntheticExchange(levels=levels)
    for side in (1, -1):
        for k in range(depth):
            ex.submit(34_200.0, side, mid - side * tick * (k + 1), size)
    return ex


def random_stream(rng: np.random.Generator, n_events: int = 400, levels: int = 5,
                  mid: int = 1_000_000, tick: int = 100) -> tuple[Messages, Book]:
    """A random but valid event stream: submits near the touch, partial cancels,
    deletes, marketable orders that walk the book, hidden prints, rare cross trades and
    halts. Few levels make orders leave the visible range often."""
    ex = seeded_exchange(levels, mid, tick)
    probs = np.array([0.45, 0.15, 0.20, 0.15, 0.04, 0.005, 0.005])
    t = 34_200.0
    for _ in range(n_events):
        if rng.random() < 0.7:                # the rest share a timestamp, as in real data
            t += float(rng.exponential(0.5))
        apply_action(ex, t, KINDS[rng.choice(len(KINDS), p=probs / probs.sum())],
                     1 if rng.random() < 0.5 else -1, int(rng.integers(-1, 4)),
                     int(rng.choice(SIZES)), int(rng.integers(0, 1_000_000)), tick, mid)
    return ex.build()
