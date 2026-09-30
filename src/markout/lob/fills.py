"""Fill simulator for hypothetical passive limit orders replayed against a LOBSTER day.

A hypothetical order is (placement message index, side, limit price, size, max wait).
It is placed right after message `place_idx`, so it sees the book row `place_idx` and
every later message. Our orders never enter the real book: each one is evaluated as if
it were the only extra order, independently of the others, and it has no impact on
anyone else's behaviour (the main limitation of any replay).

Three fill models, from optimistic to pessimistic:

* ``touch``: filled as soon as any trade (visible or hidden) prints at our price.
* ``fifo``: price-time priority, rebuilt from order ids.
    1. at placement we queue behind the displayed depth at our price (``queue_ahead``);
    2. orders submitted at our price afterwards sit behind us (their ids are tracked);
    3. a partial cancel or delete of any other id at our price shrinks the queue ahead;
    4. a visible execution (type 4) at our price consumes the queue ahead first, and the
       remainder fills us;
    5. hidden executions (type 5) at our price are ignored: Nasdaq gives displayed
       orders priority over non-displayed ones at the same price, so a hidden fill means
       nobody reached our displayed queue.
* ``through``: filled only when the market trades *through* our price. This is the
  "always last in line" model: every other order at our price, even ones that arrive
  later, is assumed to trade first, so only an aggressor who exhausts the displayed
  level and keeps going reaches us, and the data show that as a print strictly beyond
  our price (or the opposite quote reaching it). We deliberately do not also fill when
  an execution merely empties our level: when an aggressor clears the level exactly,
  FIFO leaves us first in line but unfilled, so that trigger would make the
  "pessimistic" model fill more often than FIFO (it did, for the three small-tick
  stocks, in a first version). Without it, fills nest by construction:
  through <= fifo <= touch (tested).

Rules shared by all three models:

* A trade at a price strictly beyond ours (below a buy limit, above a sell limit), or
  the opposite best quote reaching our price, fills the whole remaining quantity: the
  aggressive order that did it would have had to trade with us first. With 100-share
  orders it is almost always large enough.
* Fills are at our limit price (we are the passive side).
* The order is cancelled once more than ``max_wait`` seconds have passed since the
  placement message (``timeout``), or when its price leaves the 10 visible levels
  (``level_exit``): LOBSTER records no messages beyond level 10, so the queue can no
  longer be tracked. A halt message ends every live order (``halt``).
* An order that would cross the book at placement, or that is placed outside the
  visible levels, is ``rejected`` and never becomes active.

Order of operations for message n: (1) expire orders whose deadline is before
time[n]; (2) apply message n; (3) check the book row n for crosses and level exits;
(4) activate the orders placed at n. Each order's outcome depends only on its own
parameters and the message stream, so the processing order of concurrent orders
cannot change the result. That makes the algorithm straightforward to port (see
``cpp/queue_sim.cpp``) and to test for bit-identical output.

``simulate(..., engine="python")`` is the readable reference below; ``engine="cpp"``
runs the C++17 port; ``"auto"`` uses C++ when the extension is built.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import pandas as pd

from markout.lob.lobster import Book, Day, Messages, MsgType

MODELS: tuple[str, ...] = ("touch", "fifo", "through")
MODEL_CODE = {name: k for k, name in enumerate(MODELS)}


class End(IntEnum):
    ACTIVE = 0          # internal only; never in the output
    FILLED = 1
    TIMEOUT = 2
    LEVEL_EXIT = 3
    HALT = 4
    END_OF_DATA = 5
    REJECTED = 6


END_NAMES = {e.value: e.name.lower() for e in End}


@dataclass(frozen=True)
class Orders:
    """Hypothetical orders as a struct of arrays."""

    place_idx: np.ndarray   # int64, index of the message after which the order is placed
    side: np.ndarray        # int8, +1 buy, -1 sell
    price: np.ndarray       # int64, limit price in LOBSTER units (dollars x 10^4)
    size: np.ndarray        # int64, shares
    max_wait: np.ndarray    # float64, seconds after time[place_idx]

    def __len__(self) -> int:
        return len(self.place_idx)

    @staticmethod
    def make(place_idx, side, price, size=100, max_wait=60.0) -> "Orders":
        place_idx = np.asarray(place_idx, dtype=np.int64)
        k = len(place_idx)
        return Orders(
            place_idx=place_idx,
            side=np.broadcast_to(np.asarray(side, dtype=np.int8), (k,)).copy(),
            price=np.broadcast_to(np.asarray(price, dtype=np.int64), (k,)).copy(),
            size=np.broadcast_to(np.asarray(size, dtype=np.int64), (k,)).copy(),
            max_wait=np.broadcast_to(np.asarray(max_wait, dtype=np.float64), (k,)).copy(),
        )


@dataclass(frozen=True)
class Fills:
    """One row per order for one fill model (same order as the input orders)."""

    filled_qty: np.ndarray   # int64
    fill_idx: np.ndarray     # int64, message of the last fill (-1 if none)
    fill_time: np.ndarray    # float64, time of the last fill (NaN if none)
    fill_price: np.ndarray   # int64, our limit price if anything filled, else 0
    end_idx: np.ndarray      # int64, last message the order was alive for
    end_time: np.ndarray     # float64, when it ended (the deadline for a timeout)
    end_reason: np.ndarray   # int8, End
    queue_ahead: np.ndarray  # int64, displayed depth ahead of us at placement

    FIELDS = ("filled_qty", "fill_idx", "fill_time", "fill_price",
              "end_idx", "end_time", "end_reason", "queue_ahead")

    def __len__(self) -> int:
        return len(self.filled_qty)

    def equals(self, other: "Fills") -> bool:
        """Bit-for-bit equality of every field (NaN == NaN)."""
        for f in self.FIELDS:
            a, b = getattr(self, f), getattr(other, f)
            if a.dtype != b.dtype or a.shape != b.shape:
                return False
            if a.dtype.kind == "f":
                if not np.array_equal(a.view(np.int64), b.view(np.int64)):
                    return False
            elif not np.array_equal(a, b):
                return False
        return True


# --------------------------------------------------------------------------- reference


def _validate(messages: Messages, book: Book, orders: Orders, model: str) -> None:
    if model not in MODEL_CODE:
        raise ValueError(f"model must be one of {MODELS}")
    if len(book) != len(messages):
        raise ValueError("book and messages must be row-aligned")
    if len(orders) and (orders.place_idx.min() < 0 or orders.place_idx.max() >= len(messages)):
        raise ValueError("place_idx out of range")
    if len(orders) and not np.all(np.isin(orders.side, (-1, 1))):
        raise ValueError("side must be +1 (buy) or -1 (sell)")


def _depth(prices: np.ndarray, sizes: np.ndarray, n: int, price: int) -> int:
    """Displayed size at `price` in book row n (0 if the price is not a level)."""
    row = prices[n]
    for k in range(row.shape[0]):
        if row[k] == price:
            return int(sizes[n, k])
    return 0


def simulate_python(messages: Messages, book: Book, orders: Orders, model: str) -> Fills:
    """Reference implementation: one sequential pass over the messages."""
    _validate(messages, book, orders, model)
    fifo, touch = model == "fifo", model == "touch"
    n_msg, k_ord = len(messages), len(orders)

    # Python lists index much faster than NumPy scalars inside a loop.
    T = messages.time.tolist()
    TY = messages.type.tolist()
    OID = messages.order_id.tolist()
    SZ = messages.size.tolist()
    PX = messages.price.tolist()
    DIR = messages.direction.tolist()
    best_bid = book.bid_price[:, 0].tolist()
    best_ask = book.ask_price[:, 0].tolist()
    last_bid = book.bid_price[:, -1].tolist()
    last_ask = book.ask_price[:, -1].tolist()

    place = orders.place_idx.tolist()
    side = orders.side.tolist()
    price = orders.price.tolist()
    remaining = orders.size.tolist()
    max_wait = orders.max_wait.tolist()

    filled = [0] * k_ord
    fill_idx = [-1] * k_ord
    fill_time = [math.nan] * k_ord
    end_idx = [-1] * k_ord
    end_time = [math.nan] * k_ord
    reason = [int(End.ACTIVE)] * k_ord
    queue0 = [0] * k_ord
    ahead = [0] * k_ord                  # FIFO: displayed shares still ahead of us
    behind: list[set | None] = [None] * k_ord   # FIFO: ids that joined our level after us
    deadline = [0.0] * k_ord

    # live orders by side and price: {price: [order indices]}
    live = {1: {}, -1: {}}
    heap: list[tuple[float, int]] = []   # (deadline, order) for timeouts

    def finish(j: int, why: End, n: int, when: float) -> None:
        reason[j] = int(why)
        end_idx[j] = n
        end_time[j] = when
        level = live[side[j]]
        level[price[j]].remove(j)
        if not level[price[j]]:
            del level[price[j]]

    def fill(j: int, qty: int, n: int) -> None:
        qty = min(qty, remaining[j])
        if qty <= 0:
            return
        remaining[j] -= qty
        filled[j] += qty
        fill_idx[j] = n
        fill_time[j] = T[n]
        if remaining[j] == 0:
            finish(j, End.FILLED, n, T[n])

    def fill_all(orders_at_levels: list[int], n: int) -> None:
        for j in orders_at_levels:
            fill(j, remaining[j], n)

    order_of = sorted(range(k_ord), key=lambda j: (place[j], j))
    ptr = 0
    start = place[order_of[0]] if k_ord else n_msg
    for n in range(max(start, 0), n_msg):
        tn = T[n]
        # (1) timeouts: deadline strictly before this message
        while heap and heap[0][0] < tn:
            dl, j = heapq.heappop(heap)
            if reason[j] == End.ACTIVE:
                finish(j, End.TIMEOUT, n - 1, dl)

        # (2) the message itself
        ty = TY[n]
        if ty == MsgType.HALT:
            for lv in (live[1], live[-1]):
                for j in [j for js in lv.values() for j in js]:
                    finish(j, End.HALT, n, tn)
        elif live[1] or live[-1]:
            d, p = DIR[n], PX[n]
            at = live[d].get(p) if d in live else None
            if ty == MsgType.SUBMIT:
                if fifo and at:
                    for j in at:
                        behind[j].add(OID[n])
            elif ty == MsgType.CANCEL or ty == MsgType.DELETE:
                if fifo and at:
                    oid, s = OID[n], SZ[n]
                    for j in at:
                        if oid in behind[j]:
                            if ty == MsgType.DELETE:
                                behind[j].discard(oid)
                        else:
                            ahead[j] = max(0, ahead[j] - s)
            elif ty == MsgType.EXECUTE or ty == MsgType.HIDDEN:
                # trades at our price (the through model ignores them)
                if touch:
                    for lv in (live[1], live[-1]):
                        if p in lv:
                            fill_all(list(lv[p]), n)
                elif fifo and at and ty == MsgType.EXECUTE:
                    for j in list(at):
                        x = SZ[n]
                        take = min(ahead[j], x)
                        ahead[j] -= take
                        if x - take > 0:
                            fill(j, x - take, n)
                # trades strictly through our price: every model
                for q in [q for q in live[1] if q > p]:
                    fill_all(list(live[1][q]), n)
                for q in [q for q in live[-1] if q < p]:
                    fill_all(list(live[-1][q]), n)

        # (3) book row n: the opposite quote reaching our price, or our level leaving view
        if live[1]:
            ba, lb = best_ask[n], last_bid[n]
            for q in [q for q in live[1] if q >= ba]:
                fill_all(list(live[1][q]), n)
            for q in [q for q in live[1] if q < lb]:
                for j in list(live[1][q]):
                    finish(j, End.LEVEL_EXIT, n, tn)
        if live[-1]:
            bb, la = best_bid[n], last_ask[n]
            for q in [q for q in live[-1] if q <= bb]:
                fill_all(list(live[-1][q]), n)
            for q in [q for q in live[-1] if q > la]:
                for j in list(live[-1][q]):
                    finish(j, End.LEVEL_EXIT, n, tn)

        # (4) activate orders placed after message n
        while ptr < k_ord and place[order_of[ptr]] == n:
            j = order_of[ptr]
            ptr += 1
            s, q = side[j], price[j]
            if s > 0:
                queue0[j] = _depth(book.bid_price, book.bid_size, n, q)
                bad = q >= best_ask[n] or q < last_bid[n]
            else:
                queue0[j] = _depth(book.ask_price, book.ask_size, n, q)
                bad = q <= best_bid[n] or q > last_ask[n]
            if bad or remaining[j] <= 0:
                reason[j], end_idx[j], end_time[j] = int(End.REJECTED), n, tn
                continue
            ahead[j] = queue0[j]
            behind[j] = set() if fifo else None
            deadline[j] = tn + max_wait[j]
            live[s].setdefault(q, []).append(j)
            heapq.heappush(heap, (deadline[j], j))
        if ptr == k_ord and not live[1] and not live[-1]:
            break

    # orders still alive when the data run out
    for lv in (live[1], live[-1]):
        for j in [j for js in lv.values() for j in js]:
            finish(j, End.END_OF_DATA, n_msg - 1, T[n_msg - 1])

    filled_arr = np.array(filled, dtype=np.int64)
    return Fills(
        filled_qty=filled_arr,
        fill_idx=np.array(fill_idx, dtype=np.int64),
        fill_time=np.array(fill_time, dtype=np.float64),
        fill_price=np.where(filled_arr > 0, orders.price, 0).astype(np.int64),
        end_idx=np.array(end_idx, dtype=np.int64),
        end_time=np.array(end_time, dtype=np.float64),
        end_reason=np.array(reason, dtype=np.int8),
        queue_ahead=np.array(queue0, dtype=np.int64),
    )


# --------------------------------------------------------------------------- dispatch


def cpp_available() -> bool:
    try:
        from markout.lob import _queue_sim  # noqa: F401
    except ImportError:
        return False
    return True


def simulate_cpp(messages: Messages, book: Book, orders: Orders, model: str) -> Fills:
    from markout.lob import _queue_sim

    _validate(messages, book, orders, model)
    out = _queue_sim.simulate(
        messages.time, messages.type, messages.order_id, messages.size, messages.price,
        messages.direction, book.bid_price, book.bid_size, book.ask_price, book.ask_size,
        orders.place_idx, orders.side, orders.price, orders.size, orders.max_wait,
        MODEL_CODE[model],
    )
    return Fills(**{f: out[f] for f in Fills.FIELDS})


def simulate(messages: Messages, book: Book, orders: Orders, model: str = "fifo",
             engine: str = "auto") -> Fills:
    """Run one fill model. engine: "python" (reference), "cpp", or "auto"."""
    if engine == "auto":
        engine = "cpp" if cpp_available() else "python"
    if engine == "python":
        return simulate_python(messages, book, orders, model)
    if engine == "cpp":
        return simulate_cpp(messages, book, orders, model)
    raise ValueError("engine must be 'auto', 'python' or 'cpp'")


def simulate_all(day: Day, orders: Orders, models: tuple[str, ...] = MODELS,
                 engine: str = "auto") -> pd.DataFrame:
    """One row per order and model: order fields plus fill results."""
    frames = []
    for model in models:
        f = simulate(day.messages, day.book, orders, model, engine)
        frames.append(pd.DataFrame({
            "order": np.arange(len(orders)),
            "model": model,
            "place_idx": orders.place_idx,
            "side": orders.side,
            "price": orders.price,
            "size": orders.size,
            **{name: getattr(f, name) for name in Fills.FIELDS},
        }))
    df = pd.concat(frames, ignore_index=True)
    df["end_reason_name"] = df["end_reason"].map(END_NAMES)
    return df
