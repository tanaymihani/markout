"""LOBSTER sample files: constants, loaders and the message/book consistency check.

LOBSTER (Huang & Polak 2011) rebuilds the Nasdaq book from TotalView-ITCH. Each sample
has two row-aligned files:

* message file, one row per event: time (seconds after midnight), type, order id,
  size, price (dollars x 10^4) and direction (+1 buy limit order, -1 sell limit order;
  executing a sell limit order is a buyer-initiated trade);
* order-book file, the 10 best occupied levels per side *after* each event. Empty
  levels hold dummy prices (+-9999999999) with size 0.

The readme adds a detail the queue model relies on: the message file holds only events
inside the requested price range (the 10 visible levels). Activity deeper in the book is
invisible, so an order whose price drifts beyond level 10 can no longer be tracked.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import numpy as np
import polars as pl

from markout.paths import PROCESSED

TICKERS: tuple[str, ...] = ("AAPL", "AMZN", "GOOG", "INTC", "MSFT")
DATE = "2012-06-21"
LEVELS = 10
PRICE_SCALE = 10_000            # LOBSTER price = dollars x 10^4
TICK = 100                      # $0.01, the Reg NMS minimum increment above $1
DUMMY_ASK = 9_999_999_999       # price of an empty ask level
DUMMY_BID = -9_999_999_999      # price of an empty bid level
OPEN, CLOSE = 34_200.0, 57_600.0          # 09:30:00 and 16:00:00
TRIM = 300.0                              # seconds dropped at each end of the day
SESSION = (OPEN + TRIM, CLOSE - TRIM)     # 09:35:00 - 15:55:00: every analysis uses this
SPLIT = 45_900.0                          # 12:45:00: fit before, test after

LOBSTER_DIR = PROCESSED / "lobster"


class MsgType(IntEnum):
    SUBMIT = 1      # new limit order
    CANCEL = 2      # partial cancellation
    DELETE = 3      # full deletion
    EXECUTE = 4     # execution of a visible limit order
    HIDDEN = 5      # execution of a hidden limit order
    CROSS = 6       # auction cross trade
    HALT = 7        # trading halt indicator (price -1 halt, 0 quote, 1 resume)


MESSAGE_COLUMNS: tuple[str, ...] = ("time", "type", "order_id", "size", "price", "direction")
BOOK_COLUMNS: tuple[str, ...] = tuple(
    f"{field}_{i}" for i in range(1, LEVELS + 1)
    for field in ("ask_price", "ask_size", "bid_price", "bid_size")
)


def message_path(ticker: str) -> Path:
    return LOBSTER_DIR / f"{ticker}_messages.parquet"


def book_path(ticker: str) -> Path:
    return LOBSTER_DIR / f"{ticker}_book.parquet"


def available(tickers: tuple[str, ...] = TICKERS) -> bool:
    """True when the Parquet files for every ticker are on disk."""
    return all(message_path(t).exists() and book_path(t).exists() for t in tickers)


@dataclass(frozen=True)
class Messages:
    """The message file as NumPy columns (struct of arrays)."""

    time: np.ndarray        # float64, seconds after midnight
    type: np.ndarray        # int8, MsgType
    order_id: np.ndarray    # int64
    size: np.ndarray        # int64, shares
    price: np.ndarray       # int64, dollars x 10^4
    direction: np.ndarray   # int8, +1 buy limit order, -1 sell limit order

    def __len__(self) -> int:
        return len(self.time)


@dataclass(frozen=True)
class Book:
    """The order-book file: (n, levels) int64 arrays, row n = state after message n."""

    ask_price: np.ndarray
    ask_size: np.ndarray
    bid_price: np.ndarray
    bid_size: np.ndarray

    def __len__(self) -> int:
        return self.ask_price.shape[0]

    @property
    def levels(self) -> int:
        return self.ask_price.shape[1]

    @property
    def best_bid(self) -> np.ndarray:
        return self.bid_price[:, 0]

    @property
    def best_ask(self) -> np.ndarray:
        return self.ask_price[:, 0]

    def mid(self) -> np.ndarray:
        """Mid price in LOBSTER price units (NaN when a side is empty)."""
        a, b = self.ask_price[:, 0], self.bid_price[:, 0]
        m = 0.5 * (a.astype(np.float64) + b)
        m[(a == DUMMY_ASK) | (b == DUMMY_BID)] = np.nan
        return m

    def spread(self) -> np.ndarray:
        """Quoted spread in price units (NaN when a side is empty)."""
        a, b = self.ask_price[:, 0], self.bid_price[:, 0]
        s = (a - b).astype(np.float64)
        s[(a == DUMMY_ASK) | (b == DUMMY_BID)] = np.nan
        return s


@dataclass(frozen=True)
class Day:
    ticker: str
    messages: Messages
    book: Book

    def __len__(self) -> int:
        return len(self.messages)


def messages_from_frame(df: pl.DataFrame) -> Messages:
    return Messages(
        time=df["time"].to_numpy().astype(np.float64, copy=False),
        type=df["type"].to_numpy().astype(np.int8, copy=False),
        order_id=df["order_id"].to_numpy().astype(np.int64, copy=False),
        size=df["size"].to_numpy().astype(np.int64, copy=False),
        price=df["price"].to_numpy().astype(np.int64, copy=False),
        direction=df["direction"].to_numpy().astype(np.int8, copy=False),
    )


def load_messages(ticker: str) -> Messages:
    return messages_from_frame(pl.read_parquet(message_path(ticker)))


def load_book(ticker: str, levels: int = LEVELS) -> Book:
    """Read one (n, levels) block at a time, so peak memory stays near the final arrays."""
    def block(field: str) -> np.ndarray:
        cols = [f"{field}_{i}" for i in range(1, levels + 1)]
        frame = pl.read_parquet(book_path(ticker), columns=cols)
        return np.ascontiguousarray(frame.to_numpy().astype(np.int64, copy=False))

    return Book(block("ask_price"), block("ask_size"), block("bid_price"), block("bid_size"))


def load_day(ticker: str, levels: int = LEVELS) -> Day:
    return Day(ticker, load_messages(ticker), load_book(ticker, levels))


# --------------------------------------------------------------------------- helpers


def state_index(time: np.ndarray, t: np.ndarray | float) -> np.ndarray:
    """Index of the book state in effect at clock time t: the last message with time <= t.

    Returns -1 for times before the first message.
    """
    return np.searchsorted(time, t, side="right") - 1


def state_durations(time: np.ndarray, start: float = SESSION[0],
                    end: float = SESSION[1]) -> np.ndarray:
    """Seconds each book state is in effect inside [start, end).

    State n (after message n) holds on [time[n], time[n+1]); the last one until `end`.
    Time-weighting by these durations turns per-message quantities into clock averages.
    """
    nxt = np.append(time[1:], end)
    return np.clip(np.minimum(nxt, end) - np.maximum(time, start), 0.0, None)


def time_average(time: np.ndarray, values: np.ndarray, lo: np.ndarray,
                 hi: np.ndarray) -> np.ndarray:
    """Clock-time average of a per-state quantity over each interval [lo_k, hi_k).

    `values[n]` holds on [time[n], time[n+1]). With F(t) = integral of the step function
    up to t, the average is (F(hi) - F(lo)) / (hi - lo); F is a cumulative sum at the
    message times plus a partial last step. Intervals must start at or after time[0].
    """
    v = np.asarray(values, dtype=np.float64)
    cum = np.concatenate([[0.0], np.cumsum(v[:-1] * np.diff(time))])

    def integral(t: np.ndarray) -> np.ndarray:
        i = np.searchsorted(time, t, side="right") - 1
        return cum[i] + v[i] * (t - time[i])

    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    return (integral(hi) - integral(lo)) / (hi - lo)


def depth_at(prices: np.ndarray, sizes: np.ndarray, price: np.ndarray) -> np.ndarray:
    """Displayed size at `price` in each row of a (k, levels) block (0 if not a level)."""
    return np.where(prices == price[:, None], sizes, 0).sum(axis=1)


def in_view(prices: np.ndarray, side: np.ndarray, price: np.ndarray) -> np.ndarray:
    """Whether `price` lies inside the visible levels of its side, row by row.

    A bid price is in view when it is at or above the deepest visible bid (a dummy
    -9999999999 when the side has fewer levels, so everything is in view); mirror for
    asks. Inside that range an absent price genuinely has zero displayed depth.
    """
    deepest = prices[:, -1]
    return np.where(side > 0, price >= deepest, price <= deepest)


# --------------------------------------------------------------------------- consistency


@dataclass(frozen=True)
class Consistency:
    n_messages: int
    n_eligible: int         # type 1-4 messages (excluding the first row)
    n_checked: int          # ... whose price is in view both before and after
    n_pass: int
    by_type: dict[int, tuple[int, int]]   # type -> (checked, passed)
    failures: np.ndarray    # message indices that failed

    @property
    def pass_rate(self) -> float:
        return self.n_pass / self.n_checked if self.n_checked else float("nan")

    @property
    def coverage(self) -> float:
        return self.n_checked / self.n_eligible if self.n_eligible else float("nan")


def check_consistency(messages: Messages, book: Book, chunk: int = 50_000) -> Consistency:
    """Replay every type 1-4 message against the book.

    For message n at price p on side d (d = direction: +1 bid, -1 ask), if p is in view
    in book rows n-1 and n, the displayed size at p must change by exactly +size for a
    submission (type 1) and -size for a cancel, delete or visible execution (types 2-4).
    Hidden executions (5) never touch displayed depth and are not checked.
    """
    n = len(messages)
    elig = np.flatnonzero((messages.type >= 1) & (messages.type <= 4)
                          & (np.arange(n) > 0) & (np.abs(messages.direction) == 1))
    checked = np.zeros(len(elig), dtype=bool)
    ok = np.zeros(len(elig), dtype=bool)
    for s in range(0, len(elig), chunk):
        idx = elig[s:s + chunk]
        d = messages.direction[idx].astype(np.int64)
        p = messages.price[idx]
        bid = d > 0
        prev_p = np.where(bid[:, None], book.bid_price[idx - 1], book.ask_price[idx - 1])
        prev_q = np.where(bid[:, None], book.bid_size[idx - 1], book.ask_size[idx - 1])
        post_p = np.where(bid[:, None], book.bid_price[idx], book.ask_price[idx])
        post_q = np.where(bid[:, None], book.bid_size[idx], book.ask_size[idx])
        view = in_view(prev_p, d, p) & in_view(post_p, d, p)
        delta = depth_at(post_p, post_q, p) - depth_at(prev_p, prev_q, p)
        sign = np.where(messages.type[idx] == MsgType.SUBMIT, 1, -1)
        checked[s:s + chunk] = view
        ok[s:s + chunk] = view & (delta == sign * messages.size[idx])
    types = messages.type[elig]
    by_type = {int(k): (int(checked[types == k].sum()), int(ok[types == k].sum()))
               for k in (1, 2, 3, 4)}
    return Consistency(
        n_messages=n,
        n_eligible=len(elig),
        n_checked=int(checked.sum()),
        n_pass=int(ok.sum()),
        by_type=by_type,
        failures=elig[checked & ~ok],
    )
