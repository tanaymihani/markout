"""Stylized facts of one LOBSTER day: spreads, depth, activity and trade-sign memory.

Every quantity is computed on the trimmed session (09:35-15:55, `lobster.SESSION`), and
book quantities are *time-weighted*: each book state counts for the seconds it was in
effect, so a burst of 50 messages in a millisecond does not outweigh a quiet minute.

Tick size decides most of what follows (Bouchaud, Bonart, Donier & Gould 2018, ch. 3 and
Gould & Bonart 2016): when the one-cent tick is large relative to the price, the spread
is pinned at one tick, queues are long and queue sizes carry the information; when it
is small, the spread spans many ticks and queues at the best are thin. We classify a
stock as *large-tick* when its spread equals one tick more than half of the time.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from markout.lob.lobster import SESSION, TICK, Day, MsgType, state_durations

LARGE_TICK_THRESHOLD = 0.5   # share of time at a one-tick spread above which a stock is large-tick
ACF_LAGS = 10


@dataclass(frozen=True)
class Facts:
    ticker: str
    mean_price_usd: float
    tick_bps: float                 # one tick in bps of the mid
    spread_ticks: float             # time-weighted mean quoted spread
    spread_bps: float
    share_one_tick: float           # share of time the spread is exactly one tick
    depth_best: float               # time-weighted mean of (bid size + ask size) / 2 at the best
    n_messages: int
    n_by_type: dict[int, int]
    n_visible_exec: int             # type-4 messages
    n_hidden_exec: int              # type-5 messages
    n_trades: int                   # visible executions grouped by (timestamp, side)
    volume_visible: int             # shares executed against displayed orders
    sign_acf: list[float]           # trade-sign autocorrelation, lags 1..ACF_LAGS
    tick_class: str                 # "large-tick" or "small-tick"

    def to_dict(self) -> dict:
        return asdict(self)


def session_mask(time: np.ndarray, session: tuple[float, float] = SESSION) -> np.ndarray:
    return (time >= session[0]) & (time < session[1])


def time_weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    ok = np.isfinite(x) & (w > 0)
    return float(np.sum(x[ok] * w[ok]) / np.sum(w[ok]))


def trades(day: Day, session: tuple[float, float] = SESSION) -> dict[str, np.ndarray]:
    """Visible executions grouped into trades.

    One marketable order that hits several resting orders leaves several type-4 rows
    with the same timestamp and direction; grouping them gives one trade per aggressive
    order (sizes summed, price = share-weighted average). The trade sign is +1 for a
    buyer-initiated trade, i.e. the execution of a *sell* limit order (direction -1).

    Returns first/last message index of each group, sign, size, VWAP (price units), time.
    """
    m = day.messages
    ex = np.flatnonzero((m.type == MsgType.EXECUTE) & session_mask(m.time, session))
    if len(ex) == 0:
        empty = np.array([], dtype=np.int64)
        return {"first": empty, "last": empty, "sign": empty, "size": empty,
                "vwap": np.array([]), "time": np.array([])}
    new = np.ones(len(ex), dtype=bool)
    new[1:] = (m.time[ex[1:]] != m.time[ex[:-1]]) | (m.direction[ex[1:]] != m.direction[ex[:-1]]) \
        | (ex[1:] != ex[:-1] + 1)
    gid = np.cumsum(new) - 1
    first = ex[new]
    last = ex[np.append(np.flatnonzero(new)[1:] - 1, len(ex) - 1)]
    size = np.bincount(gid, weights=m.size[ex]).astype(np.int64)
    notional = np.bincount(gid, weights=m.size[ex] * m.price[ex].astype(np.float64))
    return {
        "first": first,
        "last": last,
        "sign": -m.direction[first].astype(np.int64),
        "size": size,
        "vwap": notional / size,
        "time": m.time[first],
    }


def autocorrelation(x: np.ndarray, lags: int = ACF_LAGS) -> list[float]:
    """Sample autocorrelation rho(k) = sum (x_i - xbar)(x_{i+k} - xbar) / sum (x_i - xbar)^2."""
    x = np.asarray(x, dtype=np.float64) - np.mean(x)
    denom = float(np.dot(x, x))
    return [float(np.dot(x[:-k], x[k:]) / denom) for k in range(1, lags + 1)]


def compute_facts(day: Day, session: tuple[float, float] = SESSION) -> Facts:
    m, b = day.messages, day.book
    w = state_durations(m.time, *session)
    mid = b.mid()
    spread = b.spread()
    depth = 0.5 * (b.bid_size[:, 0] + b.ask_size[:, 0]).astype(np.float64)
    in_s = session_mask(m.time, session)
    types, counts = np.unique(m.type[in_s], return_counts=True)
    tr = trades(day, session)
    share_one = time_weighted_mean((spread == TICK).astype(np.float64), w)
    return Facts(
        ticker=day.ticker,
        mean_price_usd=time_weighted_mean(mid, w) / 1e4,
        tick_bps=time_weighted_mean(TICK / mid * 1e4, w),
        spread_ticks=time_weighted_mean(spread / TICK, w),
        spread_bps=time_weighted_mean(spread / mid * 1e4, w),
        share_one_tick=share_one,
        depth_best=time_weighted_mean(depth, w),
        n_messages=int(in_s.sum()),
        n_by_type={int(k): int(v) for k, v in zip(types, counts)},
        n_visible_exec=int(np.sum(in_s & (m.type == MsgType.EXECUTE))),
        n_hidden_exec=int(np.sum(in_s & (m.type == MsgType.HIDDEN))),
        n_trades=len(tr["sign"]),
        volume_visible=int(tr["size"].sum()),
        sign_acf=autocorrelation(tr["sign"]) if len(tr["sign"]) > ACF_LAGS else [],
        tick_class="large-tick" if share_one > LARGE_TICK_THRESHOLD else "small-tick",
    )
