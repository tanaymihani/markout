"""Markouts: what the mid did after a (hypothetical or real) trade.

A markout at horizon h of a buy at price P is mid(t + h) - P (mirror for a sell). For a
passive order it starts near +half a spread (we bought below the mid) and decays as
the price moves against us. The decay is adverse selection: we are filled when an
aggressor who knows something wants to trade with us, so *conditional on being filled*
the price tends to move the wrong way. That is the passive trader's winner's curse,
and the gap between the filled and the unconditional markout measures it.

* `sample_orders` places a 100-share order at the best bid and one at the best ask
  every 5 s of the trimmed session, each alive for at most 60 s after its decision time.
* `order_markouts` marks every filled order to the mid at +100 ms, 1 s, 10 s and 60 s
  after its (last) fill, and every order, filled or not, as if it had been filled
  instantly at its decision time: the unconditional baseline.
* `spread_decomposition` does the classic split of the file's own trades,
  effective spread = realized spread + price impact, at 5 s and 60 s.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from markout.lob.facts import trades
from markout.lob.fills import MODELS, Orders, simulate
from markout.lob.lobster import SESSION, TICK, Day, state_index

HORIZONS: tuple[float, ...] = (0.1, 1.0, 10.0, 60.0)
EVERY = 5.0          # seconds between sampled decision times
MAX_WAIT = 60.0      # seconds an order may rest
SIZE = 100           # shares
QUEUE_MULTS = (0.5, 1.0, 2.0)   # queue-ahead bucket edges, in multiples of the median queue
QUEUE_LABELS = ("<0.5x", "0.5-1x", "1-2x", ">=2x")
SPREAD_HORIZONS = (5.0, 60.0)


def mid_at(day: Day, t: np.ndarray) -> np.ndarray:
    """Mid (price units) of the state in effect at clock times t."""
    return day.book.mid()[state_index(day.messages.time, t)]


@dataclass(frozen=True)
class Sample:
    orders: Orders
    decision_time: np.ndarray      # clock time of each decision (>= time[place_idx])


def sample_orders(day: Day, every: float = EVERY, max_wait: float = MAX_WAIT,
                  size: int = SIZE, horizon: float = max(HORIZONS),
                  session: tuple[float, float] = SESSION) -> Sample:
    """Both sides at the best quotes on a regular grid. The last decision leaves room
    for the longest wait plus the longest markout inside the session. `max_wait` is
    counted from the decision time: the order rests until t + max_wait."""
    m, b = day.messages, day.book
    t = np.arange(session[0], session[1] - max_wait - horizon + 1e-9, every)
    i = state_index(m.time, t)
    t2, i2 = np.concatenate([t, t]), np.concatenate([i, i])
    side = np.repeat(np.array([1, -1], dtype=np.int8), len(t))
    price = np.where(side > 0, b.bid_price[i2, 0], b.ask_price[i2, 0])
    wait = t2 + max_wait - m.time[i2]
    return Sample(Orders.make(i2, side, price, size, wait), t2)


def queue_buckets(queue: np.ndarray, median: float) -> np.ndarray:
    """0..3 for queue ahead < 0.5x, 0.5-1x, 1-2x, >= 2x the median queue."""
    return np.searchsorted(np.array(QUEUE_MULTS) * median, queue, side="right")


def order_markouts(day: Day, sample: Sample, models: tuple[str, ...] = MODELS,
                   engine: str = "auto") -> pd.DataFrame:
    """One row per order and model with fill outcome and markouts (ticks and bps)."""
    o = sample.orders
    mid0 = mid_at(day, sample.decision_time)
    base = pd.DataFrame({
        "order": np.arange(len(o)),
        "decision_time": sample.decision_time,
        "side": o.side,
        "price": o.price,
        "mid0": mid0,
    })
    to_bps = TICK / mid0 * 1e4
    for h in HORIZONS:   # unconditional: an instant fill at the decision time
        base[f"uncond_{h:g}"] = o.side * (mid_at(day, sample.decision_time + h) - o.price) / TICK
        base[f"uncond_{h:g}_bps"] = base[f"uncond_{h:g}"] * to_bps
    frames = []
    for model in models:
        f = simulate(day.messages, day.book, o, model, engine)
        df = base.copy()
        df["model"] = model
        df["filled_qty"] = f.filled_qty
        df["fill_frac"] = f.filled_qty / o.size
        df["fill_time"] = f.fill_time
        df["end_reason"] = f.end_reason
        df["queue_ahead"] = f.queue_ahead
        filled = f.filled_qty > 0
        for h in HORIZONS:
            mk = np.full(len(o), np.nan)
            mk[filled] = o.side[filled] * (mid_at(day, f.fill_time[filled] + h)
                                           - o.price[filled]) / TICK
            df[f"mk_{h:g}"] = mk
            df[f"mk_{h:g}_bps"] = mk * to_bps
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["queue_bucket"] = queue_buckets(out["queue_ahead"].to_numpy(),
                                        float(np.median(base_queue(out))))
    return out


def base_queue(df: pd.DataFrame) -> np.ndarray:
    """Queue ahead at placement (identical across models), one value per order."""
    return df.loc[df["model"] == df["model"].iloc[0], "queue_ahead"].to_numpy()


def _wmean(x: pd.Series, w: pd.Series) -> float:
    ok = x.notna() & (w > 0)
    return float(np.average(x[ok], weights=w[ok])) if ok.any() else float("nan")


def summarize_markouts(df: pd.DataFrame) -> list[dict]:
    """Per model: P(fill), quantity-weighted markouts of filled orders, and the
    unconditional markouts, in ticks and bps (each order converted at its own mid)."""
    rows = []
    for model, g in df.groupby("model", sort=False):
        r = {"model": model, "n_orders": int(len(g)), "p_fill": float((g["filled_qty"] > 0).mean()),
             "fill_frac": float(g["fill_frac"].mean())}
        for h in HORIZONS:
            r[f"mk_{h:g}"] = _wmean(g[f"mk_{h:g}"], g["filled_qty"])
            r[f"uncond_{h:g}"] = float(g[f"uncond_{h:g}"].mean())
            r[f"mk_{h:g}_bps"] = _wmean(g[f"mk_{h:g}_bps"], g["filled_qty"])
            r[f"uncond_{h:g}_bps"] = float(g[f"uncond_{h:g}_bps"].mean())
        rows.append(r)
    return rows


def by_queue_bucket(df: pd.DataFrame, horizon: float = 10.0) -> list[dict]:
    rows = []
    for (model, qb), g in df.groupby(["model", "queue_bucket"], sort=False):
        rows.append({"model": model, "queue_bucket": int(qb), "label": QUEUE_LABELS[int(qb)],
                     "n": int(len(g)), "p_fill": float((g["filled_qty"] > 0).mean()),
                     "mk": _wmean(g[f"mk_{horizon:g}"], g["filled_qty"]),
                     "median_queue": float(g["queue_ahead"].median())})
    return sorted(rows, key=lambda r: (MODELS.index(r["model"]), r["queue_bucket"]))


def spread_decomposition(day: Day, horizons: tuple[float, ...] = SPREAD_HORIZONS,
                         session: tuple[float, float] = SESSION) -> list[dict]:
    """Effective = realized + impact on the file's visible trades (share-weighted).

    With trade sign e, price P, mid m0 just before the trade and m_h at t + h:
      effective spread = 2 e (P - m0), realized = 2 e (P - m_h), impact = 2 e (m_h - m0).
    The identity holds trade by trade. bps are relative to m0.
    """
    tr = trades(day, (session[0], session[1] - max(horizons)))
    mid = day.book.mid()
    m0 = mid[tr["first"] - 1]
    e, px, w = tr["sign"], tr["vwap"], tr["size"].astype(np.float64)
    rows = []
    for h in horizons:
        mh = mid_at(day, tr["time"] + h)
        eff, real, imp = 2 * e * (px - m0), 2 * e * (px - mh), 2 * e * (mh - m0)
        row = {"horizon": h, "n_trades": int(len(e))}
        for name, x in (("effective", eff), ("realized", real), ("impact", imp)):
            row[f"{name}_ticks"] = float(np.average(x, weights=w) / TICK)
            row[f"{name}_bps"] = float(np.average(x / m0 * 1e4, weights=w))
        rows.append(row)
    return rows
