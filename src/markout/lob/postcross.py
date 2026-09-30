"""Post or cross? Monetising the queue-imbalance signal under each fill model.

Every 5 s the queue imbalance I = (qB - qA)/(qB + qA) gives a direction, sign I, and a
strength |I|. The fitted logistic P(up | I) is increasing in I with an intercept close to
zero, so sign I is the model's call except for a sliver of weak signals near I = 0. Following it, a trader buys (I > 0) or sells (I < 0) 100 shares, either

* crossing: take the opposite quote now,  EV(cross) = s (mid_{t+H} - opposite quote_t);
* posting: join the best quote on our side and wait up to H,
                                            EV(post)  = f s (mid_{t+H} - our quote_t),

where s = +1 for a buy, f is the filled fraction under the fill model, and both are
marked to the mid at the common horizon H = 30 s after the decision. An unfilled post
earns nothing (it also misses the move; that opportunity cost is inside the comparison).
Values are in ticks per share; fees and rebates are excluded.

EV(post) - EV(cross) is estimated in cells of signal strength x queue ahead (the
displayed depth we would join, in multiples of the ticker's median). Choosing the better
action cell by cell and scoring it on the same data is the optimizer's curse again, so
the rule is chosen on decisions before 12:45 and scored on decisions after.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from markout.lob.fills import MODELS, Orders, simulate
from markout.lob.imbalance import queue_imbalance
from markout.lob.lobster import SESSION, SPLIT, TICK, Day, state_index
from markout.lob.markouts import QUEUE_LABELS, mid_at, queue_buckets

H = 30.0
EVERY = 5.0
SIZE = 100
SIGNAL_EDGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
SIGNAL_LABELS = ("0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1")
BLOCK = 300.0        # seconds per bootstrap block (autocorrelated, overlapping horizons)
# Fees are excluded from EV. For scale we charge crosses the Reg NMS Rule 610(c) access-fee
# cap ($0.003 per share for quotes >= $1), the most an exchange may charge a taker; posts
# would earn a rebate of similar size, which we conservatively leave out.
TAKER_FEE_USD = 0.003
TAKER_FEE_TICKS = TAKER_FEE_USD / (TICK / 1e4)


def decisions(day: Day, horizon: float = H, every: float = EVERY, engine: str = "auto",
              session: tuple[float, float] = SESSION) -> pd.DataFrame:
    """One row per decision (I != 0) and fill model, with EV(cross) and EV(post) in ticks."""
    m, b = day.messages, day.book
    t = np.arange(session[0], session[1] - horizon + 1e-9, every)
    i = state_index(m.time, t)
    imb = queue_imbalance(b.bid_size[i, 0], b.ask_size[i, 0])
    keep = imb != 0
    t, i, imb = t[keep], i[keep], imb[keep]
    side = np.sign(imb).astype(np.int8)
    post_px = np.where(side > 0, b.bid_price[i, 0], b.ask_price[i, 0])
    cross_px = np.where(side > 0, b.ask_price[i, 0], b.bid_price[i, 0])
    queue = np.where(side > 0, b.bid_size[i, 0], b.ask_size[i, 0])
    mid0, mid_h = mid_at(day, t), mid_at(day, t + horizon)
    ev_cross = side * (mid_h - cross_px) / TICK
    orders = Orders.make(i, side, post_px, SIZE, t + horizon - m.time[i])
    base = pd.DataFrame({
        "t": t, "side": side, "imbalance": imb, "strength": np.abs(imb),
        "signal_bucket": np.clip(np.digitize(np.abs(imb), SIGNAL_EDGES[1:-1], right=True),
                                 0, len(SIGNAL_LABELS) - 1),
        "queue": queue, "queue_bucket": queue_buckets(queue, float(np.median(queue))),
        "spread": (b.ask_price[i, 0] - b.bid_price[i, 0]) / TICK,
        "tick_bps": TICK / mid0 * 1e4,
        "move": side * (mid_h - mid0) / TICK,     # the signal's realised alpha, ticks
        "ev_cross": ev_cross,
    })
    frames = []
    for model in MODELS:
        f = simulate(m, b, orders, model, engine)
        df = base.copy()
        df["model"] = model
        df["fill_frac"] = f.filled_qty / SIZE
        df["ev_post"] = df["fill_frac"] * side * (mid_h - post_px) / TICK
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["diff"] = out["ev_post"] - out["ev_cross"]
    return out


def grid(df: pd.DataFrame, value: str = "diff", min_n: int = 20) -> np.ndarray:
    """Mean of `value` per (queue bucket, signal bucket); NaN where a cell has < min_n."""
    g = np.full((len(QUEUE_LABELS), len(SIGNAL_LABELS)), np.nan)
    stats = df.groupby(["queue_bucket", "signal_bucket"])[value].agg(["mean", "size"])
    for (qb, sb), row in stats.iterrows():
        if row["size"] >= min_n:
            g[int(qb), int(sb)] = row["mean"]
    return g


def cell_table(df: pd.DataFrame) -> list[dict]:
    rows = []
    for (qb, sb), g in df.groupby(["queue_bucket", "signal_bucket"]):
        rows.append({"queue_bucket": QUEUE_LABELS[int(qb)], "signal_bucket": SIGNAL_LABELS[int(sb)],
                     "n": int(len(g)), "p_fill": float((g["fill_frac"] > 0).mean()),
                     "ev_post": float(g["ev_post"].mean()), "ev_cross": float(g["ev_cross"].mean()),
                     "diff": float(g["diff"].mean())})
    return rows


ACTIONS = ("none", "post", "cross")


def choose(train: pd.DataFrame, min_n: int = 20) -> dict[tuple[int, int], str]:
    """Best action per cell on the training decisions (cells with < min_n: no trade)."""
    rule = {}
    for key, g in train.groupby(["queue_bucket", "signal_bucket"]):
        ev = {"none": 0.0, "post": g["ev_post"].mean(), "cross": g["ev_cross"].mean()}
        rule[(int(key[0]), int(key[1]))] = max(ACTIONS, key=ev.get) if len(g) >= min_n else "none"
    return rule


def apply(rule: dict[tuple[int, int], str], df: pd.DataFrame) -> np.ndarray:
    act = [rule.get((int(q), int(s)), "none") for q, s in zip(df["queue_bucket"], df["signal_bucket"])]
    act = np.array(act)
    return np.where(act == "post", df["ev_post"], np.where(act == "cross", df["ev_cross"], 0.0))


def block_bootstrap_ci(t: np.ndarray, x: np.ndarray, block: float = BLOCK, reps: int = 2000,
                       seed: int = 0) -> tuple[float, float]:
    """95% CI of the mean, resampling whole clock blocks (keeps within-block dependence)."""
    if len(x) == 0:
        return float("nan"), float("nan")
    ids = np.floor((t - t.min()) / block).astype(int)
    blocks = [x[ids == k] for k in np.unique(ids)]
    sums = np.array([b.sum() for b in blocks])
    counts = np.array([len(b) for b in blocks])
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(blocks), size=(reps, len(blocks)))
    means = sums[draw].sum(axis=1) / counts[draw].sum(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


@dataclass(frozen=True)
class RuleResult:
    model: str
    in_sample: float        # mean EV per decision of the chosen rule, on the training half
    out_of_sample: float    # the same rule on the test half
    ci: tuple[float, float]
    oos_bps: float
    always_post: float      # test half, for comparison
    always_cross: float
    share_post: float       # of test decisions routed to each action
    share_cross: float
    n_test: int
    per_trade: float        # test EV per decision on which the rule trades (post or cross)
    net_of_fee: float       # test EV per decision after charging TAKER_FEE_TICKS on every cross
    net_ci: tuple[float, float]


def evaluate_rule(df: pd.DataFrame, model: str, split: float = SPLIT, horizon: float = H,
                  seed: int = 0, fee_ticks: float = TAKER_FEE_TICKS) -> RuleResult:
    d = df[df["model"] == model]
    train, test = d[d["t"] + horizon <= split], d[d["t"] >= split]
    rule = choose(train)
    ins, oos = apply(rule, train), apply(rule, test)
    act = np.array([rule.get((int(q), int(s)), "none")
                    for q, s in zip(test["queue_bucket"], test["signal_bucket"])])
    traded = act != "none"
    net = oos - fee_ticks * (act == "cross")
    t = test["t"].to_numpy()
    return RuleResult(
        model=model,
        in_sample=float(ins.mean()),
        out_of_sample=float(oos.mean()),
        ci=block_bootstrap_ci(t, oos, seed=seed),
        oos_bps=float(np.mean(oos * test["tick_bps"].to_numpy())),
        always_post=float(test["ev_post"].mean()),
        always_cross=float(test["ev_cross"].mean()),
        share_post=float(np.mean(act == "post")),
        share_cross=float(np.mean(act == "cross")),
        n_test=int(len(test)),
        per_trade=float(oos[traded].mean()) if traded.any() else float("nan"),
        net_of_fee=float(net.mean()),
        net_ci=block_bootstrap_ci(t, net, seed=seed),
    )
