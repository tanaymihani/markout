"""Synthetic closing-auction data with exactly the Optiver schema.

Used only for tests and for developing the pipeline before the real data is
available. Nothing in the reports is computed from it.

The generator is built so the pipeline has something real to find and to check:
- log-mid increments = market factor + a small drift driven by the lagged auction
  imbalance + fat-tailed noise, so imbalance features carry a known signal;
- WAP is computed from the book with the real formula, prices are normalized to
  the WAP at 0 s, near/far prices are null before 300 s;
- the target is the 60 s WAP move minus a fixed-weight index move, with the index
  computed over all stocks before some stock-days are dropped, exactly the
  situation the index-weight reconstruction has to handle.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from markout.auction.load import SCHEMA

STEPS_OBSERVED = 55  # 0..540 s
STEPS_TOTAL = 61  # 0..600 s; the last six only exist to build targets


def make_synthetic(n_stocks: int = 30, n_days: int = 120, seed: int = 0,
                   signal_bps: float = 0.35, drop_frac: float = 0.02,
                   null_target_frac: float = 0.001) -> tuple[pl.DataFrame, np.ndarray]:
    """Return (frame with the Optiver schema, true index weights by stock_id)."""
    rng = np.random.default_rng(seed)
    D, S, T = n_days, n_stocks, STEPS_TOTAL
    weights = rng.dirichlet(np.full(S, 2.0))
    vol = rng.uniform(1.0, 3.0, S)  # bps of noise per 10 s step
    beta = rng.uniform(0.5, 1.5, S)
    spread_bps = rng.uniform(1.0, 8.0, S)
    depth_usd = rng.lognormal(np.log(5e4), 0.5, S)

    # auction imbalance: AR(1) within each stock-day
    imb = np.zeros((D, S, T))
    shocks = rng.standard_normal((D, S, T))
    imb[:, :, 0] = shocks[:, :, 0]
    for t in range(1, T):
        imb[:, :, t] = 0.9 * imb[:, :, t - 1] + np.sqrt(1 - 0.81) * shocks[:, :, t]

    mkt = rng.standard_normal((D, 1, T))
    noise = rng.standard_t(4, (D, S, T)) / np.sqrt(2.0)  # unit variance
    drift = np.zeros((D, S, T))
    drift[:, :, 1:] = signal_bps * imb[:, :, :-1]
    dlog = beta[None, :, None] * mkt + drift + vol[None, :, None] * noise
    dlog[:, :, 0] = 0.0
    mid = np.exp(np.cumsum(dlog, axis=2) / 1e4)

    half = (spread_bps[None, :, None] * rng.uniform(0.7, 1.3, (D, S, T))) / 2e4
    bid, ask = mid * (1 - half), mid * (1 + half)
    tilt = np.tanh(0.5 * imb)  # book leans the same way as the auction imbalance
    bid_size = depth_usd[None, :, None] * rng.lognormal(0, 0.4, (D, S, T)) * (1 + 0.5 * tilt)
    ask_size = depth_usd[None, :, None] * rng.lognormal(0, 0.4, (D, S, T)) * (1 - 0.5 * tilt)
    wap = (bid * ask_size + ask * bid_size) / (bid_size + ask_size)

    scale = wap[:, :, :1]  # normalize every price to the WAP at 0 s
    bid, ask, wap, mid = bid / scale, ask / scale, wap / scale, mid / scale

    ret60 = (wap[:, :, 6:] / wap[:, :, :-6] - 1.0) * 1e4  # (D, S, 55)
    index60 = np.einsum("s,dst->dt", weights, ret60)
    target = ret60 - index60[:, None, :]

    matched = rng.lognormal(np.log(2e7), 0.6, (D, S, 1)) * np.linspace(0.6, 1.0, T)[None, None, :]
    imbalance_size = np.abs(imb) * 0.08 * matched
    flag = np.sign(imb).astype(np.int8)
    reference = mid * (1 + 0.5e-4 * imb)
    near = reference * (1 + 0.3e-4 * rng.standard_normal((D, S, T)))
    far = near * (1 + 3e-4 * rng.standard_normal((D, S, T)))
    seconds = np.arange(T) * 10
    near[:, :, seconds < 300] = np.nan
    far[:, :, seconds < 300] = np.nan

    obs = slice(0, STEPS_OBSERVED)
    dd, ss, tt = np.meshgrid(np.arange(D), np.arange(S), seconds[obs], indexing="ij")
    cols = {
        "stock_id": ss, "date_id": dd, "seconds_in_bucket": tt,
        "imbalance_size": imbalance_size[:, :, obs], "imbalance_buy_sell_flag": flag[:, :, obs],
        "reference_price": reference[:, :, obs], "matched_size": matched[:, :, obs],
        "far_price": far[:, :, obs], "near_price": near[:, :, obs],
        "bid_price": bid[:, :, obs], "bid_size": bid_size[:, :, obs],
        "ask_price": ask[:, :, obs], "ask_size": ask_size[:, :, obs],
        "wap": wap[:, :, obs], "target": target,
    }
    df = pl.DataFrame({k: v.reshape(-1) for k, v in cols.items()})
    df = df.with_columns([pl.col(c).cast(t) for c, t in SCHEMA.items()])
    # the real CSV has empty fields (nulls), not NaN
    df = df.with_columns([pl.col(c).fill_nan(None) for c, t in SCHEMA.items() if t == pl.Float32])

    # drop whole stock-days (the index still included them) and a few targets
    keep = rng.random((D, S)) >= drop_frac
    keep_rows = pl.DataFrame({
        "date_id": np.repeat(np.arange(D), S).astype(np.int16),
        "stock_id": np.tile(np.arange(S), D).astype(np.int16),
        "_keep": keep.reshape(-1),
    })
    df = df.join(keep_rows, on=["date_id", "stock_id"]).filter(pl.col("_keep")).drop("_keep")
    null_mask = rng.random(df.height) < null_target_frac
    df = df.with_columns(
        pl.when(pl.Series(null_mask)).then(None).otherwise(pl.col("target")).alias("target")
    )
    return df.sort(["stock_id", "date_id", "seconds_in_bucket"]), weights
