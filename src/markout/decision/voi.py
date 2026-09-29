"""Value of information: how good would a forecast need to be to beat costs?

Two reference points bound the answer:
- the oracle, which knows the target exactly, is the most any model could extract
  under the cost model (the expected value of perfect information);
- synthetic forecasts with a chosen correlation rho with the target,
      f = rho * z + sqrt(1 - rho^2) * eps,   z = standardized target,
  scaled so that E[target | f] = rho * sigma * f (exact under joint normality),
  pushed through the same decision rule. Where the real model's IC falls on this
  curve says how far it is from paying for its spread.

Two rules are compared. The cost-aware threshold rule never takes a trade whose
calibrated edge is below cost, so with a correctly calibrated forecast its
expected PnL is never negative: low rho simply means few trades. The cost-blind
rule trades the top decile of |f| at every instant regardless of cost; its net bps
per trade crosses zero at a well-defined break-even IC.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from markout.backtest.costs import CostModel
from markout.backtest.engine import INSTANT, Policy, backtest


def with_oracle(dec: pl.DataFrame) -> pl.DataFrame:
    return dec.with_columns(pl.col("target").cast(pl.Float64).alias("edge"))


def with_synthetic(dec: pl.DataFrame, rho: float, seed: int = 0) -> pl.DataFrame:
    y = dec["target"].to_numpy().astype(float)
    sigma = y.std()
    z = (y - y.mean()) / sigma
    eps = np.random.default_rng(seed).standard_normal(len(y))
    f = rho * z + np.sqrt(max(1 - rho ** 2, 0.0)) * eps
    return dec.with_columns(pl.Series("edge", rho * sigma * f))


def top_decile(dec: pl.DataFrame, frac: float = 0.10) -> pl.DataFrame:
    """Keep the top `frac` of |edge| at each instant (the cost-blind rule's picks)."""
    r = pl.col("edge").abs().rank(descending=True).over(INSTANT) / pl.len().over(INSTANT)
    return dec.filter(r <= frac)


def curve(dec: pl.DataFrame, policy: Policy, costs: CostModel, days: tuple[int, int],
          rhos: np.ndarray, mults=(1.0, 2.0, 3.0), seed: int = 0) -> list[dict]:
    rows = []
    for rho in rhos:
        syn = with_synthetic(dec, float(rho), seed)
        blind = top_decile(syn)
        for m in mults:
            _, _, s_thr = backtest(syn, policy, costs, days, decide_mult=m)
            # cost-blind: threshold 0 on the pre-selected decile, paying m x cost
            _, _, s_blind = backtest(blind, Policy(**{**policy.to_dict(), "threshold": 0.0}), costs, days,
                                     decide_mult=m)
            rows.append({"rho": float(rho), "cost_mult": m,
                         "threshold_sharpe_ann": s_thr["sharpe_annualized"],
                         "threshold_trades_per_day": s_thr["trades_per_day"],
                         "threshold_net_bps": s_thr["net_bps_per_trade"],
                         "blind_net_bps": s_blind["net_bps_per_trade"],
                         "blind_sharpe_ann": s_blind["sharpe_annualized"]})
    return rows


def break_even(xs, ys) -> float | None:
    """First x where y crosses from <= 0 to > 0 (linear interpolation)."""
    xs, ys = np.asarray(xs, float), np.asarray([np.nan if y is None else y for y in ys], float)
    for i in range(1, len(xs)):
        if np.isfinite(ys[i - 1]) and np.isfinite(ys[i]) and ys[i - 1] <= 0 < ys[i]:
            return float(xs[i - 1] + (0 - ys[i - 1]) * (xs[i] - xs[i - 1]) / (ys[i] - ys[i - 1]))
    if np.isfinite(ys[0]) and ys[0] > 0:
        return float(xs[0])
    return None
