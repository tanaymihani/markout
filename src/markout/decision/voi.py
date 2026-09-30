"""Value of information: how good would a forecast need to be to beat costs?

Two reference points bound the answer:
- the oracle, which knows the target exactly, is the most any model could extract
  under the cost model (the expected value of perfect information);
- synthetic forecasts with a chosen correlation rho with the target,
      f = rho * z + sqrt(1 - rho^2) * eps,   z = standardized target,
  perfectly calibrated (edge = E[target | f] by quantile bin), pushed through the
  same decision rule. Where the real model's IC falls on this
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


def with_synthetic(dec: pl.DataFrame, rho: float, seed: int = 0, n_bins: int = 100) -> pl.DataFrame:
    """Synthetic forecast with correlation rho, *perfectly calibrated*: its edge is the mean
    (volatility-standardized) target within its quantile bin, rescaled by the stock's own
    volatility, i.e. E[target | f, stock], estimated on the same rows.

    Linear scaling rho*sigma*f is exact only under joint normality. With these fat-tailed
    targets it overstates E[target | f] exactly where trading starts, so a benchmark built
    that way would lose money for reasons unrelated to information. The binned mean is the
    ideal calibration (in-sample by design: this is a yardstick, not a strategy)."""
    y = dec["target"].to_numpy().astype(float)
    # standardize by each stock's own volatility: the rule trades cheap (tight-spread, calm)
    # stocks more, and a pooled calibration would overstate their edge (a selection effect)
    if "stock_id" in dec.columns:
        sd = dec.select(pl.col("target").std().over("stock_id").alias("s"))["s"].to_numpy().astype(float)
        sd = np.where(np.isfinite(sd) & (sd > 0), sd, y.std())
    else:
        sd = np.full(len(y), y.std())
    k = ((y - y.mean()) / sd).std()
    z = (y - y.mean()) / (sd * k)  # unit variance, comparable across stocks
    eps = np.random.default_rng(seed).standard_normal(len(y))
    f = rho * z + np.sqrt(max(1 - rho ** 2, 0.0)) * eps
    if rho == 0:
        return dec.with_columns(pl.Series("edge", np.zeros(len(y))))
    edges = np.quantile(f, np.linspace(0, 1, n_bins + 1))
    idx = np.clip(np.searchsorted(edges, f, side="right") - 1, 0, n_bins - 1)
    m = np.bincount(idx, weights=z, minlength=n_bins) / np.maximum(np.bincount(idx, minlength=n_bins), 1)
    # E[target | f, stock] = mean + (stock volatility) * k * E[z | f]
    return dec.with_columns(pl.Series("edge", y.mean() + sd * k * m[idx]), pl.Series("_f", f))


def top_decile(dec: pl.DataFrame, frac: float = 0.10) -> pl.DataFrame:
    """Keep the top `frac` of |forecast| at each instant (the cost-blind rule's picks)."""
    col = "_f" if "_f" in dec.columns else "edge"
    r = pl.col(col).abs().rank(descending=True, method="ordinal").over(INSTANT) / pl.len().over(INSTANT)
    return dec.filter(r <= frac)


def curve(dec: pl.DataFrame, policy: Policy, costs: CostModel, days: tuple[int, int],
          rhos: np.ndarray, mults=(1.0, 2.0, 3.0), seed: int = 0) -> list[dict]:
    rows = []
    for rho in rhos:
        syn = with_synthetic(dec, float(rho), seed)
        blind = top_decile(syn)
        for m in mults:
            _, _, s_thr = backtest(syn, policy, costs, days, decide_mult=m)
            # cost-blind: every pre-selected decile row, flat size (a ramp would re-introduce the cost filter)
            _, _, s_blind = backtest(blind, Policy(**{**policy.to_dict(), "threshold": 0.0, "sizing": "flat"}),
                                     costs, days, decide_mult=m)
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
