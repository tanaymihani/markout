"""From calibrated edges to trades to daily PnL.

Decisions are taken every 60 s (0, 60, ..., 540 s) and held exactly 60 s, so
positions never overlap and each day's trades are independent round trips.
Forecasts exist every 10 s, but using all of them would stack six overlapping
positions and make daily PnL autocorrelated, which inflates any naive Sharpe.

The decision rule: trade a stock only if |edge| > threshold * cost. Sizing is either
flat, or a ramp (fractional-Kelly-like): full size once the edge exceeds cost by
`ramp * cost`. Every trade is capped by a share of the displayed size at the touch,
and gross exposure per instant is capped too.

PnL of a trade = notional * (side * target - cost) / 1e4, where target is the
index-relative 60 s move, so the PnL assumes the index hedge is available.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import polars as pl

from markout.backtest.costs import CostModel

ID = ["stock_id", "date_id", "seconds_in_bucket"]
INSTANT = ["date_id", "seconds_in_bucket"]


@dataclass(frozen=True)
class Policy:
    threshold: float = 1.0          # trade if |edge| > threshold * cost
    sizing: str = "flat"            # "flat" or "ramp"
    ramp: float = 1.0               # ramp sizing reaches full size at edge = (1 + ramp) * cost
    max_notional: float = 100_000.0 # USD per trade
    depth_frac: float = 0.10        # at most this share of the displayed touch size
    max_gross: float = 5_000_000.0  # USD per instant; also the capital base for returns
    every: int = 60                 # seconds between decisions (= holding period)

    def to_dict(self) -> dict:
        return asdict(self)


def decision_frame(edges: pl.DataFrame, market: pl.DataFrame, every: int = 60) -> pl.DataFrame:
    """Rows at decision times with the entry/exit spread and touch sizes attached.

    `edges` has ID + edge (+ anything else); `market` is the raw data (ID + target +
    bid/ask prices and sizes). The exit spread is the spread 60 s later; at 540 s the
    exit (600 s) is not in the data, so the entry spread is used for it.
    """
    spread = ((pl.col("ask_price") - pl.col("bid_price")) / ((pl.col("ask_price") + pl.col("bid_price")) / 2) * 1e4)
    m = market.select(ID + ["target", "bid_size", "ask_size"]).with_columns(market.select(spread.alias("spread_bps")))
    exit_ = m.select(["stock_id", "date_id", (pl.col("seconds_in_bucket") - every).alias("seconds_in_bucket"),
                      pl.col("spread_bps").alias("spread_exit_bps")])
    d = (edges.filter(pl.col("seconds_in_bucket") % every == 0)
              .join(m, on=ID, how="inner")
              .join(exit_, on=ID, how="left")
              .rename({"spread_bps": "spread_entry_bps"})
              .with_columns(pl.col("spread_exit_bps").fill_null(pl.col("spread_entry_bps")))
              .drop_nulls(["target", "edge", "spread_entry_bps"]))
    return d.sort(ID)


def trades(dec: pl.DataFrame, policy: Policy, costs: CostModel, decide_mult: float = 1.0,
           pay_mult: float | None = None, edge_col: str = "edge") -> pl.DataFrame:
    """Apply the decision rule. `decide_mult` scales the cost the rule believes in;
    `pay_mult` scales the cost actually paid (defaults to the same)."""
    pay_mult = decide_mult if pay_mult is None else pay_mult
    d = dec.with_columns(costs.expr(decide_mult).alias("cost_decide"), costs.expr(pay_mult).alias("cost_bps"))
    e = pl.col(edge_col)
    d = d.filter(e.abs() > policy.threshold * pl.col("cost_decide"))
    side = pl.when(e > 0).then(1.0).otherwise(-1.0)
    touch = pl.when(e > 0).then(pl.col("ask_size")).otherwise(pl.col("bid_size"))
    flat = pl.min_horizontal(pl.lit(policy.max_notional), policy.depth_frac * touch)
    if policy.sizing == "ramp":
        excess = (e.abs() - pl.col("cost_decide")) / (policy.ramp * pl.col("cost_decide"))
        size = flat * excess.clip(0.0, 1.0)
    elif policy.sizing == "flat":
        size = flat
    else:
        raise ValueError(policy.sizing)
    d = d.with_columns(side.alias("side"), size.alias("notional")).filter(pl.col("notional") > 0)
    gross = pl.col("notional").sum().over(INSTANT)
    scale = pl.min_horizontal(pl.lit(1.0), policy.max_gross / gross)
    d = d.with_columns((pl.col("notional") * scale).alias("notional"))
    return d.with_columns(
        (pl.col("notional") * pl.col("side") * pl.col("target") / 1e4).alias("gross_usd"),
        (pl.col("notional") * pl.col("cost_bps") / 1e4).alias("cost_usd"),
    ).with_columns((pl.col("gross_usd") - pl.col("cost_usd")).alias("pnl_usd"))


def daily(tr: pl.DataFrame, days: tuple[int, int]) -> pl.DataFrame:
    """Daily PnL over every day in `days`, zero on days without trades."""
    agg = tr.group_by("date_id").agg(
        pl.len().alias("n_trades"),
        pl.col("notional").sum().alias("notional"),
        pl.col("gross_usd").sum(), pl.col("cost_usd").sum(), pl.col("pnl_usd").sum(),
        (pl.col("pnl_usd") > 0).mean().alias("hit_rate"),
        (pl.col("edge").abs() * pl.col("notional")).sum().alias("_pred_w"),
        (pl.col("side") * pl.col("target") * pl.col("notional")).sum().alias("_real_w"),
    )
    all_days = pl.DataFrame({"date_id": np.arange(days[0], days[1] + 1)}, schema={"date_id": tr.schema["date_id"]
                            if "date_id" in tr.schema else pl.Int16})
    out = all_days.join(agg, on="date_id", how="left").with_columns(
        [pl.col(c).fill_null(0) for c in ["n_trades", "notional", "gross_usd", "cost_usd", "pnl_usd"]]
    ).sort("date_id")
    return out.with_columns(pl.col("pnl_usd").cum_sum().alias("cum_pnl_usd")).with_columns(
        (pl.col("cum_pnl_usd") - pl.col("cum_pnl_usd").cum_max().clip(lower_bound=0.0)).alias("drawdown_usd"))


def summarize(day: pl.DataFrame, tr: pl.DataFrame, capital: float) -> dict:
    """Headline metrics. Sharpe is per day (non-annualized) plus x sqrt(252) for display."""
    pnl = day["pnl_usd"].to_numpy().astype(float)
    sd = pnl.std(ddof=1) if len(pnl) > 1 else 0.0
    sr = float(pnl.mean() / sd) if sd > 0 else 0.0
    gross, cost = float(tr["gross_usd"].sum()) if tr.height else 0.0, float(tr["cost_usd"].sum()) if tr.height else 0.0
    notional = float(tr["notional"].sum()) if tr.height else 0.0
    return {
        "days": len(pnl),
        "total_pnl_usd": float(pnl.sum()),
        "mean_daily_pnl_usd": float(pnl.mean()),
        "std_daily_pnl_usd": float(sd),
        "sharpe_daily": sr,
        "sharpe_annualized": sr * np.sqrt(252),
        "max_drawdown_usd": float(day["drawdown_usd"].min()),
        "trades": int(tr.height),
        "trades_per_day": float(tr.height / max(len(pnl), 1)),
        "hit_rate": float((tr["pnl_usd"] > 0).mean()) if tr.height else 0.0,
        "turnover_usd_per_day": float(2 * notional / max(len(pnl), 1)),
        "gross_usd": gross,
        "cost_usd": cost,
        "cost_share_of_gross": float(cost / gross) if gross > 0 else None,
        "edge_pred_bps": float((tr["edge"].abs() * tr["notional"]).sum() / notional) if notional else None,
        "edge_realized_bps": float((tr["side"] * tr["target"] * tr["notional"]).sum() / notional) if notional else None,
        "net_bps_per_trade": float(1e4 * pnl.sum() / notional) if notional else None,
        "capital_usd": capital,
        "return_on_capital_daily": float(pnl.mean() / capital),
    }


def backtest(dec: pl.DataFrame, policy: Policy, costs: CostModel, days: tuple[int, int],
             decide_mult: float = 1.0, pay_mult: float | None = None, edge_col: str = "edge"):
    tr = trades(dec, policy, costs, decide_mult, pay_mult, edge_col)
    day = daily(tr, days)
    return tr, day, summarize(day, tr, policy.max_gross)


def kelly_table(daily_pnl: np.ndarray, capital: float, fractions=(0.25, 0.5, 1.0, 2.0),
                n_boot: int = 2000, block: int = 5, seed: int = 0) -> list[dict]:
    """Leverage choice as a decision under estimation error.

    Continuous-time Kelly leverage on the daily return stream is L* = mu / sigma^2.
    For each fraction of L*, report the in-sample growth rate and, across stationary
    block-bootstrap resamples of the days, the distribution of growth and of max
    drawdown, i.e. what happens when the mu used to pick L* was an overestimate.
    """
    r = np.asarray(daily_pnl, float) / capital
    mu, var = r.mean(), r.var(ddof=1)
    if var <= 0:
        return []
    lstar = mu / var
    rng = np.random.default_rng(seed)
    T = len(r)
    idx = np.empty((n_boot, T), dtype=int)
    for b in range(n_boot):  # stationary bootstrap indices (Politis & Romano 1994)
        i, t = rng.integers(T), 0
        while t < T:
            idx[b, t] = i
            t += 1
            i = rng.integers(T) if rng.random() < 1.0 / block else (i + 1) % T
    rows = []
    for f in fractions:
        L = f * lstar
        g_in = float(np.mean(np.log1p(np.clip(L * r, -0.999999, None))))
        sims = np.clip(L * r[idx], -0.999999, None)
        growth = np.log1p(sims).mean(axis=1)
        wealth = np.cumprod(1 + sims, axis=1)
        dd = (wealth / np.maximum.accumulate(wealth, axis=1) - 1).min(axis=1)
        rows.append({"kelly_fraction": f, "leverage": float(L), "growth_daily_in_sample": g_in,
                     "growth_daily_p05": float(np.quantile(growth, 0.05)),
                     "growth_daily_median": float(np.median(growth)),
                     "max_drawdown_median": float(np.median(dd)),
                     "max_drawdown_p05": float(np.quantile(dd, 0.05))})
    return rows
