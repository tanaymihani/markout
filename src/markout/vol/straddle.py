"""Harvesting the premium with options: a short at-the-money straddle, delta-hedged daily.

On the same roll dates as the variance swap in strategy.py, sell a 21-trading-day straddle
struck at the index level, priced at sigma = VIX / 100, and hedge it at every daily close
with Black-Scholes deltas at that same sigma (r = q = 0). Per straddle,

    pnl = premium - |S_T - K| + sum_i delta_i (S_{i+1} - S_i),

and to second order each day adds (1/2) Gamma_i S_i^2 (sigma^2 dt - x_{i+1}^2), with x the
day's return: a bet on variance weighted by dollar gamma. A variance swap weights every day
the same. The straddle's weight is highest near the strike and late in the month, and it
collapses once the index runs away from the strike.

Dividing by the straddle's vega per vol point at the roll puts its PnL in the variance
swap's units, (K^2 - R^2) / (2K) per $1 of vega. The two agree to first order on an
average path: along a diffusion the expected dollar gamma is flat at its starting value,
and half the vega-normalized weight a path pinned at the strike would carry. So the
straddle earns up to twice as much in a month that stays pinned, and loses less when the
index leaves the strike. The `exposure` column measures this: realized dollar-gamma weight
over the average-path value (1 = average; 2 = pinned at the strike all month).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from markout.options import bs
from markout.vol import strategy as S
from markout.vol.vrp import H

DT = 1.0 / 252


def hedge_month(returns: np.ndarray, sigma: float, hedge_bps: float = 0.0) -> dict:
    """One month of a short ATM straddle, delta-hedged at each close.

    `returns` are the month's daily log returns and `sigma` the implied vol (0.20 = 20%).
    The index starts at 1 and every PnL number is divided by the straddle's vega per vol
    point at the roll, so it is per $1 of vega, gross of the option's trading cost."""
    h = len(returns)
    s = np.exp(np.concatenate([[0.0], np.cumsum(returns)]))  # S_0 = 1, ..., S_h
    tau = (h - np.arange(h)) * DT  # time left at each hedge
    delta = 2 * bs.delta(s[:-1], 1.0, tau, 0.0, 0.0, sigma, "call") - 1  # straddle delta
    dollar_gamma = 2 * bs.gamma(s[:-1], 1.0, tau, 0.0, 0.0, sigma) * s[:-1] ** 2
    premium = float(bs.price(1.0, 1.0, tau[0], 0.0, 0.0, sigma, "call")
                    + bs.price(1.0, 1.0, tau[0], 0.0, 0.0, sigma, "put"))
    vega = 2 * float(bs.vega(1.0, 1.0, tau[0], 0.0, 0.0, sigma))  # per 1.00 of vol
    vega_pt = vega / 100
    ds = np.diff(s)
    gross = premium - abs(s[-1] - 1.0) + float(np.sum(delta * ds))
    x = ds / s[:-1]
    attribution = float(np.sum(0.5 * dollar_gamma * (sigma ** 2 * DT - x ** 2)))
    trades = np.abs(np.diff(np.concatenate([[0.0], delta, [0.0]])))  # open, rebalance, close
    hedge_cost = hedge_bps * 1e-4 * float(np.sum(trades * s))
    weight = 0.5 * float(np.sum(dollar_gamma)) * DT  # multiplies (sigma^2 - R_eff^2)
    return {"pnl": (gross - hedge_cost) / vega_pt, "gross": gross / vega_pt,
            "attribution": attribution / vega_pt, "hedge_cost": hedge_cost / vega_pt,
            "exposure": weight / (vega / (2 * sigma)),
            "eff_vol": 100 * float(np.sqrt(np.sum(dollar_gamma * x ** 2) / (np.sum(dollar_gamma) * DT))),
            "premium_per_vega": premium / vega_pt}


def run(fr: pd.DataFrame, start: str, end: str | None = None, cost_vol: float = 0.5,
        hedge_bps: float = 1.0, vol_shift: float = 0.0) -> pd.DataFrame:
    """Every roll from strategy.rolls: the hedged straddle and the variance swap on the same
    month, both per $1 of vega and net of `cost_vol`. `vol_shift` prices (and hedges) the
    straddle that many vol points away from VIX; real ATM options trade below VIX, so the
    realistic shifts are negative."""
    r = S.rolls(fr, start, end)
    pos = fr.index.get_indexer(r.index)
    rets = fr["r"].to_numpy()
    rows = []
    for p, row in zip(pos, r.itertuples()):
        m = hedge_month(rets[p + 1:p + 1 + H], (row.vix + vol_shift) / 100, hedge_bps)
        rows.append({"vix": row.vix, "rv_vol": row.rv_vol, "straddle": m["pnl"] - cost_vol,
                     "var_swap": float(S.short_var_pnl(row.vix, row.rv_vol, cost_vol)),
                     "gross": m["gross"], "attribution": m["attribution"], "hedge_cost": m["hedge_cost"],
                     "exposure": m["exposure"], "eff_vol": m["eff_vol"]})
    return pd.DataFrame(rows, index=r.index)


def as_strategy(res: pd.DataFrame, col: str) -> pd.DataFrame:
    """One instrument's PnL in the shape strategy.stats expects."""
    return pd.DataFrame({"pnl": res[col], "traded": True}, index=res.index)
