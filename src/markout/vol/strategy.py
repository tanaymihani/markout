"""A stylized short-variance strategy: every 21 trading days, sell one month of variance.

Per $1 of vega notional, a short variance swap struck at K (vol points) that realizes R
pays
    pnl = (K^2 - R^2) / (2K) - cost,
which is zero when R = K, at most K/2 when nothing moves, and unbounded below: selling
variance is selling insurance. K is VIX at the roll (VIX^2 is the 30-day variance-swap
rate it replicates) and R is the S&P 500's realized vol over the next 21 trading days.

Timing rules, all using only what is known at the roll:
- always:    sell every month (the baseline);
- contango:  sell only when VIX < VIX3M (an upward-sloping term structure; inversion
             signals stress);
- har(tau):  sell only when the premium the HAR forecast implies, IV^2 - forecast RV,
             exceeds tau * IV^2.
Sizing is studied separately: the growth-optimal (Kelly) vega per dollar of capital is
computed exactly on the empirical monthly outcomes, because mu / sigma^2 is badly wrong
for a payoff with a fat left tail.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from markout.vol.vrp import H

ROLLS_PER_YEAR = 252 / H


@dataclass(frozen=True)
class Rule:
    name: str
    use_contango: bool = False
    har_tau: float | None = None

    def trade(self, row) -> bool:
        if self.use_contango and not (pd.notna(row.vix3m) and row.vix < row.vix3m):
            return False
        if self.har_tau is not None:
            if pd.isna(row.har):
                return False
            if row.iv2 - row.har <= self.har_tau * row.iv2:
                return False
        return True


RULES = [Rule("always"), Rule("contango", use_contango=True), Rule("har (premium > 0)", har_tau=0.0),
         Rule("har (premium > 25%)", har_tau=0.25), Rule("har + contango", use_contango=True, har_tau=0.0)]


def short_var_pnl(k_vol: np.ndarray, r_vol: np.ndarray, cost_vol: float = 0.0) -> np.ndarray:
    k_vol, r_vol = np.asarray(k_vol, float), np.asarray(r_vol, float)
    return (k_vol ** 2 - r_vol ** 2) / (2 * k_vol) - cost_vol


def rolls(fr: pd.DataFrame, start: str, end: str | None = None) -> pd.DataFrame:
    """Non-overlapping roll dates every H trading days with a complete forward window."""
    d = fr.loc[start:end].dropna(subset=["rv"])
    return d.iloc[::H]


def run(fr: pd.DataFrame, rule: Rule, start: str, end: str | None = None, cost_vol: float = 0.5) -> pd.DataFrame:
    r = rolls(fr, start, end)
    traded = np.array([rule.trade(row) for row in r.itertuples()])
    pnl = np.where(traded, short_var_pnl(r["vix"].to_numpy(), r["rv_vol"].to_numpy(), cost_vol), 0.0)
    return pd.DataFrame({"vix": r["vix"], "rv_vol": r["rv_vol"], "traded": traded, "pnl": pnl}, index=r.index)


def stats(res: pd.DataFrame) -> dict:
    p = res["pnl"].to_numpy()
    t = res["traded"].to_numpy()
    sd = p.std(ddof=1)
    cum = np.cumsum(p)
    dd = cum - np.maximum.accumulate(np.maximum(cum, 0.0))
    worst = res["pnl"].idxmin()
    tail = np.sort(p)[: max(1, int(0.05 * len(p)))]
    return {
        "rolls": int(len(p)), "traded": int(t.sum()), "share_traded": float(t.mean()),
        "mean_pnl": float(p.mean()), "sd_pnl": float(sd),
        "sharpe_ann": float(p.mean() / sd * np.sqrt(ROLLS_PER_YEAR)) if sd > 0 else 0.0,
        "skew": float(((p - p.mean()) ** 3).mean() / sd ** 3) if sd > 0 else 0.0,
        "hit_rate": float((p[t] > 0).mean()) if t.any() else 0.0,
        "worst_pnl": float(p.min()), "worst_date": str(worst.date()),
        "cvar5": float(tail.mean()), "max_drawdown": float(dd.min()), "total": float(cum[-1]),
    }


def kelly_exact(pnl: np.ndarray, grid: np.ndarray | None = None) -> dict:
    """Growth-optimal vega per $1 of capital, v*, maximizing mean log(1 + v * pnl) over the
    empirical outcomes. v must keep 1 + v * min(pnl) > 0: the worst month bounds it."""
    pnl = np.asarray(pnl, float)
    vmax = 0.999 / max(-pnl.min(), 1e-9)
    grid = np.linspace(0, vmax, 2001) if grid is None else grid
    g = np.array([np.mean(np.log1p(v * pnl)) for v in grid])
    i = int(np.argmax(g))
    mu, var = pnl.mean(), pnl.var(ddof=1)
    return {"v_star": float(grid[i]), "growth_star": float(g[i]), "v_max": float(vmax),
            "v_continuous": float(mu / var) if var > 0 else None}


def wealth_path(pnl: np.ndarray, v: float) -> np.ndarray:
    """Wealth after each month at v dollars of vega per dollar of capital. Ruin is absorbing:
    once a month's loss exceeds the account, wealth stays at zero (two losing months must
    not multiply back to a positive number)."""
    growth = np.maximum(1 + v * np.asarray(pnl, float), 0.0)
    return np.cumprod(growth)


def max_drawdown(w: np.ndarray) -> float:
    w = np.asarray(w, float)
    peak = np.maximum.accumulate(np.maximum(w, 1e-300))
    return float((w / peak - 1).min())
