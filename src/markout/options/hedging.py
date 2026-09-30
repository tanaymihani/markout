"""Delta-hedging a short call at the implied vol: discrete rebalancing, proportional
transaction costs, and the gamma-theta PnL attribution.

Model
-----
The stock follows GBM with the *realized* volatility: dS/S = mu dt + sigma_real dW,
simulated exactly on a fine grid of bars (default: 5-minute bars, 78 per 6.5-hour
session, time measured in trading time with 252 days a year and no overnight gap).
At t = 0 the hedger sells one call at the implied vol sigma_imp, receives
C(S_0; sigma_imp), and holds Delta = dC/dS (BSM at sigma_imp) shares, rebalanced every
`h` bars. Cash earns r, shares earn the dividend yield q. At expiry the call is cash
settled and the shares are sold. A proportional cost kappa is charged on the dollar
value of every share trade, including the initial hedge and the final unwind, so
cost = kappa * turnover with turnover = sum |d shares| * S.

PnL (discounted to t = 0, before costs):
    PnL = e^{-rT} [cash_T + Delta_T S_T - (S_T - K)^+]

Attribution (Wilmott 2006, ch. 10; Ahmad & Wilmott 2005). Over one rebalance
interval of length dt_i starting at (t_i, S_i), a Taylor expansion of the hedged
book plus the BSM PDE at sigma_imp give
    dPnL_i = 1/2 Gamma_i S_i^2 (sigma_imp^2 dt_i - R_i^2) + O(dt^{3/2}),  R_i = S_{i+1}/S_i - 1,
where Gamma_i is the BSM gamma at sigma_imp and R_i^2 is the realized variance
sigma_real,i^2 dt_i over the interval. So
    PnL ~= sum_i e^{-r t_{i+1}} 1/2 Gamma_i S_i^2 (sigma_imp^2 - sigma_real,i^2) dt_i.
The short option earns theta and pays gamma: it makes money when the stock moves
less than the implied vol priced in (sigma_real < sigma_imp). With mu = r - q, the
expected PnL in the continuous limit is exactly C(sigma_imp) - C(sigma_real), which
to first order is vega * (sigma_imp - sigma_real).

Discrete hedging error (Boyle & Emanuel 1980; Derman & Kamal 1999): with N
rebalances the std of the replication error falls like 1/sqrt(N), while expected
turnover grows like sqrt(N) (Leland 1985). Minimising E[cost] + k std(PnL) therefore
gives an interior optimum N* ~ k / kappa.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Sequence

import numpy as np
from scipy.special import ndtr

from markout.options import bs

SESSION_MINUTES = 390  # 6.5-hour US equity session


@dataclass(frozen=True)
class HedgeConfig:
    S0: float = 100.0
    K: float = 100.0
    days: int = 30                 # trading days to expiry
    bars_per_day: int = 78         # simulation grid: 78 five-minute bars per session
    days_per_year: int = 252
    sigma_imp: float = 0.20
    sigma_real: float = 0.20
    r: float = 0.0
    q: float = 0.0
    mu: float | None = None        # real-world drift; None means r - q
    n_paths: int = 20_000
    seed: int = 0

    @property
    def n_steps(self) -> int:
        return self.days * self.bars_per_day

    @property
    def dt(self) -> float:
        return 1.0 / (self.days_per_year * self.bars_per_day)

    @property
    def T(self) -> float:
        return self.days / self.days_per_year

    @property
    def drift(self) -> float:
        return self.r - self.q if self.mu is None else self.mu

    def with_(self, **kw) -> "HedgeConfig":
        return replace(self, **kw)


@dataclass
class HedgeResult:
    """Per-path outcomes for one rebalancing interval (in bars)."""
    interval: int
    n_rebalances: int                 # hedge intervals over the option's life
    pnl: np.ndarray                   # PV PnL before costs
    turnover: np.ndarray              # PV of dollars traded; cost = kappa * turnover
    attribution: np.ndarray           # PV gamma-theta attribution
    rebalance_turnover: np.ndarray    # turnover excluding the initial hedge and the final unwind
    config: HedgeConfig = field(repr=False, default=None)

    def net(self, kappa: float) -> np.ndarray:
        return self.pnl - kappa * self.turnover


def interval_label(interval: int, bars_per_day: int) -> str:
    """Human label for a rebalance interval given in bars."""
    minutes = interval * SESSION_MINUTES / bars_per_day
    if interval % bars_per_day == 0:
        d = interval // bars_per_day
        return "1 day" if d == 1 else ("1 week" if d == 5 else f"{d} days")
    if minutes >= 60 and minutes % 60 == 0:
        return f"{int(minutes // 60)} hour" + ("s" if minutes > 60 else "")
    return f"{minutes:g} min"


def _delta_gamma(S: np.ndarray, cfg: HedgeConfig, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """Call delta and gamma at sigma_imp from a single d1 evaluation."""
    v = cfg.sigma_imp * math.sqrt(tau)
    d1 = (np.log(S / cfg.K) + (cfg.r - cfg.q + 0.5 * cfg.sigma_imp**2) * tau) / v
    dq = math.exp(-cfg.q * tau)
    return dq * ndtr(d1), dq * np.exp(-0.5 * d1 * d1) / (math.sqrt(2 * math.pi) * S * v)


def simulate(cfg: HedgeConfig, intervals: Sequence[int]) -> dict[int, HedgeResult]:
    """Hedge the same simulated paths at every interval in `intervals` (bars).

    One pass over the fine grid updates every hedger, so frequencies are compared
    on identical paths (common random numbers). Memory is O(n_paths * len(intervals)).
    """
    intervals = sorted(set(int(h) for h in intervals))
    for h in intervals:
        if cfg.n_steps % h:
            raise ValueError(f"interval {h} does not divide {cfg.n_steps} steps")
    rng = np.random.default_rng(cfg.seed)
    n, dt, T = cfg.n_paths, cfg.dt, cfg.T
    premium = float(bs.call_price(cfg.S0, cfg.K, T, cfg.r, cfg.q, cfg.sigma_imp))
    d0, g0 = (float(x) for x in _delta_gamma(np.array(cfg.S0), cfg, T))

    st = {h: {"shares": np.full(n, d0), "cash": np.full(n, premium - d0 * cfg.S0),
              "turnover": np.full(n, d0 * cfg.S0), "attr": np.zeros(n),
              "S_last": np.full(n, cfg.S0), "G_last": np.full(n, g0)} for h in intervals}
    drift = (cfg.drift - 0.5 * cfg.sigma_real**2) * dt
    vol = cfg.sigma_real * math.sqrt(dt)
    grow, div = math.exp(cfg.r * dt), math.expm1(cfg.q * dt)
    S = np.full(n, cfg.S0)
    for j in range(cfg.n_steps):
        S_prev = S
        S = S_prev * np.exp(drift + vol * rng.standard_normal(n))
        t = (j + 1) * dt
        disc = math.exp(-cfg.r * t)
        last = j + 1 == cfg.n_steps
        rebal = [h for h in intervals if (j + 1) % h == 0]
        dg = None if last or not rebal else _delta_gamma(S, cfg, T - t)
        for h in intervals:
            s = st[h]
            if cfg.r or cfg.q:
                s["cash"] = s["cash"] * grow + s["shares"] * S_prev * div
            if h not in rebal:
                continue
            R = S / s["S_last"] - 1.0
            s["attr"] += disc * 0.5 * s["G_last"] * s["S_last"] ** 2 * (cfg.sigma_imp**2 * h * dt - R * R)
            if last:
                continue
            new, G = dg
            trade = new - s["shares"]
            s["cash"] -= trade * S
            s["turnover"] += disc * np.abs(trade) * S
            s["shares"], s["S_last"], s["G_last"] = new, S, G

    payoff = np.maximum(S - cfg.K, 0.0)
    disc_T = math.exp(-cfg.r * T)
    out = {}
    for h in intervals:
        s = st[h]
        pnl = disc_T * (s["cash"] + s["shares"] * S - payoff)
        turnover = s["turnover"] + disc_T * np.abs(s["shares"]) * S
        out[h] = HedgeResult(h, cfg.n_steps // h, pnl, turnover, s["attr"], s["turnover"] - d0 * cfg.S0, cfg)
    return out


# ------------------------------------------------------------------ theory & summaries

def expected_pnl(cfg: HedgeConfig) -> dict:
    """Continuous-limit expectations of the short-call PnL hedged at sigma_imp."""
    args = (cfg.S0, cfg.K, cfg.T, cfg.r, cfg.q)
    c_imp = float(bs.call_price(*args, cfg.sigma_imp))
    c_real = float(bs.call_price(*args, cfg.sigma_real))
    vega = float(bs.vega(*args, cfg.sigma_imp))
    return {"premium": c_imp, "exact": c_imp - c_real, "first_order": vega * (cfg.sigma_imp - cfg.sigma_real),
            "vega": vega}


def summarize(res: HedgeResult) -> dict:
    """Mean, SE and std of PnL; mean turnover; and the attribution fit."""
    n = len(res.pnl)
    resid = res.pnl - res.attribution
    corr = float(np.corrcoef(res.pnl, res.attribution)[0, 1])
    sd = float(res.pnl.std(ddof=1))
    return {"interval": res.interval, "n_rebalances": res.n_rebalances,
            "label": interval_label(res.interval, res.config.bars_per_day),
            "mean": float(res.pnl.mean()), "se": sd / math.sqrt(n), "std": sd,
            "p05": float(np.quantile(res.pnl, 0.05)), "p95": float(np.quantile(res.pnl, 0.95)),
            "turnover_mean": float(res.turnover.mean()),
            "rebalance_turnover_mean": float(res.rebalance_turnover.mean()),
            "attr_corr": corr, "attr_r2": 1.0 - float(resid.var()) / float(res.pnl.var()),
            "resid_mean": float(resid.mean()), "resid_std": float(resid.std(ddof=1)),
            "resid_std_ratio": float(resid.std(ddof=1)) / sd}


def loglog_slope(x: Sequence[float], y: Sequence[float]) -> float:
    """Least-squares slope of log y on log x."""
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def frontier(results: dict[int, HedgeResult], kappas: Sequence[float],
             risk_weights: Sequence[float]) -> tuple[list[dict], list[dict]]:
    """Cost-vs-risk frontier and the frequency minimising E[cost] + k * std(net PnL).

    Returns (points, choices): one point per (kappa, interval) and, for each
    (kappa, k), the interval with the smallest objective.
    """
    points = []
    for kappa in kappas:
        for h, res in sorted(results.items(), reverse=True):
            net = res.net(kappa)
            points.append({"kappa": kappa, "interval": h, "n_rebalances": res.n_rebalances,
                           "label": interval_label(h, res.config.bars_per_day),
                           "cost_mean": kappa * float(res.turnover.mean()), "std_net": float(net.std(ddof=1)),
                           "mean_net": float(net.mean())})
    choices = []
    for kappa in kappas:
        pts = [p for p in points if p["kappa"] == kappa]
        for k in risk_weights:
            obj = [p["cost_mean"] + k * p["std_net"] for p in pts]
            best = pts[int(np.argmin(obj))]
            choices.append({"kappa": kappa, "k": k, "interval": best["interval"], "label": best["label"],
                            "n_rebalances": best["n_rebalances"], "objective": float(min(obj)),
                            "cost_mean": best["cost_mean"], "std_net": best["std_net"]})
    return points, choices
