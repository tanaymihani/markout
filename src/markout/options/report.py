"""Module F report:  python -m markout.options.report

Recomputes everything with fixed seeds and reads the saved SPY snapshot (it never
refetches). Writes reports/results/options.json, reports/figures/f_*.png and
reports/04_options.md. Every number in the markdown is interpolated from the
computed results.
"""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone

import numpy as np
from scipy.special import ndtr

from markout import plotting as mp
from markout import reporting as rp
from markout import results
from markout.options import bs, smile
from markout.options import hedging as hg
from markout.paths import ROOT

SEED = 20260929
HEDGE = hg.HedgeConfig(S0=100.0, K=100.0, days=30, bars_per_day=78, days_per_year=252,
                       sigma_imp=0.20, r=0.04, q=0.0, n_paths=20_000, seed=SEED)
SCENARIOS = (0.15, 0.20, 0.25)                 # realized vols vs sigma_imp = 0.20
INTERVALS = (390, 78, 12, 6, 3, 1)             # bars: week, day, hour, 30/15/5 min
SWEEP = tuple(round(0.10 + 0.025 * i, 4) for i in range(9))
SWEEP_INTERVAL = 12                            # hourly
DRIFT_MU = 0.15                                # robustness: a real-world drift far from r
DRIFT_INTERVALS = (78, 12, 1)
KAPPAS_BPS = (0, 1, 2, 5, 10)                  # proportional cost on stock trades
RISK_WEIGHTS = (0.5, 1.0, 2.0)                 # k in E[cost] + k * std
K_HEADLINE = 1.0                               # the risk weight quoted in the text and marked on the frontier
ATTR_SIGMA = 0.15
TARGET_DAYS = (7, 30, 91)
STOCK_HALF_SPREAD = 0.005                      # assumed SPY half-spread at the option close
SLOPE_ERR = 0.001                              # illustrates why parity cannot pin down r
GREEKS_EXAMPLE = dict(S=100.0, K=100.0, T=30 / 365, r=0.04, q=0.0, sigma=0.20)
GREEK_TENORS = (7, 30, 90)


# ============================================================ 1. pricing engine checks

FD_CASES = [(100.0, 100.0, 0.25, 0.03, 0.01, 0.20), (100.0, 80.0, 1.00, 0.05, 0.00, 0.30),
            (100.0, 125.0, 0.10, 0.02, 0.02, 0.45), (764.0, 700.0, 0.08, 0.041, 0.012, 0.15),
            (50.0, 55.0, 2.00, 0.00, 0.03, 0.60)]


def bs_checks(seed: int = SEED, n: int = 20_000) -> dict:
    """Parity, bounds, Greeks vs central differences, and IV round-trip, as numbers."""
    rng = np.random.default_rng(seed)
    S, K = rng.uniform(10, 500, n), rng.uniform(10, 500, n)
    T, r = rng.uniform(0.01, 3.0, n), rng.uniform(-0.01, 0.10, n)
    q, sig = rng.uniform(0.0, 0.08, n), rng.uniform(0.05, 1.5, n)
    c, p = bs.call_price(S, K, T, r, q, sig), bs.put_price(S, K, T, r, q, sig)
    parity = np.abs(c - p - (S * np.exp(-q * T) - K * np.exp(-r * T))) / np.maximum(S, K)
    lo_c, hi_c = bs.bounds(S, K, T, r, q, "call")
    lo_p, hi_p = bs.bounds(S, K, T, r, q, "put")
    tol = 1e-9 * np.maximum(S, K)
    bound_viol = int(((c < lo_c - tol) | (c > hi_c + tol) | (p < lo_p - tol) | (p > hi_p + tol)).sum())

    fd_err = {g: 0.0 for g in ("delta", "gamma", "vega", "theta", "rho")}
    for (s0, k0, t0, r0, q0, v0) in FD_CASES:
        for kind in ("call", "put"):
            V = lambda **kw: bs.price(**{**dict(S=s0, K=k0, T=t0, r=r0, q=q0, sigma=v0), **kw}, kind=kind)  # noqa: E731
            hS = 1e-4 * s0
            fd = {"delta": (V(S=s0 + hS) - V(S=s0 - hS)) / (2 * hS),
                  "gamma": (V(S=s0 + hS) - 2 * V() + V(S=s0 - hS)) / hS**2,
                  "vega": (V(sigma=v0 + 1e-5) - V(sigma=v0 - 1e-5)) / 2e-5,
                  "theta": -(V(T=t0 + 1e-6) - V(T=t0 - 1e-6)) / 2e-6,
                  "rho": (V(r=r0 + 1e-6) - V(r=r0 - 1e-6)) / 2e-6}
            g = bs.greeks(s0, k0, t0, r0, q0, v0, kind)
            for name, val in fd.items():
                scale = max(abs(float(g[name])), 1e-3 * s0 if name != "gamma" else 1e-3 / s0)
                fd_err[name] = max(fd_err[name], abs(float(g[name]) - val) / scale)

    iv_err, iv_n, nan_ok, nan_n = 0.0, 0, 0, 0
    for k0 in np.linspace(60.0, 160.0, 21):
        for t0 in (1 / 52, 0.1, 0.5, 1.0, 2.0):
            for v0 in (0.08, 0.25, 0.6, 1.2):
                for kind in ("call", "put"):
                    px = float(bs.price(100.0, k0, t0, 0.03, 0.01, v0, kind))
                    lo, hi = (float(x) for x in bs.bounds(100.0, k0, t0, 0.03, 0.01, kind))
                    if bs.vega(100.0, k0, t0, 0.03, 0.01, v0) > 1e-4:
                        iv_err = max(iv_err, abs(bs.implied_vol(px, 100.0, k0, t0, 0.03, 0.01, kind) - v0))
                        iv_n += 1
                    for bad in (lo - 0.01, hi + 0.01):
                        nan_n += 1
                        nan_ok += math.isnan(bs.implied_vol(bad, 100.0, k0, t0, 0.03, 0.01, kind))
    return {"n_random": n, "parity_max_rel_err": float(parity.max()), "bound_violations": bound_viol,
            "fd_cases": len(FD_CASES) * 2, "fd_max_rel_err": fd_err, "iv_cases": iv_n, "iv_max_abs_err": iv_err,
            "nan_cases": nan_n, "nan_returned": int(nan_ok)}


# ================================================================== 2. delta hedging

def run_hedging() -> tuple[dict, dict]:
    t0 = time.time()
    scen = {}
    for i, sr in enumerate(SCENARIOS):
        cfg = HEDGE.with_(sigma_real=sr, seed=SEED + i)
        scen[sr] = (cfg, hg.simulate(cfg, INTERVALS))

    by_freq = []
    for sr, (cfg, res) in scen.items():
        th = hg.expected_pnl(cfg)
        for h in sorted(res, reverse=True):
            s = hg.summarize(res[h])
            by_freq.append({"sigma_real": sr, **s, "exact": th["exact"], "first_order": th["first_order"],
                            "vol_points": 100 * s["mean"] / th["vega"]})

    sweep = []
    for j, sr in enumerate(SWEEP):
        cfg = HEDGE.with_(sigma_real=sr, seed=SEED + 100 + j)
        s = hg.summarize(hg.simulate(cfg, (SWEEP_INTERVAL,))[SWEEP_INTERVAL])
        th = hg.expected_pnl(cfg)
        sweep.append({"sigma_real": sr, "mean": s["mean"], "se": s["se"], "std": s["std"],
                      "exact": th["exact"], "first_order": th["first_order"]})

    base_cfg = HEDGE.with_(sigma_real=ATTR_SIGMA)
    drift_cfg = HEDGE.with_(sigma_real=ATTR_SIGMA, mu=DRIFT_MU, seed=SEED + 200)
    drift_res = hg.simulate(drift_cfg, DRIFT_INTERVALS)
    drift = []
    for h in sorted(drift_res, reverse=True):
        s = hg.summarize(drift_res[h])
        base = next(b for b in by_freq if b["sigma_real"] == ATTR_SIGMA and b["interval"] == h)
        drift.append({"label": s["label"], "mean_mu": s["mean"], "se_mu": s["se"], "mean_base": base["mean"],
                      "se_base": base["se"]})

    fair_cfg, fair = scen[HEDGE.sigma_imp]
    order = sorted(fair, reverse=True)
    n_reb = [fair[h].n_rebalances for h in order]
    stds = [float(fair[h].pnl.std(ddof=1)) for h in order]
    reb_turn = [float(fair[h].rebalance_turnover.mean()) for h in order]
    points, choices = hg.frontier(fair, [k * 1e-4 for k in KAPPAS_BPS], RISK_WEIGHTS)
    for p in points + choices:
        p["kappa_bps"] = round(p.pop("kappa") * 1e4, 6)

    attr_rows = [b for b in by_freq if b["sigma_real"] == ATTR_SIGMA]
    attr_res = scen[ATTR_SIGMA][1]
    rng = np.random.default_rng(SEED)
    pick = rng.choice(HEDGE.n_paths, size=2500, replace=False)
    figdata = {"sweep": sweep, "fair_n": n_reb, "fair_std": stds, "fair_reb_turn": reb_turn,
               "points": points, "choices": choices,
               "scatter": {h: (attr_res[h].attribution[pick], attr_res[h].pnl[pick]) for h in (78, 1)},
               "scatter_fit": {h: hg.summarize(attr_res[h]) for h in (78, 1)}}
    out = {
        "config": {"S0": HEDGE.S0, "K": HEDGE.K, "days": HEDGE.days, "bars_per_day": HEDGE.bars_per_day,
                   "bar_minutes": hg.SESSION_MINUTES / HEDGE.bars_per_day, "days_per_year": HEDGE.days_per_year,
                   "sigma_imp": HEDGE.sigma_imp, "r": HEDGE.r, "q": HEDGE.q, "mu": HEDGE.drift,
                   "n_paths": HEDGE.n_paths, "seed": SEED, "scenarios": list(SCENARIOS),
                   "intervals": list(INTERVALS), "kappas_bps": list(KAPPAS_BPS), "risk_weights": list(RISK_WEIGHTS),
                   "sweep_interval": SWEEP_INTERVAL, "drift_mu": DRIFT_MU, "attr_sigma": ATTR_SIGMA},
        "theory": {str(sr): hg.expected_pnl(cfg) for sr, (cfg, _) in scen.items()},
        "by_frequency": by_freq, "sweep": sweep, "drift": drift,
        "scaling": {"n_rebalances": n_reb, "std": stds, "rebalance_turnover": reb_turn,
                    "slope_std": hg.loglog_slope(n_reb, stds), "slope_turnover": hg.loglog_slope(n_reb, reb_turn),
                    "labels": [hg.interval_label(h, HEDGE.bars_per_day) for h in order]},
        "frontier": points, "choices": choices, "attribution": attr_rows,
        "base_T": base_cfg.T, "seconds": time.time() - t0,
    }
    return out, figdata


# ============================================================== 3. the SPY snapshot

def run_snapshot() -> tuple[dict | None, dict | None]:
    path = smile.latest_snapshot()
    if path is None:
        return {"available": False, "reason": f"no snapshot in {smile.OPTIONS_DIR.relative_to(ROOT)}"}, None
    df, meta = smile.load_snapshot(path)
    fresh, n_stale = smile.screen_stale(df, meta)
    if fresh["usable"].sum() < 50:
        return {"available": False, "reason": f"{path.name} has only {int(fresh['usable'].sum())} usable quotes"}, None
    S = meta["underlying_price"]
    qt = smile.quote_datetime(meta)
    expiries = smile.pick_expiries(df["expiry"].unique(), qt.date(), TARGET_DAYS)
    per, figdata = [], {"smiles": {}, "parity": {}}
    for e in expiries:
        cy = smile.carry_for(meta, e)
        sm = smile.smile(fresh, meta, e)
        summ = smile.smile_summary(sm, cy.T)
        fwd = smile.implied_forward(smile.pairs(fresh, e), cy, S)
        checks = {}
        for name, frame in (("fresh", fresh), ("all", df)):
            chk = smile.parity_check(smile.pairs(frame, e), S, S - STOCK_HALF_SPREAD, S + STOCK_HALF_SPREAD, cy)
            checks[name] = (chk, smile.parity_summary(chk))
        chk = checks["fresh"][0]
        am = chk[chk["american_violation"]]
        if len(am):
            edge = np.maximum(am["american_conv_edge"], am["american_rev_edge"])
            i = int(np.argmax(edge.to_numpy()))
            worst = {"strike": float(am["strike"].iloc[i]), "edge": float(edge.iloc[i]),
                     "rate_bps": 1e4 * smile.edge_as_rate(float(edge.iloc[i]), float(am["strike"].iloc[i]), cy)}
        else:
            worst = None
        per.append({"expiry": e.isoformat(), "days": (e - qt.date()).days, "T_days": cy.T * 365, "q": cy.q,
                    "pv_div": cy.pv_div, "forward": cy.forward(S), **{f"smile_{k}": v for k, v in summ.items()},
                    "implied_forward": fwd, "parity_fresh": checks["fresh"][1], "parity_all": checks["all"][1],
                    "american_worst": worst, "stale_in_expiry": int((df["usable"] & ~fresh["usable"]
                                                                     & (df["expiry"] == e)).sum())})
        figdata["smiles"][e] = (sm, cy.T)
        figdata["parity"][e] = chk
    fresh_near = fresh[(np.abs(fresh["strike"] / S - 1) <= 0.10)]
    atm = fresh[fresh["usable"] & (np.abs(fresh["strike"] / S - 1) <= 0.02) & fresh["expiry"].isin(expiries)]
    stale = df[df["usable"] & ~fresh["usable"]]
    intrinsic = np.where(stale["kind"] == "C", S - stale["strike"], stale["strike"] - S)
    out = {"available": True, "file": str(path.relative_to(ROOT)), "size_mb": path.stat().st_size / 1e6,
           "atm_median_spread": float((atm["ask"] - atm["bid"]).median()),
           "stale_far_share": float((np.abs(stale["strike"] / S - 1) > 0.10).mean()) if len(stale) else float("nan"),
           "stale_below_intrinsic": int((stale["bid"] < intrinsic - smile.TICK).sum()),
           "meta": {k: meta[k] for k in ("source", "source_key", "snapshot_utc", "quote_time_et", "underlying_price",
                                         "underlying_price_source", "underlying_close_1600", "underlying_last",
                                         "underlying_last_time_et", "rate", "usability", "cboe_crosscheck", "notes")},
           "dividends": {k: meta["dividends"][k] for k in ("last_ex_date", "last_amount", "rule")},
           "n_contracts": int(len(df)), "n_expiries": int(df["expiry"].nunique()),
           "n_usable": int(df["usable"].sum()), "n_stale_removed": n_stale,
           "fresh_near_money_share": float(fresh_near["usable"].mean()),
           "stock_half_spread": STOCK_HALF_SPREAD, "tick": smile.TICK, "expiries": per}
    return out, figdata


# =========================================================== 4. crib-sheet numbers

def crib_numbers() -> dict:
    ex = GREEKS_EXAMPLE
    g = bs.greeks(ex["S"], ex["K"], ex["T"], ex["r"], ex["q"], ex["sigma"], "call")
    gp = bs.greeks(ex["S"], ex["K"], ex["T"], ex["r"], ex["q"], ex["sigma"], "put")
    rule_price = 0.4 * ex["S"] * ex["sigma"] * math.sqrt(ex["T"])
    zero_r = bs.greeks(ex["S"], ex["K"], ex["T"], 0.0, 0.0, ex["sigma"], "call")
    return {"example": {k: (v if not isinstance(v, float) else v) for k, v in ex.items()},
            "call": {k: float(v) for k, v in g.items()}, "put": {k: float(v) for k, v in gp.items()},
            "rule_price": rule_price, "rule_vega": 0.4 * ex["S"] * math.sqrt(ex["T"]),
            "itm_forward": ex["S"] * math.exp((ex["r"] - ex["q"]) * ex["T"]) - ex["K"],
            "zero_r_theta": float(zero_r["theta"]),
            "zero_r_half_gamma_s2_sigma2": float(0.5 * zero_r["gamma"] * ex["S"] ** 2 * ex["sigma"] ** 2),
            "n_d2": float(ndtr(bs.d1_d2(ex["S"], ex["K"], ex["T"], ex["r"], ex["q"], ex["sigma"])[1]))}


# ======================================================================== figures

def every(label: str) -> str:
    """'1 hour' -> 'every hour', '5 min' -> 'every 5 min'."""
    return "every " + (label[2:] if label.startswith("1 ") else label)


def fig_vol_bet(sweep: list[dict]) -> None:
    def draw():
        fig, (a1, a2) = mp.subplots(1, 2, w=9.0, h=3.9)
        x = np.array([s["sigma_real"] for s in sweep]) * 100
        xs = np.linspace(x.min(), x.max(), 200)
        th = [hg.expected_pnl(HEDGE.with_(sigma_real=v / 100)) for v in xs]
        exact = np.array([t["exact"] for t in th])
        first = np.array([t["first_order"] for t in th])
        mean = np.array([s["mean"] for s in sweep])
        se2 = 2 * np.array([s["se"] for s in sweep])
        ex_pts = np.array([s["exact"] for s in sweep])
        freq = every(hg.interval_label(SWEEP_INTERVAL, HEDGE.bars_per_day))
        for ax in (a1, a2):
            mp.zero_line(ax)
            ax.axvline(HEDGE.sigma_imp * 100, color=mp.baseline_color(), linewidth=0.9, linestyle=":")
            ax.set_xlabel("realized volatility σ_real (%)")
        a1.plot(xs, exact, color=mp.series(0), label="exact: C(σ_imp) − C(σ_real)")
        a1.plot(xs, first, color=mp.series(1), linestyle="--", label="first order: vega × (σ_imp − σ_real)")
        a1.errorbar(x, mean, yerr=se2, fmt="o", color=mp.series(2), label=f"simulated, hedged {freq}", zorder=3)
        a1.set_ylabel("mean PnL per option ($)")
        mp.legend(a1, loc="upper right")
        mp.title(a1, "Short gamma is short realized vol",
                 f"{HEDGE.days}-day ATM call sold at σ_imp = {HEDGE.sigma_imp:.0%} and delta-hedged")
        a2.plot(xs, first - exact, color=mp.series(1), linestyle="--", label="first order − exact")
        a2.errorbar(x, mean - ex_pts, yerr=se2, fmt="o", color=mp.series(2), label="simulated − exact (± 2 SE)",
                    zorder=3)
        a2.set_ylabel("difference from exact ($)")
        mp.legend(a2, loc="lower left")
        mp.title(a2, "How close is the theory?", f"{HEDGE.n_paths:,} paths per point")
        return fig
    mp.render("f_vol_bet", draw)


def fig_scaling(fd: dict, slope_std: float, slope_turn: float) -> None:
    def draw():
        fig, (a1, a2) = mp.subplots(1, 2, w=8.6, h=3.8)
        n = np.array(fd["fair_n"], float)
        for ax, y, slope, lab, col in ((a1, fd["fair_std"], slope_std, "std of hedging error ($)", 0),
                                       (a2, fd["fair_reb_turn"], slope_turn, "rebalancing turnover ($ traded)", 1)):
            y = np.array(y)
            ax.loglog(n, y, "o-", color=mp.series(col), label="simulated")
            ref = y[0] * (n / n[0]) ** (-0.5 if ax is a1 else 0.5)
            ax.loglog(n, ref, color=mp.ink("muted"), linestyle="--", linewidth=1.0,
                      label="1/√N reference" if ax is a1 else "√N reference")
            ax.set_xlabel("rebalances over the option's life, N")
            ax.set_ylabel(lab)
            ax.grid(True, which="major", axis="y")
            mp.legend(ax, loc="upper right" if ax is a1 else "upper left")
        mp.title(a1, "Hedging error falls like 1/√N", f"fitted log-log slope {slope_std:.3f}; σ_real = σ_imp")
        mp.title(a2, "Trading grows like √N", f"fitted log-log slope {slope_turn:.3f}")
        return fig
    mp.render("f_hedge_scaling", draw)


def fig_frontier(points: list[dict], choices: list[dict]) -> None:
    def draw():
        fig, ax = mp.subplots(w=7.6, h=4.4)
        ks = [k for k in KAPPAS_BPS if k > 0]
        for i, k in enumerate(ks):
            pts = sorted((p for p in points if p["kappa_bps"] == k), key=lambda p: p["n_rebalances"])
            ax.plot([p["std_net"] for p in pts], [p["cost_mean"] for p in pts], "o-", color=mp.series(i),
                    label=f"κ = {k:g} bp")
            best = next(c for c in choices if c["kappa_bps"] == k and c["k"] == K_HEADLINE)
            ax.plot(best["std_net"], best["cost_mean"], "o", markersize=13, markerfacecolor="none",
                    markeredgecolor=mp.ink(), markeredgewidth=1.3, zorder=4)
            if k == ks[0]:   # label frequencies once, on the lowest-cost line (it is monotone)
                for p in pts:
                    ax.annotate(p["label"], xy=(p["std_net"], p["cost_mean"]), xytext=(0, -13),
                                textcoords="offset points", ha="center", va="top", fontsize=8.5,
                                color=mp.ink("secondary"))
        ax.set_xscale("log")
        ax.set_yscale("log")
        low = min(p["cost_mean"] for p in points if p["kappa_bps"] > 0)
        ax.set_ylim(bottom=low / 2.2)     # room for the frequency labels under the lowest line
        ax.set_xlabel("std of hedged PnL net of costs ($ per option)")
        ax.set_ylabel("expected transaction cost ($ per option)")
        ax.plot([], [], "o", markersize=11, markerfacecolor="none", markeredgecolor=mp.ink(),
                label=f"minimum of cost + {K_HEADLINE:g} × std")
        mp.legend(ax, loc="upper left")
        mp.title(ax, "Rebalancing frequency is a cost–risk decision",
                 "each line runs from weekly (right) to every 5 minutes (left)")
        return fig
    mp.render("f_hedge_frontier", draw)


def fig_attribution(fd: dict) -> None:
    def draw():
        fig, axes = mp.subplots(1, 2, w=8.6, h=4.0)
        for ax, h in zip(axes, (78, 1)):
            x, y = fd["scatter"][h]
            fit = fd["scatter_fit"][h]
            lim = [min(x.min(), y.min()), max(x.max(), y.max())]
            ax.plot(lim, lim, color=mp.ink("muted"), linewidth=1.0, linestyle="--", label="PnL = attribution")
            ax.scatter(x, y, s=5, color=mp.series(0), alpha=0.35, linewidths=0, label="simulated path")
            ax.set_xlabel("attribution Σ ½ Γ S² (σ_imp² − σ_real,t²) Δt ($)")
            ax.set_ylabel("simulated hedged PnL ($)")
            mp.legend(ax, loc="upper left")
            mp.title(ax, f"Hedged {every(fit['label'])}",
                     f"corr {fit['attr_corr']:.4f}, residual std / PnL std {fit['resid_std_ratio']:.3f}")
        return fig
    mp.render("f_attribution", draw)


def fig_smile(fd: dict) -> None:
    def draw():
        fig, (a1, a2) = mp.subplots(1, 2, w=8.8, h=4.0)
        for i, (e, (sm, T)) in enumerate(fd["smiles"].items()):
            s = sm.dropna(subset=["iv_mid"]).sort_values("log_moneyness")
            lab = f"{e:%b %d} ({T * 365:.0f} days)"
            for ax, x in ((a1, s["log_moneyness"] * 100), (a2, s["z"])):
                ax.fill_between(x, s["iv_bid"] * 100, s["iv_ask"] * 100, color=mp.series(i), alpha=0.25, linewidth=0)
                ax.plot(x, s["iv_mid"] * 100, color=mp.series(i), label=lab)
        a1.set_xlabel("log-moneyness ln(K/F) (%)")
        a2.set_xlabel("standardized moneyness ln(K/F) / (σ_ATM √T)")
        for ax in (a1, a2):
            ax.axvline(0, color=mp.baseline_color(), linewidth=0.9, linestyle=":")
            ax.set_ylabel("implied vol (%)")
        mp.legend(a1, loc="upper right")
        mp.title(a1, "SPY implied-vol smile", "OTM puts left of the forward, OTM calls right; band = bid–ask IV")
        mp.title(a2, "Same smiles in ATM standard deviations", "x = ln(K/F) / (σ_ATM √T)")
        return fig
    mp.render("f_smile", draw)


def fig_parity(fd: dict) -> None:
    def draw():
        n = len(fd["parity"])
        fig, axes = mp.subplots(1, n, w=3.2 * n + 0.6, h=3.9)
        axes = np.atleast_1d(axes)
        for ax, (e, chk) in zip(axes, fd["parity"].items()):
            c = chk.sort_values("moneyness")
            ax.fill_between(c["moneyness"], -c["half_band"], c["half_band"], step="mid", color=mp.series(0),
                            alpha=0.18, linewidth=0, label="± half the bid–ask cost")
            ok = ~c["spread_violation"]
            ax.plot(c.loc[ok, "moneyness"], c.loc[ok, "residual_mid"], "o", markersize=3.5, markeredgewidth=0.6,
                    color=mp.series(0), label="inside the band")
            eu = c["spread_violation"] & ~c["american_violation"]
            ax.plot(c.loc[eu, "moneyness"], c.loc[eu, "residual_mid"], "o", markersize=4.5, markeredgewidth=0.6,
                    color=mp.series(1), label="outside: European bound only")
            am = c["american_violation"]
            ax.plot(c.loc[am, "moneyness"], c.loc[am, "residual_mid"], "D", markersize=4.5, markeredgewidth=0.6,
                    color=mp.series(3), label="outside the American band")
            mp.zero_line(ax)
            ax.set_xlabel("strike / spot")
            mp.title(ax, f"{e:%b %d}", f"{len(c)} strike pairs")
        axes[0].set_ylabel("parity residual at mids ($)")
        axes[-1].legend(loc="lower left", fontsize=8)
        return fig
    mp.render("f_parity", draw)


def fig_greeks() -> None:
    ex = GREEKS_EXAMPLE
    S = np.linspace(75, 125, 301)

    def draw():
        fig, axes = mp.subplots(2, 2, w=8.4, h=5.6)
        panels = (("delta", "call delta"), ("gamma", "gamma (per $)"), ("vega", "vega ($ per vol point)"),
                  ("theta", "theta ($ per calendar day)"))
        for ax, (g, ylab) in zip(axes.ravel(), panels):
            for i, days in enumerate(GREEK_TENORS):
                T = days / 365
                if g == "delta":
                    y = bs.delta(S, ex["K"], T, ex["r"], ex["q"], ex["sigma"], "call")
                elif g == "gamma":
                    y = bs.gamma(S, ex["K"], T, ex["r"], ex["q"], ex["sigma"])
                elif g == "vega":
                    y = bs.vega(S, ex["K"], T, ex["r"], ex["q"], ex["sigma"]) / 100
                else:
                    y = bs.theta(S, ex["K"], T, ex["r"], ex["q"], ex["sigma"], "call") / 365
                ax.plot(S, y, color=mp.series(i), label=f"{days} days")
            ax.axvline(ex["K"], color=mp.baseline_color(), linewidth=0.9, linestyle=":")
            ax.set_ylabel(ylab)
            ax.set_title(g.capitalize(), loc="left")
        for ax in axes[1]:
            ax.set_xlabel(f"spot (strike {ex['K']:g}, σ = {ex['sigma']:.0%})")
        mp.legend(axes[0, 1], loc="upper right")
        return fig
    mp.render("f_greeks", draw)


# ======================================================================= markdown

def _pct(x: float, d: int = 1) -> str:
    return "n/a" if x is None or not math.isfinite(x) else f"{100 * x:.{d}f}%"


def _money(x: float, d: int = 2, signed: bool = False) -> str:
    if x is None or not math.isfinite(x):
        return "n/a"
    s = f"{abs(x):,.{d}f}"
    sign = "−" if x < 0 else ("+" if signed and x > 0 else "")
    return f"{sign}${s}"


def section_engine(c: dict) -> str:
    fd = c["fd_max_rel_err"]
    worst_greek = max(fd, key=fd.get)
    return f"""## 1. Pricing engine (`markout.options.bs`)

Black–Scholes–Merton with a continuous dividend yield q: prices, the five Greeks and
implied vol by Brent's method on a bracket (returns NaN when no volatility fits, for
example a price below intrinsic value). The formulas are in the module docstring.
The same checks the tests assert, as numbers:

{rp.table([
    {"check": f"put–call parity, {c['n_random']:,} random parameter sets", "result": f"max error {c['parity_max_rel_err']:.1e} × max(S, K)"},
    {"check": "no-arbitrage bounds on the same sets", "result": f"{c['bound_violations']} violations"},
    {"check": f"Greeks vs central finite differences ({c['fd_cases']} cases)",
     "result": f"max relative error {fd[worst_greek]:.1e} ({worst_greek})"},
    {"check": f"IV round-trip ({c['iv_cases']} strike × maturity × vol cases)", "result": f"max |σ̂ − σ| = {c['iv_max_abs_err']:.1e}"},
    {"check": "IV of prices outside the no-arbitrage bounds", "result": f"NaN in {c['nan_returned']} of {c['nan_cases']} cases"},
], columns=["check", "result"])}
"""


def section_hedging(H: dict) -> str:
    cfg = H["config"]
    rows = H["by_frequency"]
    fair = {r["label"]: r for r in rows if r["sigma_real"] == cfg["sigma_imp"]}
    low = {r["label"]: r for r in rows if r["sigma_real"] == min(cfg["scenarios"])}
    high = {r["label"]: r for r in rows if r["sigma_real"] == max(cfg["scenarios"])}
    th_low = H["theory"][str(min(cfg["scenarios"]))]
    sc = H["scaling"]
    labels = sc["labels"]
    fine = labels[-1]
    daily = hg.interval_label(HEDGE.bars_per_day, HEDGE.bars_per_day)
    std_ratio = fair[daily]["std"] / fair[fine]["std"]
    n_ratio = fair[fine]["n_rebalances"] / fair[daily]["n_rebalances"]
    turn_ratio = fair[fine]["rebalance_turnover_mean"] / fair[daily]["rebalance_turnover_mean"]
    sign_ok = low[fine]["mean"] > 0 > high[fine]["mean"]
    max_z = max(abs(r["mean"] - r["exact"]) / r["se"] for r in rows)
    floor_ratio = low[fine]["std"] / fair[fine]["std"]

    rows_a = [{"σ_real": _pct(r["sigma_real"], 0), "hedged": every(r["label"]), "N": r["n_rebalances"],
               "mean PnL": f"{r['mean']:+.4f} ± {r['se']:.4f}", "exact": f"{r['exact']:+.4f}",
               "vega × Δσ": f"{r['first_order']:+.4f}", "std": f"{r['std']:.4f}",
               "5% … 95%": f"{r['p05']:+.3f} … {r['p95']:+.3f}"} for r in rows]
    rows_sweep = [{"σ_real": _pct(s["sigma_real"], 1), "simulated mean": f"{s['mean']:+.4f} ± {s['se']:.4f}",
                   "exact": f"{s['exact']:+.4f}", "vega × Δσ": f"{s['first_order']:+.4f}",
                   "(simulated − exact) / SE": f"{(s['mean'] - s['exact']) / s['se']:+.1f}"} for s in H["sweep"]]
    base_mu = _pct(cfg["r"] - cfg["q"], 0)
    rows_drift = [{"hedged": every(d["label"]), f"mean, μ = {base_mu}": f"{d['mean_base']:+.4f} ± {d['se_base']:.4f}",
                   f"mean, μ = {_pct(cfg['drift_mu'], 0)}": f"{d['mean_mu']:+.4f} ± {d['se_mu']:.4f}",
                   "change": _pct(d["mean_mu"] / d["mean_base"] - 1)} for d in H["drift"]]
    drift_rel = max(abs(d["mean_mu"] / d["mean_base"] - 1) for d in H["drift"])
    drift_abs = max(abs(d["mean_mu"] - d["mean_base"]) for d in H["drift"])
    drift_word = "barely matters" if drift_rel < 0.05 else "matters"
    rows_scale = [{"hedged": every(lab), "N": n, "std of hedging error": f"{s:.4f}",
                   "std × √N": f"{s * math.sqrt(n):.3f}", "rebalancing turnover ($)": f"{t:,.1f}"}
                  for lab, n, s, t in zip(labels, sc["n_rebalances"], sc["std"], sc["rebalance_turnover"])]

    ch = H["choices"]
    grid = []
    for k in cfg["risk_weights"]:
        row = {"risk weight k": f"{k:g}"}
        for kap in cfg["kappas_bps"]:
            row[f"κ = {kap:g} bp"] = next(c for c in ch if c["k"] == k and c["kappa_bps"] == kap)["label"]
        grid.append(row)
    k1 = [next(c for c in ch if c["k"] == K_HEADLINE and c["kappa_bps"] == kap) for kap in cfg["kappas_bps"]]
    n_star = [c["n_rebalances"] for c in k1]
    monotone = all(a >= b for a, b in zip(n_star, n_star[1:]))
    lo_k, hi_k = min(k for k in cfg["kappas_bps"] if k > 0), max(cfg["kappas_bps"])
    front_rows = [{"κ (bp)": f"{p['kappa_bps']:g}", "hedged": every(p["label"]), "E[cost] ($)": f"{p['cost_mean']:.4f}",
                   "std net ($)": f"{p['std_net']:.4f}", f"E[cost] + {K_HEADLINE:g} × std": f"{p['cost_mean'] + K_HEADLINE * p['std_net']:.4f}"}
                  for p in H["frontier"] if p["kappa_bps"] in (lo_k, hi_k)]
    hi_pts = [p for p in H["frontier"] if p["kappa_bps"] == hi_k]
    hi_min = min(hi_pts, key=lambda p: p["std_net"])
    hi_fine = next(p for p in hi_pts if p["label"] == fine)
    random_cost = ""
    if hi_fine["std_net"] > 1.2 * hi_min["std_net"]:
        random_cost = (f" At κ = {hi_k:g} bp the std of net PnL rises again at high frequency, from "
                       f"{_money(hi_min['std_net'], 3)} hedging {every(hi_min['label'])} to {_money(hi_fine['std_net'], 3)} "
                       f"hedging {every(fine)}, because the cost itself is random: paths that pin the strike near "
                       f"expiry, where gamma is largest, trade the most.")

    attr = H["attribution"]
    rows_attr = [{"hedged": every(a["label"]), "N": a["n_rebalances"], "corr(PnL, attribution)": f"{a['attr_corr']:.4f}",
                  "R²": f"{a['attr_r2']:.4f}", "mean residual": f"{a['resid_mean']:+.4f}",
                  "residual std / PnL std": f"{a['resid_std_ratio']:.3f}"} for a in attr]
    a_fine, a_daily = attr[-1], next(a for a in attr if a["label"] == daily)
    shrinking = all(a["resid_std_ratio"] >= b["resid_std_ratio"] for a, b in zip(attr, attr[1:]))
    return f"""## 2. Delta-hedging a short call (`markout.options.hedging`)

**Setup.** Sell one {cfg['days']}-trading-day at-the-money call (S₀ = K = {cfg['S0']:g}) at
σ_imp = {_pct(cfg['sigma_imp'], 0)}, and delta-hedge it at σ_imp while the stock follows GBM with
realized vol σ_real. Paths are simulated exactly on {cfg['bar_minutes']:g}-minute bars ({cfg['bars_per_day']}
per {hg.SESSION_MINUTES / 60:g}-hour session, {cfg['days_per_year']} days a year, trading time with no overnight gap), with
r = {_pct(cfg['r'], 0)}, q = {_pct(cfg['q'], 0)}, drift μ = r − q, and {cfg['n_paths']:,} paths per
scenario (seed {cfg['seed']}). Every frequency hedges the *same* paths. Costs are proportional (κ
per dollar of stock traded, including the initial hedge and the final unwind), and PnL is
discounted to t = 0 and quoted per option in dollars. The call is worth {_money(th_low['premium'], 3)}
and has vega {_money(th_low['vega'] / 100, 4)} per vol point.

### 2a. The mean PnL is the vega-scaled vol difference

{mp.picture('f_vol_bet', 'Mean hedged PnL against realized vol, with the exact and first-order theory')}

A short option hedged at σ_imp collects theta and pays ½ΓΔS² on every move, so it is short
realized variance. With μ = r − q its expected PnL in the continuous limit is exactly
C(σ_imp) − C(σ_real), and to first order vega × (σ_imp − σ_real).
{"The sign comes out as theory says" if sign_ok else "The sign does NOT come out as theory says"}. At σ_real =
{_pct(min(cfg['scenarios']), 0)} the short call makes {_money(low[fine]['mean'], 4, True)} hedged {every(fine)}
({low[fine]['vol_points']:.2f} vol points × vega), against an exact {_money(th_low['exact'], 4, True)}. At
σ_real = {_pct(max(cfg['scenarios']), 0)} it loses {_money(-high[fine]['mean'], 4)}.

The sweep behind the figure (hedged {every(hg.interval_label(cfg['sweep_interval'], cfg['bars_per_day']))}):

{rp.table(rows_sweep)}

Every scenario and frequency:

{rp.table(rows_a)}

Discrete hedging adds noise, not bias: every mean in the table is within {max_z:.1f} standard errors
of C(σ_imp) − C(σ_real). But when σ_real ≠ σ_imp the std does not go to zero as hedging gets finer.
At σ_real = {_pct(min(cfg['scenarios']), 0)} it levels off at {_money(low[fine]['std'], 3)}, {floor_ratio:.1f}× the
σ_real = σ_imp value at the same frequency. Hedging at the implied vol leaves the PnL path-dependent:
the vol difference is earned in proportion to ½ΓS², so it depends on how much time the path spent
near the strike.

The drift {drift_word}. Raising μ from {base_mu} to {_pct(cfg['drift_mu'], 0)} changes the mean by at most
{_money(drift_abs, 4)} ({_pct(drift_rel)}). The PnL depends on realized variance, not on direction, and the small
drop comes from paths drifting away from the strike, where gamma, and so the vol edge earned per day,
is largest:

{rp.table(rows_drift)}

### 2b. Hedging error vs cost: a decision problem

{mp.picture('f_hedge_scaling', 'Hedging-error std and turnover against the number of rebalances, log-log')}

With σ_real = σ_imp the hedged PnL is pure replication error (mean ≈ 0). Its std falls with the
number of rebalances N with a fitted log-log slope of {sc['slope_std']:.3f}, against −1/2 for the 1/√N law.
Going from {every(daily)} to {every(fine)} ({n_ratio:.0f}× more rebalances, √{n_ratio:.0f} = {math.sqrt(n_ratio):.1f})
cuts the std {std_ratio:.1f}×. Rebalancing turnover grows with a slope of {sc['slope_turnover']:.3f} (√N predicts
+1/2): {turn_ratio:.1f}× over the same range. With a proportional cost κ per dollar of stock traded, the
expected cost is κ × turnover, so more frequent hedging buys lower risk at a rising price.

{rp.table(rows_scale)}

{mp.picture('f_hedge_frontier', 'Expected cost against PnL std for each cost level and frequency')}

The decision: choose the frequency that minimises E[cost] + k × std(net PnL), where k prices a
dollar of PnL std. Minimising a√N + k·b/√N gives N* = k·b/a, so the optimal number of rebalances
should rise with k and fall with κ. {"The simulation agrees" if monotone else "The simulation does not fully agree"}:
at k = {K_HEADLINE:g} the best frequency goes from {every(k1[0]['label'])} at κ = {k1[0]['kappa_bps']:g} bp to
{every(k1[-1]['label'])} at κ = {k1[-1]['kappa_bps']:g} bp.{random_cost}

Best frequency by cost level (columns) and risk weight (rows):

{rp.table(grid)}

The frontier's numbers at the lowest and highest nonzero cost:

{rp.table(front_rows)}

### 2c. Pathwise attribution

{mp.picture('f_attribution', 'Simulated hedged PnL against the gamma-theta attribution, daily and 5-minute hedging')}

Over each rebalance interval the hedged book earns ½ΓS²(σ_imp² Δt − R²). Γ is the BSM gamma at
σ_imp at the start of the interval, and R² is the squared stock return over it (the realized
σ²_real,t Δt). Summed and discounted, this should reproduce the simulated PnL path by path. At
σ_real = {_pct(cfg['attr_sigma'], 0)} it explains {_pct(a_fine['attr_r2'], 2)} of the PnL variance hedging
{every(a_fine['label'])} (correlation {a_fine['attr_corr']:.4f}) and {_pct(a_daily['attr_r2'], 1)} hedging
{every(daily)}. The residual comes from third-order Taylor terms{", and it shrinks as the interval does" if shrinking else ""}:

{rp.table(rows_attr)}

Real prices have far larger daily moves than these lognormal paths, so the same daily hedge is attributed less
cleanly on them. [Report 06](06_vol_premium.md#collecting-the-premium-with-options-instead) runs it on every month
of S&P 500 data since 1993, as a short straddle against a variance swap.
"""


def section_snapshot(P: dict) -> str:
    if not P.get("available"):
        return f"""## 3. SPY option chain

No usable SPY snapshot: {P['reason']}. Run `python scripts/snapshot_options.py` (ideally during market
hours) and rerun this report. This section is skipped rather than filled with made-up numbers.
"""
    m = P["meta"]
    rate = m["rate"]
    cc = m["cboe_crosscheck"]
    ex = P["expiries"]
    qt = datetime.fromisoformat(m["quote_time_et"])
    snap = datetime.fromisoformat(m["snapshot_utc"]).astimezone(smile.ET)
    gap = m["underlying_price"] - m["underlying_close_1600"]
    gap_ratio = abs(gap) / P["atm_median_spread"]

    rows_smile = [{"expiry": e["expiry"], "days": e["days"], "strikes": e["smile_n_strikes"],
                   "ATM IV": _pct(e["smile_atm_iv"]), "25Δ put": _pct(e["smile_iv_25d_put"]),
                   "25Δ call": _pct(e["smile_iv_25d_call"]), "25Δ RR (vol pts)": f"{100 * e['smile_rr25']:+.2f}",
                   "slope per 10% (vol pts)": f"{100 * e['smile_skew_per_10pct']:+.2f}",
                   "slope per σ (vol pts)": f"{100 * e['smile_skew_per_sigma']:+.2f}",
                   "IV width ATM / wings (vol pts)": f"{100 * e['smile_iv_spread_atm']:.2f} / {100 * e['smile_iv_spread_wings']:.2f}"}
                  for e in ex]
    short, long_ = ex[0], ex[-1]
    per10_steeper_short = abs(short["smile_skew_per_10pct"]) > abs(long_["smile_skew_per_10pct"])
    persig_steeper_long = abs(long_["smile_skew_per_sigma"]) > abs(short["smile_skew_per_sigma"])
    if per10_steeper_short and persig_steeper_long:
        skew_txt = (f"Per 10% of moneyness the {short['days']}-day smile is the steepest "
                    f"({100 * short['smile_skew_per_10pct']:+.2f} vol points vs {100 * long_['smile_skew_per_10pct']:+.2f} "
                    f"at {long_['days']} days). Measured per ATM standard deviation the order reverses "
                    f"({100 * short['smile_skew_per_sigma']:+.2f} vs {100 * long_['smile_skew_per_sigma']:+.2f}): a short-dated "
                    f"smile looks steep only because its strikes span few standard deviations.")
    else:
        skew_txt = (f"The near-the-money slope is {100 * short['smile_skew_per_10pct']:+.2f} vol points per 10% of moneyness "
                    f"at {short['days']} days and {100 * long_['smile_skew_per_10pct']:+.2f} at {long_['days']} days "
                    f"({100 * short['smile_skew_per_sigma']:+.2f} and {100 * long_['smile_skew_per_sigma']:+.2f} per ATM "
                    f"standard deviation).")
    wings_wider = all(e["smile_iv_spread_wings"] > e["smile_iv_spread_atm"] for e in ex)
    all_neg_rr = all(e["smile_rr25"] < 0 for e in ex)

    rows_par = []
    for e in ex:
        f, a = e["parity_fresh"], e["parity_all"]
        rows_par.append({"expiry": e["expiry"], "pairs": f["pairs"],
                         "at mids": f"{f['mid_violations']} ({f['mid_violations'] / max(f['pairs'], 1):.0%})",
                         "after the spread": f"{f['spread_violations']} ({f['spread_violations'] / max(f['pairs'], 1):.0%})",
                         "reversals, K > S": f["reversal_violations_itm_put"],
                         "conversions": f["spread_violations_conversion"],
                         "outside American band": f["american_violations"],
                         "American, stale kept": f"{a['american_violations']} of {a['pairs']}"})
    tot = {k: sum(e["parity_fresh"][k] for e in ex)
           for k in ("pairs", "mid_violations", "spread_violations", "american_violations",
                     "spread_violations_reversal", "reversal_violations_itm_put", "spread_violations_conversion")}
    conv_div = sum(e["parity_fresh"]["spread_violations_conversion"] for e in ex if e["pv_div"] > 0)
    conv_nodiv = tot["spread_violations_conversion"] - conv_div
    am_nodiv = sum(e["parity_fresh"]["american_violations"] for e in ex if e["pv_div"] == 0)
    div_names = ", ".join(e["expiry"] for e in ex if e["pv_div"] > 0) or "none"
    tot_all_am = sum(e["parity_all"]["american_violations"] for e in ex)
    max_all = max(e["parity_all"]["max_american_edge"] for e in ex)
    max_fresh = max(e["parity_fresh"]["max_american_edge"] for e in ex)
    worst = [e["american_worst"] | {"expiry": e["expiry"]} for e in ex if e["american_worst"]]
    worst_txt = ""
    if worst:
        w = max(worst, key=lambda x: x["edge"])
        worst_txt = (f" The largest is {_money(w['edge'], 3)} per share at the {w['strike']:g} strike ({w['expiry']}). "
                     f"A conversion is a synthetic loan, and this one lends at {w['rate_bps']:.0f} bp a year above the "
                     f"T-bill rate used for r, before commissions and exchange fees. That is a financing-rate "
                     f"difference, not a free lunch.")
    conv_txt = f"{tot['spread_violations_conversion']} are conversions (buy stock, sell the call, buy the put)"
    div_conv = [e for e in ex if e["pv_div"] > 0 and e["parity_fresh"]["spread_violations_conversion"]]
    if div_conv:
        e = div_conv[0]
        lo, hi = e["parity_fresh"]["conversion_moneyness_range"]
        conv_txt += (f". {e['parity_fresh']['spread_violations_conversion']} of them are in {e['expiry']}, the expiry with an "
                     f"ex-date before it, at strikes from {lo:.0%} to {hi:.0%} of spot. They need the dividend to arrive, "
                     f"and the holder of an American call can exercise before the ex-date and keep it, so they are not "
                     f"riskless. The model forward also sits {_money(abs(e['implied_forward']['diff']), 2)} "
                     f"{'below' if e['implied_forward']['diff'] > 0 else 'above'} the market's (see the caveats), which on "
                     f"its own shifts every residual in that expiry by "
                     f"{_money(e['implied_forward']['diff'] * math.exp(-rate['r_cc'] * e['T_days'] / 365), 2, True)}")
        if conv_nodiv:
            conv_txt += f". The remaining {conv_nodiv} are in expiries without a dividend"
    nodiv_note = (" With no dividend before expiry the American conversion bound is the same as the European "
                  "one, so the conversions there are the ones left." if am_nodiv and am_nodiv == tot["american_violations"]
                  else "")

    fw = [e["implied_forward"] for e in ex]
    rate_spread = [1e4 * math.log(f["forward_implied"] / f["forward_model"]) / (e["T_days"] / 365) for f, e in zip(fw, ex)]
    rows_fwd = [{"expiry": e["expiry"], "PV of dividends": _money(e["pv_div"], 3), "q (escrowed)": _pct(e["q"], 2),
                 "model forward": _money(f["forward_model"], 2), "parity-implied forward": _money(f["forward_implied"], 2),
                 "difference": f"{_money(f['diff'], 2, True)} ({f['diff_bps']:+.1f} bp)",
                 "as a rate spread": f"{rs:+.0f} bp/yr"} for e, f, rs in zip(ex, fw, rate_spread)]
    same_sign = all(f["diff"] > 0 for f in fw) or all(f["diff"] < 0 for f in fw)
    fwd_txt = (f"The parity-implied forwards sit {'above' if fw[0]['diff'] > 0 else 'below'} the model forwards at every "
               f"expiry, by at most {max(abs(f['diff_bps']) for f in fw):.1f} bp of spot. As a financing rate that is "
               f"{min(rate_spread):+.0f} to {max(rate_spread):+.0f} bp a year relative to the T-bill rate. At one week a "
               f"few cents of stock-price timing error is enough to produce it"
               if same_sign else
               f"The parity-implied forwards differ from the model forwards by at most "
               f"{max(abs(f['diff_bps']) for f in fw):.1f} bp of spot, with mixed signs")
    div_exp = [e for e in ex if e["pv_div"] > 0]
    if div_exp:
        d = div_exp[0]
        dd = -d["implied_forward"]["diff"] * math.exp(-rate["r_cc"] * d["T_days"] / 365)
        fwd_txt += (f". For {d['expiry']} it is equally consistent with a dividend "
                    f"{_money(abs(dd), 2)} {'smaller' if dd < 0 else 'larger'} than the assumed "
                    f"{_money(P['dividends']['last_amount'], 3)}")
    fwd_txt += "."

    return f"""## 3. SPY option chain (`scripts/snapshot_options.py`, `markout.options.smile`)

**Data.** One snapshot, `{P['file']}` ({P['size_mb']:.2f} MB), with {P['n_contracts']:,} contracts over
{P['n_expiries']} expiries from {m['source']}. It was fetched at {snap:%Y-%m-%d %H:%M} ET, after the close, so the
quotes are the {qt:%H:%M} ET closing quotes (SPY options trade until then).

* **Underlying.** {_money(m['underlying_price'])}, taken from the {m['underlying_price_source']}. The 16:00 close was
  {_money(m['underlying_close_1600'])}, and the post-market last at fetch time was {_money(m['underlying_last'])}.
  Using the 16:00 close would have put the stock {_money(abs(gap), 2)} away from the option quotes,
  {gap_ratio:.0f}× the median near-the-money option spread of {_money(P['atm_median_spread'], 3)}.
* **Rate.** r = {_pct(rate['r_cc'], 3)} continuously compounded, from the {rate['irx_date']} 13-week T-bill
  discount yield (^IRX) of {rate['irx_discount_pct']:.3f}% ({rate['source'].split(', converted: ')[-1]}).
  It is not backed out of parity, which is unreliable at these maturities (see the caveats).
* **Dividends.** Discrete and escrowed. SPY last paid {_money(P['dividends']['last_amount'], 3)} (ex-date
  {P['dividends']['last_ex_date']}); the rule used is {P['dividends']['rule']}. Only expiries after the next
  ex-date carry one.
* **Quote quality.** {_pct(m['usability']['two_sided_share'])} of all quotes are two-sided
  ({_pct(m['usability']['near_money_two_sided_share'])} within {smile.NEAR_MONEY:.0%} of spot at expiries of {smile.NEAR_DAYS[0]}–{smile.NEAR_DAYS[1]} days), so yfinance
  was used and the Cboe fallback was not needed. As a staleness check, Cboe's delayed-quotes JSON (fetched
  {cc.get('timestamp_utc', 'n/a')} UTC) was merged in by contract: {_pct(cc.get('identical_bid_ask_share', float('nan')))}
  of the yfinance quotes are identical to Cboe's. The {P['n_stale_removed']:,} two-sided quotes that are not are dropped
  below. {_pct(P['stale_far_share'], 0)} of them are more than 10% from spot, and {P['stale_below_intrinsic']} bid below
  intrinsic value, which a live American quote cannot do.

### 3a. The smile and the skew

{mp.picture('f_smile', 'SPY implied vol against log-moneyness and standardized moneyness for three expiries')}

Implied vols come from the mid, the bid and the ask of out-of-the-money options (puts below the
forward, calls above). That is the liquid side, and it carries little of the early-exercise premium
that would bias a European IV. The forward uses the T-bill r and the escrowed dividend. The slopes
are fitted within {smile.SKEW_Z:g} ATM standard deviation{"" if smile.SKEW_Z == 1 else "s"} of the forward.

{rp.table(rows_smile)}

{"The equity skew is there at every expiry" if all_neg_rr else "The skew is mixed across expiries"}: the 25-delta risk
reversal (call IV − put IV) runs from {100 * short['smile_rr25']:+.2f} vol points at {short['days']} days to
{100 * long_['smile_rr25']:+.2f} at {long_['days']} days. {skew_txt}
{f"The bid–ask band in IV is thin at the money and wider in the wings (median {100 * short['smile_iv_spread_atm']:.2f} vs {100 * short['smile_iv_spread_wings']:.2f} vol points at {short['days']} days)." if wings_wider else ""}

### 3b. Put–call parity against the bid and ask

{mp.picture('f_parity', 'Put-call parity residuals at mid prices with the bid-ask band, by strike')}

For each strike with a two-sided call and put, the residual (C − P)_mid − (S e^{{−qT}} − K e^{{−rT}})
counts as an apparent arbitrage when it exceeds one tick ({_money(P['tick'])}). Crossing the spread gives the
conversion bound C_bid − P_ask ≤ S_ask e^{{−qT}} − K e^{{−rT}} and the reversal bound
C_ask − P_bid ≥ S_bid e^{{−qT}} − K e^{{−rT}}, with a stock half-spread of {_money(P['stock_half_spread'], 3)}.
SPY options are American, so the trade is only riskless outside the American band
S e^{{−qT}} − K ≤ C − P ≤ S − K e^{{−rT}}.

{rp.table(rows_par)}

At the mids, parity fails for {tot['mid_violations']} of {tot['pairs']} strike pairs
({tot['mid_violations'] / tot['pairs']:.0%}). After paying the spread, {tot['spread_violations']} remain against the
European bounds. Of these, {tot['spread_violations_reversal']} are reversals (short stock, buy the call, sell the put),
{"all" if tot['reversal_violations_itm_put'] == tot['spread_violations_reversal'] else tot['reversal_violations_itm_put']} of them at strikes above spot, where the put sold is in the money. An ITM American
put is worth more than a European one, because exercising early earns interest on K, so selling it
risks early assignment. The other {conv_txt}. Against the American band, {tot['american_violations']} remain.{nodiv_note}{worst_txt}
Keeping the stale yfinance quotes would instead have produced {tot_all_am} American-band "violations",
with edges up to {_money(max_all)} per share (against {_money(max_fresh, 3)} on the screened quotes). That is fake
arbitrage from stale data.

This is the backtest's lesson again. An edge measured at the mid mostly disappears once you pay
the spread, and most of what is left is modelling error: European vs American exercise, the dividend,
the financing rate, stale quotes.

**Caveats.**
* *American exercise:* a conversion or reversal that sells an ITM put, or sells a call before an
  ex-date, can be assigned early, so the European bound does not describe a riskless trade.
* *Dividend and financing:* the next dividend is an assumption (the last amount paid). {fwd_txt}

{rp.table(rows_fwd)}

* *Borrow:* a reversal shorts SPY and pays the borrow fee, which this test ignores.
* *Backing r out of parity:* a regression of C − P on K cannot pin down r at these maturities. At
  {short['days']} days a slope error of {SLOPE_ERR:g} is {_pct(SLOPE_ERR / (short['T_days'] / 365), 1)} of rate, and early exercise
  tilts C − P away from the money.
* *Synchronicity:* the stock and option quotes are matched to the minute, not the millisecond.
"""


def section_crib(C: dict, P: dict) -> str:
    ex, cl, pu = C["example"], C["call"], C["put"]
    spy = ""
    if P.get("available"):
        e = P["expiries"][1] if len(P["expiries"]) > 1 else P["expiries"][0]
        S = P["meta"]["underlying_price"]
        T = e["T_days"] / 365
        g = bs.greeks(S, round(e["forward"]), T, P["meta"]["rate"]["r_cc"], e["q"], e["smile_atm_iv"], "call")
        spy = (f"On the snapshot, a {e['days']}-day SPY call struck at {round(e['forward']):g} (near the forward) at its "
               f"{_pct(e['smile_atm_iv'])} ATM vol is worth {_money(float(g['price']))}: delta {float(g['delta']):.2f}, "
               f"vega {_money(float(g['vega']) / 100)} per vol point, theta {_money(float(g['theta']) / 365, 2)} a day, "
               f"and gamma {float(g['gamma']):.4f}, so a {_money(S * 0.01)} (1%) move changes delta by {float(g['gamma']) * S * 0.01:.2f}.")
    return f"""## 4. Greeks crib sheet

{mp.picture('f_greeks', 'Call delta, gamma, vega and theta against spot for three maturities')}

Worked example: a call with S = K = {ex['S']:g}, T = {ex['T'] * 365:.0f} days, σ = {_pct(ex['sigma'], 0)},
r = {_pct(ex['r'], 0)}, q = {_pct(ex['q'], 0)} is worth {_money(cl['price'], 3)}; the put is worth {_money(pu['price'], 3)}.
{spy}

| Greek | Meaning | Example call | Long call | Short call | Long put | Short put |
|---|---|---|---|---|---|---|
| Delta Δ = ∂V/∂S | shares-equivalent; hedge ratio | {cl['delta']:.3f} | + | − | − | + |
| Gamma Γ = ∂²V/∂S² | change in Δ per $1; convexity | {cl['gamma']:.4f} | + | − | + | − |
| Vega = ∂V/∂σ | $ per vol point (÷100) | {_money(cl['vega'] / 100, 4)} | + | − | + | − |
| Theta Θ = ∂V/∂t | $ per day (÷365); the rent for gamma | {_money(cl['theta'] / 365, 4)} | − | + | − (usually) | + (usually) |
| Rho ρ = ∂V/∂r | $ per rate point (÷100) | {_money(cl['rho'] / 100, 4)} | + | − | − | + |

**Rules of thumb (checked on the example).** ATM call ≈ 0.4·S·σ·√T = {_money(C['rule_price'], 3)} (exact
{_money(cl['price'], 3)}: the rule is for a strike at the forward, and K = S is {_money(C['itm_forward'], 3)} in the money
forward). ATM vega ≈ 0.4·S·√T = {_money(C['rule_vega'] / 100, 4)} per
vol point (exact {_money(cl['vega'] / 100, 4)}). With r = q = 0 the BSM PDE is Θ = −½Γ S²σ²: {_money(C['zero_r_theta'] / 365, 4)} vs
{_money(-C['zero_r_half_gamma_s2_sigma2'] / 365, 4)} a day. Delta is not the probability of finishing in the money:
the risk-neutral probability is N(d₂) = {C['n_d2']:.3f}, below the delta N(d₁)e^{{−qT}} = {cl['delta']:.3f}.

* **Gamma scalping.** Long an option and delta-hedged, you buy the dips and sell the rallies as the
  hedge rebalances. Each move earns ½ΓΔS², and you pay Θ ≈ −½ΓS²σ_imp² per unit time. Net
  ½ΓS²(σ_real² − σ_imp²)dt: you win if the stock moves more than the implied vol you paid (§2).
* **Short gamma = short realized vol.** The hedged short option collects theta and pays ½ΓΔS² on
  every move, so it loses when realized vol beats implied (§2a). The loss is biggest when big moves
  happen near the strike close to expiry, where Γ is largest.
* **Straddle** (long call + put, same K): long vol, about zero delta at the money, maximum gamma
  and vega. **Strangle** (OTM put + OTM call): cheaper, less gamma near spot, pays off only on large moves.
* **Verticals** (buy K₁, sell K₂ > K₁ calls, or the put version): a directional view with capped
  payoff (at most K₂ − K₁), with vega and gamma partly offset. A tight call spread ÷ (K₂ − K₁) ≈ a
  digital, so it prices the risk-neutral probability of finishing above K, skew included.
* **Skew.** For equity indices, OTM puts trade at higher implied vol than OTM calls: crash
  protection is in demand, and vol rises when prices fall (the leverage effect). Quote it as the
  25-delta risk reversal (call IV − put IV, negative for equities: §3a) or as the IV slope per unit
  of moneyness. A long risk reversal (long call, short put) is long skew.
* **Put–call parity** C − P = S e^{{−qT}} − K e^{{−rT}}; a conversion (long stock, short call, long put)
  or a reversal is a synthetic loan, and a box spread is a pure rate trade (§3b).
"""


def write_markdown(R: dict) -> str:
    H, P, C = R["hedging"], R["snapshot"], R["crib"]
    cfg = H["config"]
    fine = H["scaling"]["labels"][-1]
    low = next(r for r in H["by_frequency"] if r["sigma_real"] == min(cfg["scenarios"]) and r["label"] == fine)
    a_fine = H["attribution"][-1]
    k1 = [next(c for c in H["choices"] if c["k"] == K_HEADLINE and c["kappa_bps"] == kap) for kap in cfg["kappas_bps"]]
    bullets = [
        f"Pricing engine: put–call parity holds to {R['bs']['parity_max_rel_err']:.0e} of the strike, Greeks match "
        f"finite differences to {max(R['bs']['fd_max_rel_err'].values()):.0e} relative, and IVs round-trip to "
        f"{R['bs']['iv_max_abs_err']:.0e}.",
        f"Short {cfg['days']}-day ATM call sold at σ_imp = {_pct(cfg['sigma_imp'], 0)} and hedged every {fine} with "
        f"σ_real = {_pct(min(cfg['scenarios']), 0)}: mean PnL {_money(low['mean'], 4, True)} ± {low['se']:.4f} vs "
        f"C(σ_imp) − C(σ_real) = {_money(low['exact'], 4, True)}. Short gamma earns when realized < implied.",
        f"Hedging-error std falls like N^{H['scaling']['slope_std']:.2f} (theory: 1/√N) while turnover grows like "
        f"N^{H['scaling']['slope_turnover']:.2f}. The best frequency for cost + {K_HEADLINE:g} × std moves from "
        f"{k1[0]['label']} (κ = {k1[0]['kappa_bps']:g} bp) to {k1[-1]['label']} (κ = {k1[-1]['kappa_bps']:g} bp).",
        f"The gamma–theta attribution explains {_pct(a_fine['attr_r2'], 2)} of hedged-PnL variance path by path "
        f"at {fine} hedging (corr {a_fine['attr_corr']:.4f}).",
    ]
    if P.get("available"):
        ex = P["expiries"]
        tot = {k: sum(e["parity_fresh"][k] for e in ex) for k in ("pairs", "mid_violations", "spread_violations",
                                                                   "american_violations")}
        bullets.append(
            f"SPY ({P['meta']['source_key']}, {datetime.fromisoformat(P['meta']['quote_time_et']):%Y-%m-%d %H:%M} ET): ATM IV "
            + ", ".join(f"{_pct(e['smile_atm_iv'])} at {e['days']} days" for e in ex)
            + f". Parity 'fails' for {tot['mid_violations']} of {tot['pairs']} strike pairs at mids, {tot['spread_violations']} "
            f"after crossing the spread, and {tot['american_violations']} against the American band.")
    else:
        bullets.append(f"SPY snapshot unavailable ({P['reason']}); the real-data section is skipped.")
    head = f"""# 04 · Options: Black–Scholes, delta hedging and the SPY smile

*Generated by `python -m markout.options.report` on {R['generated_utc'][:16].replace('T', ' ')} UTC (fixed seeds; the SPY
section reads the saved snapshot and never refetches). Numbers: `reports/results/options.json`.*

{rp.bullets(bullets)}
"""
    return "\n".join([head, section_engine(R["bs"]), section_hedging(H), section_snapshot(P), section_crib(C, P)])


# =========================================================================== main

def main() -> dict:
    t0 = time.time()
    R = {"generated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat()}
    R["bs"] = bs_checks()
    R["hedging"], hfig = run_hedging()
    R["snapshot"], sfig = run_snapshot()
    R["crib"] = crib_numbers()
    fig_vol_bet(hfig["sweep"])
    fig_scaling(hfig, R["hedging"]["scaling"]["slope_std"], R["hedging"]["scaling"]["slope_turnover"])
    fig_frontier(hfig["points"], hfig["choices"])
    fig_attribution(hfig)
    fig_greeks()
    if sfig is not None:
        fig_smile(sfig)
        fig_parity(sfig)
    R["seconds"] = time.time() - t0
    results.save("options", R)
    rp.write("04_options", write_markdown(R))
    print(f"options report: {R['seconds']:.0f} s")
    return R


if __name__ == "__main__":
    main()
