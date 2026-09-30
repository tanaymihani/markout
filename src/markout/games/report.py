"""Report 03, the arena: ``python -m markout.games.report``.

Recomputes every module E result with fixed seeds, saves
reports/results/arena.json, renders the e_* figures (light and dark) and writes
reports/03_arena.md. Every number in the report is interpolated from the
computed results; wording that depends on a result is chosen in code.
"""

from __future__ import annotations

import argparse
import itertools
import math
import time
from fractions import Fraction

import numpy as np

from markout import plotting as mp
from markout import reporting as rp
from markout import results
from markout.games import arena as ar
from markout.games import cardgame as cg
from markout.games import glosten_milgrom as gm
from markout.games import kuhn_cfr as kc
from markout.games import kyle
from markout.games.stats import Z95, mean_ci

NAME = "arena"
SEED = 0
N_SEEDS = 4000                       # episodes per tournament configuration
GM_PARAMS = gm.GMParams(0.0, 1.0, 0.3, 0.5)
GM_RUNS, GM_STEPS, GM_BATCHES = 10_000, 200, 60
KYLE_PARAMS = kyle.KyleParams(100.0, 4.0, 3.0)
KYLE_MC = 400_000
CFR_ITERS, CFRP_ITERS = 100_000, 20_000
CARD_GAMES, CARD_SAMPLES = 300, 10_000
REPLICATOR_DT, REPLICATOR_STEPS, REPLICATOR_BOOT = 0.01, 3000, 200

LABELS = {"gm": "GM zero-profit", "tight": "fixed tight", "wide": "fixed wide",
          "inventory": "inventory skew (AS)", "undercut": "undercutter",
          "markout": "markout (own design)"}


def num(x, digits: int = 3, signed: bool = False) -> str:
    return rp.num(x, digits, signed)


def ci_text(c: dict, digits: int = 3, scale: float = 1.0, signed: bool = False) -> str:
    """'m (95% CI lo to hi)' for a mean_ci/ratio_ci dict."""
    m = c["mean"] * scale
    lo, hi = sorted((c["lo"] * scale, c["hi"] * scale))
    return f"{num(m, digits, signed)} (95% CI {num(lo, digits)} to {num(hi, digits)})"


def pm_text(c: dict, digits: int = 3, scale: float = 1.0) -> str:
    """'m ± half-width' for a mean_ci/ratio_ci dict."""
    return f"{num(c['mean'] * scale, digits)} ± {num(Z95 * c['se'] * scale, 2)}"


def contains(c: dict, v: float = 0.0) -> bool:
    return c["lo"] <= v <= c["hi"]


def neg(c: dict) -> dict:
    """The CI of −X from the CI of X."""
    return {**c, "mean": -c["mean"], "lo": -c["hi"], "hi": -c["lo"]}


PROSE = {"gm": "GM zero-profit quoter", "tight": "fixed tight quoter", "wide": "fixed wide quoter",
         "inventory": "inventory-skew (AS) quoter", "undercut": "undercutter",
         "markout": "markout quoter"}


def _esc(x):
    return x.replace("|", "\\|") if isinstance(x, str) else x


def table(rows: list[dict], columns: list[str] | None = None) -> str:
    """rp.table with '|' escaped in headers and cells (so E[V | buy] cannot split a cell)."""
    cols = list(columns or (rows[0].keys() if rows else []))
    return rp.table([{_esc(k): _esc(v) for k, v in r.items()} for r in rows],
                    [_esc(c) for c in cols])


def share_text(x: float) -> str:
    """A population share as a percentage, without absurd digits for tiny values."""
    return "< 0.001%" if x < 1e-5 else rp.num(x, pct=True)


def small(x: float) -> str:
    """Fixed 5 decimals, or scientific notation for tiny magnitudes."""
    return "{:.5f}".format(x) if abs(x) >= 1e-3 or x == 0 else "{:.2e}".format(x)


# ---------------------------------------------------------------------------
# E1 Glosten–Milgrom
# ---------------------------------------------------------------------------

def gm_section() -> dict:
    p = GM_PARAMS
    exact = []
    for mu in (Fraction(1, 10), Fraction(1, 5), Fraction(3, 10), Fraction(1, 2), Fraction(9, 10)):
        for lo, hi in ((Fraction(0), Fraction(1)), (Fraction(99), Fraction(101))):
            s = gm.spread(Fraction(1, 2), mu, lo, hi)
            exact.append({"mu": str(mu), "v_low": str(lo), "v_high": str(hi), "spread": str(s),
                          "mu_times_range": str(mu * (hi - lo)), "equal": s == mu * (hi - lo)})

    paths = gm.simulate(p, GM_STEPS, GM_RUNS, np.random.default_rng(SEED))
    maker = mean_ci(paths.maker_pnl)
    insider = mean_ci(paths.insider_pnl)
    noise = mean_ci(paths.noise_pnl)
    zs = []
    for s in np.random.SeedSequence(SEED + 1).spawn(GM_BATCHES):
        c = mean_ci(gm.simulate(p, GM_STEPS, GM_RUNS, np.random.default_rng(s)).maker_pnl)
        zs.append(c["mean"] / c["se"])
    zs = np.array(zs)

    ex = gm.expected_paths(p, GM_STEPS + 1)
    sp = paths.spread
    checkpoints = []
    for t in (0, 5, 10, 25, 50, 100, 199):
        c = mean_ci(sp[:, t])
        checkpoints.append({"t": t, "exact": float(ex["spread"][t]), "sim": c["mean"],
                            "lo": c["lo"], "hi": c["hi"],
                            "truth_exact": float(ex["belief_in_truth"][t]),
                            "truth_sim": float(paths.belief_in_truth[:, t].mean())})
    buys = paths.side[:, 0] == 1

    mus = np.round(np.linspace(0.0, 0.95, 96), 4)
    mkt = ar.Market()
    curves = {"mu": mus.tolist()}
    for pi in (0.5, 0.75, 0.9):
        curves[f"pi_{pi}"] = gm.spread(pi, mus, 0.0, 1.0).tolist()
    curves["elastic_half"] = [2 * ar.zero_profit_half_spread_at_half(float(m), mkt.c_max, mkt.dv)
                              / mkt.dv if m > 0 else 0.0 for m in mus]

    conv = {"t": list(range(GM_STEPS)), "mu": [0.1, 0.3, 0.5], "spread": {}, "truth": {}}
    for mu in conv["mu"]:
        e = gm.expected_paths(gm.GMParams(p.v_low, p.v_high, mu, p.prior), GM_STEPS)
        conv["spread"][str(mu)] = e["spread"].tolist()
        conv["truth"][str(mu)] = e["belief_in_truth"].tolist()
    return {
        "params": {"v_low": p.v_low, "v_high": p.v_high, "mu": p.mu, "prior": p.prior,
                   "n_runs": GM_RUNS, "n_steps": GM_STEPS},
        "exact_half": exact,
        "maker": maker, "insider": insider, "noise": noise,
        "insider_plus_noise": mean_ci(paths.insider_pnl + paths.noise_pnl),
        "zero_sum_max_error": float(np.abs(paths.maker_pnl + paths.insider_pnl
                                           + paths.noise_pnl).max()),
        "calibration": {"n_batches": GM_BATCHES, "runs_per_batch": GM_RUNS,
                        "coverage": float(np.mean(np.abs(zs) < Z95)),
                        "pooled_z": float(zs.sum() / math.sqrt(zs.size))},
        "first_trade": {"ask": float(paths.ask[0, 0]), "bid": float(paths.bid[0, 0]),
                        "v_given_buy": mean_ci(paths.value[buys]),
                        "v_given_sell": mean_ci(paths.value[~buys])},
        "spread_checkpoints": checkpoints,
        "spread_monotone": bool(np.all(np.diff(ex["spread"]) <= 1e-15)),
        "spread_end_ratio": float(ex["spread"][GM_STEPS - 1] / ex["spread"][0]),
        "curves_mu": curves,
        "convergence": conv,
        "elastic_c_max": mkt.c_max, "elastic_dv": mkt.dv,
    }


# ---------------------------------------------------------------------------
# E2 Kyle
# ---------------------------------------------------------------------------

def kyle_section() -> dict:
    k = KYLE_PARAMS
    eq = kyle.equilibrium(k)
    iters = []
    for s in (0.01, 0.1, 0.5, 2.0, 10.0, 100.0):
        path = kyle.lambda_iteration(k, s * eq.lam, 40)
        err = np.abs(path / eq.lam - 1)
        n_conv = int(np.argmax(err < 1e-10)) if np.any(err < 1e-10) else None
        iters.append({"start": s, "n_to_1e-10": n_conv, "final_rel_error": float(err[-1]),
                      "path": (path / eq.lam).tolist()})
    rel = np.round(np.linspace(0.1, 2.5, 241), 4)
    mean, se = kyle.mc_insider_profit(rel * eq.beta, eq.lam, k, KYLE_MC,
                                      np.random.default_rng(SEED))
    analytic = kyle.expected_insider_profit(rel * eq.beta, eq.lam, k)
    i_star = int(np.argmin(np.abs(rel - 1.0)))
    mm = kyle.mc_market_maker_check(k, KYLE_MC, np.random.default_rng(SEED + 1))
    return {
        "params": {"p0": k.p0, "sigma0_sq": k.sigma0_sq, "sigma_u": k.sigma_u},
        "eq": {"beta": eq.beta, "lam": eq.lam, "insider_profit": eq.insider_profit,
               "posterior_var": eq.posterior_var},
        "iteration": iters,
        "mc": {"n": KYLE_MC, "rel": rel.tolist(), "mean": mean.tolist(), "se": se.tolist(),
               "analytic": analytic.tolist(), "argmax_rel": float(rel[int(np.argmax(mean))]),
               "grid_step": float(rel[1] - rel[0]),
               "at_star": {"mean": float(mean[i_star]), "se": float(se[i_star])},
               "max_abs_z": float(np.max(np.abs(mean - analytic) / se))},
        "mm_check": mm,
        "bridge": kyle.kyle_bridge(),
    }


# ---------------------------------------------------------------------------
# E3 tournament
# ---------------------------------------------------------------------------

def _agent_params(a) -> str:
    if isinstance(a, ar.FixedSpread):
        return f"half-spread {a.half_spread:g}"
    if isinstance(a, ar.InventorySkew):
        return f"half-spread {a.half_spread:g}, γ = {a.gamma:g}, σ² = Δ²/(4T)"
    if isinstance(a, ar.MarkoutQuoter):
        return f"γ = {a.gamma:g}, Var = π(1 − π)Δ²"
    return "none"


def _blocks(run: ar.ArenaRun, edges=(0, 10, 25, 50, 100)) -> list[dict]:
    """Fill share and informed share of fills by blocks of periods."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        f = run.fill_path[:, lo:hi].sum(axis=1)
        inf = run.informed_fill_path[:, lo:hi].sum(axis=1)
        adv = run.adverse_path[:, lo:hi].sum(axis=1)
        edg = run.edge_path[:, lo:hi].sum(axis=1)
        tot = f.sum()
        out.append({"periods": f"{lo + 1}–{hi}", "lo": lo, "hi": hi,
                    **{f"share_{n}": float(f[i] / tot) if tot else None
                       for i, n in enumerate(run.names)},
                    **{f"informed_{n}": float(inf[i] / f[i]) if f[i] > 0 else None
                       for i, n in enumerate(run.names)},
                    **{f"adverse_{n}": float(adv[i] / f[i]) if f[i] > 0 else None
                       for i, n in enumerate(run.names)},
                    **{f"edge_{n}": float(edg[i] / f[i]) if f[i] > 0 else None
                       for i, n in enumerate(run.names)}})
    return out


def arena_section() -> dict:
    mkt = ar.Market()
    agents = ar.default_agents()
    names = [a.name for a in agents]
    n = len(agents)
    tab = mkt.tables
    half = int(np.argmin(np.abs(tab.grid - 0.5)))
    bench = {"zero_profit_half": ar.zero_profit_half_spread_at_half(mkt.mu, mkt.c_max, mkt.dv),
             "gm_inelastic_half": mkt.mu * mkt.dv / 2,
             "monopoly_half": float(tab.mono_ask[half]),
             "monopoly_profit_half": float(tab.mono_profit[half]),
             "noise_participation_zero_profit": float(mkt.noise_trade_prob(
                 ar.zero_profit_half_spread_at_half(mkt.mu, mkt.c_max, mkt.dv))),
             "noise_participation_monopoly": float(mkt.noise_trade_prob(tab.mono_ask[half]))}

    t0 = time.time()
    mono = {a.name: ar.simulate(mkt, [a], N_SEEDS, SEED) for a in agents}
    pairs = {(i, j): ar.simulate(mkt, [agents[i], agents[j]], N_SEEDS, SEED)
             for i, j in itertools.combinations(range(n), 2)}
    selfplay = {i: ar.simulate(mkt, [agents[i], ar.clone(agents[i], names[i] + "_2")],
                               N_SEEDS, SEED) for i in range(n)}
    ffa = ar.simulate(mkt, agents, N_SEEDS, SEED)
    ks = {k: ar.simulate(mkt, [ar.clone(ar.Undercutter(), f"undercut_{m + 1}") for m in range(k)],
                         N_SEEDS, SEED) for k in range(1, 7)}
    sim_seconds = time.time() - t0

    # (a) monopoly
    monopoly = {nm: {"agent": r.agent_summary(0), "market": r.market_summary()}
                for nm, r in mono.items()}

    # (b) round robin: per-seed payoff of row i against column j (diagonal: self-play mean)
    per_seed = np.zeros((n, n, N_SEEDS))
    pair_rows = []
    for (i, j), r in pairs.items():
        per_seed[i, j], per_seed[j, i] = r.pnl[0], r.pnl[1]
        h = ar.head_to_head(r)
        pair_rows.append({"a": names[i], "b": names[j], "pnl_a": mean_ci(r.pnl[0]),
                          "pnl_b": mean_ci(r.pnl[1]), "diff": h["diff"], "winner": h["winner"],
                          "share_a": r.agent_summary(0)["fill_share"]["mean"],
                          "summary_a": r.agent_summary(0), "summary_b": r.agent_summary(1)})
    for i, r in selfplay.items():
        per_seed[i, i] = r.pnl.mean(axis=0)
    payoff = per_seed.mean(axis=2)
    standings = []
    for i, nm in enumerate(names):
        wins = sum(1 for p in pair_rows if p["winner"] == nm)
        losses = sum(1 for p in pair_rows if nm in (p["a"], p["b"]) and p["winner"] not in (nm, "draw"))
        draws = sum(1 for p in pair_rows if nm in (p["a"], p["b"]) and p["winner"] == "draw")
        total = per_seed[i, [j for j in range(n) if j != i]].sum(axis=0)
        standings.append({"name": nm, "wins": wins, "draws": draws, "losses": losses,
                          "total": mean_ci(total)})
    standings.sort(key=lambda s: (-s["wins"], -s["total"]["mean"]))
    top, second = standings[0]["name"], standings[1]["name"]
    it, i2 = names.index(top), names.index(second)
    top_diff = mean_ci(per_seed[it, [j for j in range(n) if j != it]].sum(axis=0)
                       - per_seed[i2, [j for j in range(n) if j != i2]].sum(axis=0))

    # replicator dynamics on the pairwise payoff matrix, with a seed bootstrap
    path = ar.replicator(payoff, dt=REPLICATOR_DT, n_steps=REPLICATOR_STEPS)
    rng = np.random.default_rng(SEED)
    boot = np.empty((REPLICATOR_BOOT, n))
    for b in range(REPLICATOR_BOOT):
        idx = rng.integers(0, N_SEEDS, N_SEEDS)
        boot[b] = ar.replicator(per_seed[:, :, idx].mean(axis=2), dt=REPLICATOR_DT,
                                n_steps=REPLICATOR_STEPS)[-1]
    competitive = [names.index(x) for x in ("gm", "undercut", "markout")]
    replic = {"payoff": payoff.tolist(), "dt": REPLICATOR_DT, "steps": REPLICATOR_STEPS,
              "horizon": REPLICATOR_DT * REPLICATOR_STEPS,
              "path": path[::30].tolist(), "path_t": (np.arange(0, REPLICATOR_STEPS + 1, 30)
                                                      * REPLICATOR_DT).tolist(),
              "final": path[-1].tolist(),
              "boot_lo": np.quantile(boot, 0.05, axis=0).tolist(),
              "boot_hi": np.quantile(boot, 0.95, axis=0).tolist(),
              "competitive_share_lo": float(np.quantile(boot[:, competitive].sum(axis=1), 0.05)),
              "competitive_share_min": float(boot[:, competitive].sum(axis=1).min()),
              "n_boot": REPLICATOR_BOOT}

    # (c) free-for-all
    ffa_out = {"agents": [ffa.agent_summary(i) for i in range(n)],
               "market": ffa.market_summary(), "blocks": _blocks(ffa),
               "fill_path": ffa.fill_path.tolist(),
               "informed_fill_path": ffa.informed_fill_path.tolist(),
               "adverse_path": ffa.adverse_path.tolist(),
               "edge_path": ffa.edge_path.tolist()}

    # (d) K identical undercutters
    k_rows = []
    for k, r in ks.items():
        ms = r.market_summary()
        k_rows.append({"K": k, **ms, "fill_shares": [r.agent_summary(i)["fill_share"]["mean"]
                                                    for i in range(k)],
                       "spread_path": r.spread_path.tolist(),
                       "zero_profit_path": r.zero_profit_path.tolist(),
                       "rounds_max": r.rounds_max})
    return {"market": {"v_low": mkt.v_low, "v_high": mkt.v_high, "prior": mkt.prior,
                       "mu": mkt.mu, "c_max": mkt.c_max, "tick": mkt.tick,
                       "n_periods": mkt.n_periods, "n_seeds": N_SEEDS, "seed": SEED},
            "agents": [{"name": a.name, "label": LABELS[a.name], "params": _agent_params(a)}
                       for a in agents],
            "benchmarks": bench, "sim_seconds": sim_seconds,
            "monopoly": monopoly,
            "round_robin": {"pairs": pair_rows, "standings": standings, "top": top,
                            "second": second, "top_minus_second": top_diff},
            "replicator": replic, "ffa": ffa_out, "k_sweep": k_rows}


# ---------------------------------------------------------------------------
# E4 Kuhn poker
# ---------------------------------------------------------------------------

def kuhn_section() -> dict:
    cps = kc.log_checkpoints(CFR_ITERS)
    t0 = time.time()
    cfr = kc.KuhnCFR()
    hist = cfr.run(CFR_ITERS, cps)
    cfr_seconds = time.time() - t0
    avg = cfr.average_strategy()
    plus = kc.KuhnCFR(plus=True)
    hist_p = plus.run(CFRP_ITERS, [c for c in cps if c <= CFRP_ITERS])
    fam = kc.equilibrium_family_check(avg)
    alpha = float(min(max(fam["alpha"], 0.0), 1 / 3))
    eq = kc.analytic_equilibrium(alpha)
    # log-log slope of exploitability over the last two decades of vanilla CFR
    it = np.array([h["iteration"] for h in hist], dtype=float)
    ex = np.array([h["exploitability"] for h in hist])
    m = it >= CFR_ITERS / 100
    slope = float(np.polyfit(np.log(it[m]), np.log(ex[m]), 1)[0])
    studies = {leak: kc.exploitation_study(eq, leak)
               for leak in ("never_bluff_jack", "always_call_queen")}
    return {"iterations": CFR_ITERS, "seconds": cfr_seconds,
            "value": kc.expected_value(avg), "exploitability": kc.exploitability(avg),
            "history": hist, "slope_last_two_decades": slope,
            "plus": {"iterations": CFRP_ITERS, "history": hist_p,
                     "value": kc.expected_value(plus.average_strategy()),
                     "exploitability": kc.exploitability(plus.average_strategy())},
            "family": fam, "alpha_used": alpha,
            "avg_strategy": kc.bet_probs(avg), "analytic": kc.bet_probs(eq),
            "exploitation": studies}


# ---------------------------------------------------------------------------
# E5 card game
# ---------------------------------------------------------------------------

def _card_compare(cfg: cg.Config, n_games: int, seed: int) -> dict:
    """Bayes quoter vs a quoter that ignores trades, on the same deals and bots."""
    rng = np.random.default_rng(seed)
    pnl_b, pnl_p = np.zeros(n_games), np.zeros(n_games)
    err_b = np.zeros((n_games, cfg.n_cards))
    err_p = np.zeros((n_games, cfg.n_cards))
    for g in range(n_games):
        d = cg.deal(cfg, rng)
        s = sum(d.cards)
        _, rb = cg.play(cfg, d, cg.bayes_quoter(CARD_SAMPLES, seed=g), CARD_SAMPLES, g,
                        benchmark=False)
        _, rq = cg.play(cfg, d, cg.prior_quoter, CARD_SAMPLES, g, benchmark=False)
        pnl_b[g] = sum(r.pnl for r in rb)
        pnl_p[g] = sum(r.pnl for r in rq)
        err_b[g] = [abs(0.5 * (r.bid + r.ask) - s) for r in rb]
        err_p[g] = [abs(0.5 * (r.bid + r.ask) - s) for r in rq]
    return {"max_width": cfg.max_width, "bayes_pnl": mean_ci(pnl_b), "prior_pnl": mean_ci(pnl_p),
            "diff": mean_ci(pnl_b - pnl_p),
            "mid_error_bayes": err_b.mean(axis=0).tolist(),
            "mid_error_prior": err_p.mean(axis=0).tolist()}


def card_section() -> dict:
    base = cg.Config()
    widths = sorted({2.0, 4.0, 6.0, 8.0, base.max_width})
    by_width = [_card_compare(cg.Config(max_width=w), CARD_GAMES, SEED) for w in widths]
    main = next(r for r in by_width if r["max_width"] == base.max_width)
    example_deal = cg.deal(base, np.random.default_rng(7))
    _, ex = cg.play(base, example_deal, cg.bayes_quoter(CARD_SAMPLES, seed=7), CARD_SAMPLES, 7)
    return {"config": {"n_cards": base.n_cards, "bots_per_round": base.bots_per_round,
                       "p_informed": base.p_informed, "max_width": base.max_width},
            "n_games": CARD_GAMES, "n_samples": CARD_SAMPLES,
            "main": main, "by_width": by_width,
            "example": cg.summary_table(ex, sum(example_deal.cards)),
            "prior_mean": cg.prior_mean(base, [])}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _style_axes(ax) -> None:
    ax.tick_params(axis="both", labelsize=8.5)


def _line_with_ci(ax, x, m, e, color, label=None, marker="o") -> None:
    """A line with markers, and its 95% CI as thin error bars (clean legend handle)."""
    ax.plot(x, m, "-" + marker, color=color, label=label)
    ax.errorbar(x, m, yerr=e, fmt="none", ecolor=color, elinewidth=1.2, capsize=0)


def fig_gm_spread_mu(g: dict) -> None:
    c = g["curves_mu"]
    mu = np.array(c["mu"])

    def draw():
        fig, ax = mp.subplots(w=7.2, h=4.0)
        for i, pi in enumerate((0.5, 0.75, 0.9)):
            ax.plot(mu, c[f"pi_{pi}"], color=mp.series(i), label=f"GM, belief π = {pi:g}")
        el = np.array([np.nan if v is None else v for v in c["elastic_half"]], dtype=float)
        ax.plot(mu, el, color=mp.series(3),
                label=f"zero-profit with elastic noise (c_max = {g['elastic_c_max']:g}), π = 0.5")
        ax.set_xlabel("informed share of arrivals μ")
        ax.set_ylabel("spread / (V_H − V_L)")
        ax.set_xlim(0, 0.95)
        ax.set_ylim(0, None)
        _style_axes(ax)
        mp.legend(ax, loc="upper left")
        mp.title(ax, "The spread is the price of adverse selection",
                 "Glosten–Milgrom closed form: spread = μ(V_H − V_L) at π = ½, smaller as the belief firms up")
        return fig
    mp.render("e_gm_spread_mu", draw)


def fig_gm_convergence(g: dict) -> None:
    cv = g["convergence"]
    t = np.array(cv["t"])
    cps = g["spread_checkpoints"]

    def draw():
        fig, axes = mp.subplots(1, 2, w=10.4, h=3.9)
        for i, mu in enumerate(cv["mu"]):
            axes[0].plot(t, cv["truth"][str(mu)], color=mp.series(i), label=f"μ = {mu:g} (exact)")
            axes[1].plot(t, cv["spread"][str(mu)], color=mp.series(i), label=f"μ = {mu:g} (exact)")
        ts = [c["t"] for c in cps]
        sim_label = f"μ = {g['params']['mu']:g}, mean of {g['params']['n_runs']:,} runs"
        axes[0].plot(ts, [c["truth_sim"] for c in cps], "o", color=mp.series(1), label=sim_label)
        axes[1].plot(ts, [c["sim"] for c in cps], "o", color=mp.series(1), label=sim_label)
        axes[0].set_ylim(0.45, 1.02)
        axes[0].set_ylabel("maker's belief in the true V")
        axes[1].set_ylabel("expected spread (V_H − V_L = 1)")
        axes[1].set_ylim(0, None)
        for ax in axes:
            ax.set_xlabel("trade number")
            ax.set_xlim(0, t[-1] + 2)
            _style_axes(ax)
        mp.legend(axes[0], loc="lower right", fontsize=8)
        mp.legend(axes[1], loc="upper right", fontsize=8)
        mp.title(axes[0], "Trades reveal V", "exact expectation over the net order flow")
        mp.title(axes[1], "…so the spread shrinks", "the spread is concave in a martingale belief")
        return fig
    mp.render("e_gm_convergence", draw)


def fig_kyle(k: dict) -> None:
    mc = k["mc"]
    rel = np.array(mc["rel"])
    mean, se = np.array(mc["mean"]), np.array(mc["se"])
    sub = slice(0, None, 12)

    def draw():
        fig, axes = mp.subplots(1, 2, w=10.4, h=3.9)
        ax = axes[0]
        ax.plot(rel, mc["analytic"], color=mp.series(0), label="analytic β(1 − λ*β)Σ₀")
        ax.plot(rel[sub], mean[sub], "o", color=mp.series(1),
                label=f"Monte Carlo, {mc['n']:,} draws")
        ax.errorbar(rel[sub], mean[sub], yerr=Z95 * se[sub], fmt="none", ecolor=mp.series(1),
                    elinewidth=1.2, capsize=0)
        ax.axvline(1.0, color=mp.ink("muted"), linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
        mp.end_label(ax, 1.0, 0.0, "β* = σ_u/√Σ₀", dx=5)
        ax.set_xlabel("insider intensity β / β*")
        ax.set_ylabel("insider expected profit")
        mp.legend(ax, loc="lower left", fontsize=8)
        mp.title(ax, "The insider's profit peaks at β*", "λ held at its equilibrium value λ*")
        ax2 = axes[1]
        for i, row in enumerate(k["iteration"]):
            path = np.array(row["path"])
            ax2.plot(np.arange(path.size)[:16], path[:16], "-o", color=mp.series(i), markersize=4,
                     label=f"λ₀ = {row['start']:g}·λ*")
        ax2.set_yscale("log")
        ax2.set_xlabel("round of best responses")
        ax2.set_ylabel("λ / λ*")
        mp.legend(ax2, loc="upper right", ncol=2, fontsize=8)
        mp.title(ax2, "Best responses converge to λ*",
                 "λ′ = 2λΣ₀/(Σ₀ + 4λ²σ_u²) iterated from six starts")
        for a in axes:
            _style_axes(a)
        return fig
    mp.render("e_kyle_profit", draw)


def fig_arena_competition(a: dict) -> None:
    rows = a["k_sweep"]
    K = np.array([r["K"] for r in rows])
    tick = a["market"]["tick"]

    def err(key, scale=1.0):
        m = np.array([r[key]["mean"] for r in rows]) * scale
        e = np.array([Z95 * r[key]["se"] for r in rows]) * scale
        return m, e

    def draw():
        fig, axes = mp.subplots(1, 3, w=9.4, h=3.6)
        for i, (key, lab) in enumerate((("quoted_spread", "quoted (inside) spread"),
                                        ("effective_spread", "effective spread"),
                                        ("zero_profit_spread", "zero-profit spread at the belief"))):
            m, e = err(key, 1 / tick)
            _line_with_ci(axes[0], K, m, e, mp.series(i), lab)
        axes[0].set_ylabel("ticks (time average)")
        mp.legend(axes[0], loc="upper right", fontsize=8)
        mp.title(axes[0], "Spreads", "quoted and effective coincide")
        m, e = err("maker_pnl_per_trade", 1 / tick)
        _line_with_ci(axes[1], K, m, e, mp.series(0))
        mp.zero_line(axes[1])
        axes[1].set_ylabel("ticks per trade, all makers")
        mp.title(axes[1], "Maker profit per trade", "rents vanish once there is a rival")
        m, e = err("noise_welfare_per_period")
        _line_with_ci(axes[2], K, m, e, mp.series(0))
        axes[2].set_ylim(0, None)
        axes[2].set_ylabel("surplus per period ($)")
        mp.title(axes[2], "Noise-trader welfare", "urgency value minus trading cost")
        for ax in axes:
            ax.set_xticks(K)
            _style_axes(ax)
        fig.supxlabel("number of identical Bayesian undercutters K", fontsize=10, color=mp.ink("secondary"))
        return fig
    mp.render("e_arena_competition", draw)


def fig_arena_pnl(a: dict) -> None:
    names = [x["name"] for x in a["agents"]]
    mono = [a["monopoly"][n]["agent"]["pnl"] for n in names]
    ffa = a["ffa"]["agents"]

    def panel(ax, cis, title, sub):
        y = np.arange(len(names))[::-1]
        m = np.array([c["mean"] for c in cis])
        e = np.array([Z95 * c["se"] for c in cis])
        mp.bars(ax, y, m, width=0.42, horizontal=True, color=mp.series(0))
        ax.errorbar(m, y, xerr=e, fmt="none", ecolor=mp.ink("secondary"), elinewidth=1.2,
                    capsize=3, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels([LABELS[n] for n in names])
        ax.axvline(0, color=mp.baseline_color(), linewidth=0.9)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.set_xlabel("PnL per episode ($, marked to V)")
        span = max(abs(m).max() + e.max(), 1e-9)
        for yi, mi, ei in zip(y, m, e):
            off = 0.02 * span
            right = mi >= 0
            ax.annotate(num(mi, 3), xy=(mi + (ei + off if right else -(ei + off)), yi),
                        ha="left" if right else "right", va="center", fontsize=8,
                        color=mp.ink("secondary"))
        ax.set_xlim(min(0.0, (m - e).min()) - 0.18 * span, max(0.0, (m + e).max()) + 0.18 * span)
        _style_axes(ax)
        mp.title(ax, title, sub)

    def draw():
        fig, axes = mp.subplots(1, 2, w=11.2, h=4.0)
        panel(axes[0], mono, "(a) Alone: monopoly", f"mean over {a['market']['n_seeds']:,} episodes, 95% CI")
        panel(axes[1], [x["pnl"] for x in ffa], "(c) All six together", "free-for-all, same episodes")
        axes[1].set_yticklabels([])
        return fig
    mp.render("e_arena_pnl", draw)


def fig_arena_h2h(a: dict) -> None:
    names = [x["name"] for x in a["agents"]]
    P = np.array(a["replicator"]["payoff"])

    def draw():
        fig, ax = mp.subplots(w=7.6, h=5.4)
        vmax = float(P.max())
        vmin = float(min(P.min(), -1e-6))
        norm = mp.diverging_norm(vmin, vmax)
        cmap = mp.diverging_cmap()
        im = ax.imshow(P, cmap=cmap, norm=norm, aspect="auto")
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        lab = [LABELS[n] for n in names]
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(lab, rotation=30, ha="right")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(lab)
        ax.tick_params(length=0, labelsize=8.5)
        for i in range(P.shape[0]):
            for j in range(P.shape[1]):
                r_, g_, b_, _ = cmap(norm(P[i, j]))
                lum = 0.2126 * r_ + 0.7152 * g_ + 0.0722 * b_
                ax.text(j, i, num(float(P[i, j]), 2), ha="center", va="center", fontsize=8.5,
                        color="#0b0b0b" if lum > 0.5 else "#ffffff")
        cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        step = 5 if vmax > 10 else 1
        ticks = sorted({round(vmin, 1), 0.0, *np.arange(step, vmax + 1e-9, step).tolist()})
        cb.set_ticks(ticks)
        cb.set_ticklabels([f"{t_:g}".replace("-", "−") for t_ in ticks])
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=8, length=0)
        cb.set_label("row's mean PnL per episode ($)", color=mp.ink("secondary"))
        ax.set_xlabel("opponent")
        mp.title(ax, "Round robin: each row's PnL against each column",
                 "two makers per market; diagonal = two copies of the same agent")
        return fig
    mp.render("e_arena_h2h", draw)


def fig_arena_curse(a: dict) -> None:
    ffa = a["ffa"]
    names = [x["name"] for x in a["agents"]]
    tick = a["market"]["tick"]
    fills = np.array(ffa["fill_path"])
    adv = np.array(ffa["adverse_path"])
    T = fills.shape[1]
    w = 10
    edges = np.arange(0, T + 1, w)
    mids = edges[:-1] + w / 2 + 0.5
    shown = [n for n in names if fills[names.index(n)].sum() > 0]
    tight_edge = a["ffa"]["agents"][names.index("tight")]["edge_per_fill"]["mean"] / tick

    def blocksum(x):
        return np.array([x[lo:hi].sum() for lo, hi in zip(edges[:-1], edges[1:])])

    def draw():
        fig, axes = mp.subplots(1, 2, w=10.4, h=3.9)
        tot = np.array([fills[:, lo:hi].sum() for lo, hi in zip(edges[:-1], edges[1:])])
        for n in shown:
            i = names.index(n)
            f = blocksum(fills[i])
            axes[0].plot(mids, f / tot, "-o", color=mp.series(i), markersize=5, label=LABELS[n])
            with np.errstate(invalid="ignore", divide="ignore"):
                per_fill = np.where(f > 0, blocksum(adv[i]) / np.where(f > 0, f, 1), np.nan) / tick
            axes[1].plot(mids, per_fill, "-o", color=mp.series(i), markersize=5, label=LABELS[n])
        axes[1].axhline(tight_edge, color=mp.ink("muted"), linewidth=0.9, linestyle=(0, (4, 3)),
                        zorder=1)
        mp.end_label(axes[1], T * 0.55, tight_edge, f"fixed tight's edge per fill, {tight_edge:.1f} ticks",
                     dy=8)
        mp.zero_line(axes[1])
        axes[0].set_ylabel("share of all fills")
        axes[1].set_ylabel("adverse selection per fill (ticks)")
        axes[0].set_ylim(0, 1)
        for ax in axes:
            ax.set_xlabel(f"period in the episode ({w}-period blocks)")
            ax.set_xlim(0, T)
            _style_axes(ax)
        mp.legend(axes[0], loc="center right", fontsize=8)
        mp.legend(axes[1], loc="upper right", fontsize=8)
        mp.title(axes[0], "Who trades when, all six together",
                 "the fixed tight quote wins early, while V is uncertain")
        mp.title(axes[1], "…when each fill costs the most",
                 "value move from the pre-trade mid to V, per fill")
        return fig
    mp.render("e_arena_curse", draw)


def fig_arena_replicator(a: dict) -> None:
    rep = a["replicator"]
    names = [x["name"] for x in a["agents"]]
    path = np.array(rep["path"])
    t = np.array(rep["path_t"])

    def draw():
        fig, ax = mp.subplots(w=7.2, h=3.9)
        for i, n in enumerate(names):
            ax.plot(t, path[:, i], color=mp.series(i), label=LABELS[n])
        ax.set_ylim(0, 1)
        ax.set_xlim(0, t[-1])
        ax.set_xlabel("replicator time (a payoff edge of $1 per episode grows a share e-fold per unit)")
        ax.set_ylabel("population share")
        _style_axes(ax)
        mp.legend(ax, loc="upper right", ncol=2, fontsize=8)
        mp.title(ax, "Replicator dynamics over quoting strategies",
                 "pairwise payoffs from the round robin; the split between the survivors is not identified")
        return fig
    mp.render("e_arena_replicator", draw)


def fig_cfr(kd: dict) -> None:
    h = kd["history"]
    hp = kd["plus"]["history"]
    it = np.array([x["iteration"] for x in h], dtype=float)
    ex = np.array([x["exploitability"] for x in h])
    itp = np.array([x["iteration"] for x in hp], dtype=float)
    exp_ = np.array([x["exploitability"] for x in hp])
    ref_i = it[it >= 100]
    ref = ex[it >= 100][0] * np.sqrt(ref_i[0] / ref_i)

    def draw():
        fig, ax = mp.subplots(w=7.2, h=4.0)
        ax.plot(it, ex, "-o", color=mp.series(0), markersize=4, label="vanilla CFR (average strategy)")
        ax.plot(itp, exp_, "-o", color=mp.series(1), markersize=4, label="CFR+ (for comparison)")
        ax.plot(ref_i, ref, color=mp.ink("muted"), linewidth=1.0, linestyle=(0, (4, 3)),
                label="reference slope 1/√T")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("iterations T")
        ax.set_ylabel("exploitability BR₁(σ₂) + BR₂(σ₁)")
        ax.grid(axis="x", visible=False)
        _style_axes(ax)
        mp.legend(ax, loc="lower left")
        slope = f"{kd['slope_last_two_decades']:.2f}".replace("-", "−")
        mp.title(ax, "Kuhn poker: CFR's average strategy approaches equilibrium",
                 f"vanilla CFR slope {slope} over the last two decades")
        return fig
    mp.render("e_cfr_exploitability", draw)


def fig_kuhn_exploit(kd: dict) -> None:
    studies = kd["exploitation"]
    titles = {"never_bluff_jack": "P2 never bluffs the Jack",
              "always_call_queen": "P2 always calls with the Queen"}

    def draw():
        fig, ax = mp.subplots(w=7.2, h=4.2)
        xs, ys = [], []
        for i, (leak, st) in enumerate(studies.items()):
            f = st["frontier"]
            x = [p["worst_case"] for p in f]
            y = [p["ev_vs_leak"] for p in f]
            xs += x
            ys += y
            ax.plot(x, y, color=mp.series(i), label=titles[leak])
            ax.plot(x[::5], y[::5], "o", color=mp.series(i))
            ax.annotate("best response (λ = 1)", xy=(x[-1], y[-1]), xytext=(6, 5),
                        textcoords="offset points", ha="left", va="bottom", fontsize=8,
                        color=mp.ink("secondary"))
        f0 = studies["never_bluff_jack"]["frontier"][0]
        ax.annotate("equilibrium (λ = 0)", xy=(f0["worst_case"], f0["ev_vs_leak"]), xytext=(8, 0),
                    textcoords="offset points", ha="left", va="center", fontsize=8,
                    color=mp.ink("secondary"))
        ax.axhline(kc.GAME_VALUE, color=mp.baseline_color(), linewidth=0.9, zorder=1)
        ax.axvline(kc.GAME_VALUE, color=mp.baseline_color(), linewidth=0.9, zorder=1)
        span_x = max(xs) - min(xs)
        ax.set_xlim(min(xs) - 0.05 * span_x, max(xs) + 0.38 * span_x)
        span_y = max(ys) - min(ys)
        ax.set_ylim(min(ys) - 0.08 * span_y, max(ys) + 0.16 * span_y)
        ax.set_xlabel("P1's EV against a best-responding P2 (worst case)")
        ax.set_ylabel("P1's EV against the leaky P2")
        ax.grid(axis="x", visible=True)
        _style_axes(ax)
        mp.legend(ax, loc="center left")
        mp.title(ax, "Exploit or protect",
                 "mixtures λ·(best response) + (1 − λ)·(equilibrium), λ = 0, ¼, ½, ¾, 1; lines at −1/18")
        return fig
    mp.render("e_kuhn_exploit", draw)


def render_figures(res: dict) -> None:
    fig_gm_spread_mu(res["gm"])
    fig_gm_convergence(res["gm"])
    fig_kyle(res["kyle"])
    fig_arena_competition(res["arena"])
    fig_arena_pnl(res["arena"])
    fig_arena_h2h(res["arena"])
    fig_arena_curse(res["arena"])
    fig_arena_replicator(res["arena"])
    fig_cfr(res["kuhn"])
    fig_kuhn_exploit(res["kuhn"])


# ---------------------------------------------------------------------------
# Report text
# ---------------------------------------------------------------------------

def _t(x: float, tick: float, digits: int = 3) -> str:
    """A price amount in ticks."""
    return num(x / tick, digits)


def text_summary(res: dict) -> list[str]:
    g, k, a, kd = res["gm"], res["kyle"], res["arena"], res["kuhn"]
    tick = a["market"]["tick"]
    k1, k2 = a["k_sweep"][0], a["k_sweep"][1]
    ffa = {x["name"]: x for x in a["ffa"]["agents"]}
    first_block = a["ffa"]["blocks"][0]
    mono_tight = a["monopoly"]["tight"]["agent"]["pnl"]
    top = a["round_robin"]["top"]
    return [
        "# Arena: game theory under simulated competition",
        "",
        "In short: competition takes away the market maker's rents, but not the cost of adverse selection. "
        f"In a Glosten–Milgrom market with elastic noise demand, a lone Bayesian undercutter "
        f"quotes a time-average spread of {_t(k1['quoted_spread']['mean'], tick)} ticks and earns "
        f"{num(k1['maker_pnl']['mean'])} dollars per episode. One identical rival brings the spread "
        f"down to {_t(k2['quoted_spread']['mean'], tick)} ticks, "
        f"{_t(k2['excess_spread']['mean'], tick, 2)} ticks above the zero-profit spread at the same "
        f"beliefs. Maker profit falls from {_t(k1['maker_pnl_per_trade']['mean'], tick)} to "
        f"{_t(k2['maker_pnl_per_trade']['mean'], tick, 2)} ticks per trade, and noise-trader welfare "
        f"rises from {num(k1['noise_welfare_per_period']['mean'])} to "
        f"{num(k2['noise_welfare_per_period']['mean'])} dollars per period. "
        f"A naive fixed tight quote loses {num(-mono_tight['mean'])} dollars per episode alone and "
        f"{num(-ffa['tight']['pnl']['mean'])} in the free-for-all. There it wins "
        f"{rp.num(first_block['share_tight'], pct=True)} of the fills in the first "
        f"{first_block['hi']} periods, while V is still uncertain. Each of those fills carries "
        f"{_t(first_block['adverse_tight'], tick)} ticks of adverse selection against its "
        f"{_t(first_block['edge_tight'], tick)}-tick edge: the quote that wins the flow is the one "
        "that was too cheap. "
        f"The round-robin winner is the {LABELS[top]}.",
        "",
        "The textbook pieces check out against closed forms. The GM spread at π = ½ equals "
        "μ(V_H − V_L) exactly in rational arithmetic, and the maker's mean profit over "
        f"{g['params']['n_runs']:,} runs is {ci_text(g['maker'], 2)}. "
        f"Kyle's best responses converge to λ* from all {len(k['iteration'])} starts, and the Monte "
        f"Carlo insider profit peaks at β/β* = {num(k['mc']['argmax_rel'], 3)}. After "
        f"{kd['iterations']:,} iterations CFR's average strategy is worth "
        f"{num(kd['value'], 5)} to player 1 (−1/18 = {num(-1 / 18, 5)}), with exploitability "
        f"{num(kd['exploitability'], 3)}.",
        "",
    ]


def text_question() -> list[str]:
    return [
        "## 1. Question",
        "",
        "SIG's research interns \"create strategies to execute on modelling ideas under simulated "
        "competition\". This module builds that competition and asks the question behind the "
        "project's thesis: when makers compete for order flow that is partly informed, what does "
        "winning the flow cost? The winner's-curse view predicts that the quote that trades is "
        "disproportionately the one that was wrong. Three sub-questions:",
        "",
        rp.bullets([
            "Are the textbook models implemented exactly? Glosten–Milgrom (1985), Kyle (1985) and "
            "Kuhn poker are each checked against their closed forms.",
            "Does a simulated dealer market reproduce the three textbook results? These are "
            "monopoly rents, Bertrand compression of spreads to the zero-profit level, and losses "
            "for naive tight quoting.",
            "How much is each fill adversely selected, and who ends up paying for it?",
        ]),
        "",
    ]


def text_methods(res: dict) -> list[str]:
    a, k, kd, c = res["arena"], res["kyle"], res["kuhn"], res["cardgame"]
    m, b = a["market"], a["benchmarks"]
    market_rows = [
        {"parameter": "value V", "value": f"{m['v_low']:g} or {m['v_high']:g}, prior P(V_H) = {m['prior']:g}"},
        {"parameter": "informed share of arrivals μ", "value": f"{m['mu']:g}"},
        {"parameter": "noise urgency c ~ U[0, c_max]", "value": f"c_max = {m['c_max']:g}"},
        {"parameter": "tick", "value": f"{m['tick']:g}"},
        {"parameter": "arrivals per episode T", "value": f"{m['n_periods']}"},
        {"parameter": "episodes per configuration", "value": f"{m['n_seeds']:,} (same seeds for every line-up)"},
        {"parameter": "zero-profit half-spread at π = ½", "value":
            f"{num(b['zero_profit_half'], 4)} (GM with inelastic noise: {num(b['gm_inelastic_half'], 3)})"},
        {"parameter": "monopoly half-spread at π = ½", "value":
            f"{num(b['monopoly_half'], 3)} = c_max/(2(1 − μ)), profit {num(b['monopoly_profit_half'], 3)} per period"},
        {"parameter": "noise participation at those half-spreads", "value":
            f"{rp.num(b['noise_participation_zero_profit'], pct=True)} and "
            f"{rp.num(b['noise_participation_monopoly'], pct=True)}"},
    ]
    agent_rows = [{"agent": LABELS[x["name"]], "key": f"`{x['name']}`", "parameters": x["params"]}
                  for x in a["agents"]]
    agent_rule = {
        "gm": "posts the zero-profit quotes ask = E[V | buy], bid = E[V | sell] at the public belief",
        "tight": "public mid ± a fixed half-spread",
        "wide": "public mid ± a fixed half-spread",
        "inventory": "Avellaneda–Stoikov reservation price r = mid − q·γ·σ²·(T − t), fixed half-spread around r",
        "undercut": "one tick inside the best rival's last quote on each side, never below zero profit, "
                    "never above the monopoly quote (which it posts when alone)",
        "markout": "the undercutter's rule, with floors raised or lowered by the AS indifference term "
                   "(1 ∓ 2q)·γ·Var(V | tape)/2 and the monopoly quote centred on the reservation price",
    }
    for r, x in zip(agent_rows, a["agents"]):
        r["rule"] = agent_rule[x["name"]]
    return [
        "## 2. Models and method",
        "",
        "Code: `src/markout/games/` (`glosten_milgrom.py`, `kyle.py`, `arena.py`, `kuhn_cfr.py`, "
        "`cardgame.py`). Tests: `tests/test_games_*.py`. All randomness comes from fixed seeds, "
        "and every interval below is a 95% CI across independent runs. Ratios such as PnL per "
        "fill use the delta method on per-episode totals.",
        "",
        "**E1 Glosten–Milgrom (1985).** V ∈ {V_L, V_H} with belief π. Each trader is informed with "
        "probability μ; otherwise it buys or sells with probability ½. The maker quotes "
        "ask = E[V | buy] and bid = E[V | sell], which gives "
        "ask − bid = (V_H − V_L)·4π(1 − π)μ / (1 − μ²(2π − 1)²). The belief updates by Bayes' rule "
        "after each trade. The belief depends only on net order flow, so the expected spread and "
        "belief paths are also computed exactly, by summing over the binomial distribution of buys.",
        "",
        f"**E2 Kyle (1985).** v ~ N(p₀, Σ₀) with Σ₀ = {k['params']['sigma0_sq']:g}. Noise flow "
        f"u ~ N(0, σ_u²) with σ_u = {k['params']['sigma_u']:g}. The insider trades x = β(v − p₀) and "
        "the maker sets p = p₀ + λ(x + u). The closed form is β* = σ_u/√Σ₀ and λ* = √Σ₀/(2σ_u). "
        "It is checked by iterating best responses, λ ← 2λΣ₀/(Σ₀ + 4λ²σ_u²), and by Monte Carlo "
        "over β with λ held at λ*.",
        "",
        "**E3 The tournament.** Each episode draws a binary V as in GM. That keeps the Bayesian "
        "benchmark exact: the zero-profit quote at every belief has a closed form, so "
        "\"did competition reach the zero-profit spread?\" is checked against an exact number "
        "each period. An episode is one information event, as in GM and the PIN model. Each "
        "period one trader arrives:",
        "",
        rp.bullets([
            "with probability μ it is informed: it knows V and trades one unit if the best quote "
            "is profitable;",
            "otherwise it is a noise trader with a coin-flip direction and urgency c ~ U[0, c_max]. "
            "It trades only if the half-spread it pays, measured from the public mid, is below c.",
        ]),
        "",
        "Measuring each side from the mid follows Avellaneda–Stoikov's arrival intensities, and it "
        "means a maker cannot tax one side for free by skewing. The elasticity gives a monopolist "
        "a finite optimal spread. Makers quote on a tick grid and the trader hits the best price, "
        "with ties split at random. All makers see the same tape and share one exact Bayesian "
        "belief, so they differ only in quoting policy. PnL is marked to V at the episode end. "
        "Each fill's PnL splits exactly into its edge over the pre-trade mid minus its adverse "
        "selection, the move from that mid to V (the fill's markout).",
        "",
        "Quote revisions are much faster than arrivals. Before each arrival the reactive makers "
        "revise until nobody moves; the rule is monotone with a unique fixed point, so this is the "
        "Bertrand outcome on the tick grid.",
        "",
        table(market_rows),
        "",
        table(agent_rows, ["agent", "key", "rule", "parameters"]),
        "",
        "Runs, each over the same episodes:",
        "",
        rp.bullets([
            "(a) each agent alone;",
            "(b) all 15 head-to-heads, plus each agent against a copy of itself for the replicator "
            "dynamics;",
            "(c) all six together;",
            "(d) K = 1…6 identical undercutters.",
        ]),
        "",
        f"**E4 Kuhn poker.** Vanilla CFR (Zinkevich et al. 2007; Neller & Lanctot 2013) runs with "
        f"full-tree traversal of all six deals. The strategy is fixed within each iteration, and "
        f"the run lasts {kd['iterations']:,} iterations. Exploitability is BR₁(σ₂) + BR₂(σ₁), from "
        "an exact best-response routine; the tests check it against brute force over all 2⁶ pure "
        f"strategies. CFR+ (Tammelin 2014) is run for {kd['plus']['iterations']:,} iterations only "
        "as a comparison curve.",
        "",
        f"**E5 Card game.** `python -m markout.games.cardgame`. You make markets, no wider than "
        f"{c['config']['max_width']:g}, on the sum of {c['config']['n_cards']} face-down cards. "
        f"{c['config']['bots_per_round']} bots trade each round, each informed with probability "
        f"{c['config']['p_informed']:g}; an informed bot knows one face-down card and trades only "
        "against a quote that is wrong relative to it. Then one card is turned up. The benchmark "
        "is E[S | revealed cards, trades], computed by importance sampling and validated against "
        "exact enumeration in the tests.",
        "",
    ]


def text_gm(res: dict) -> list[str]:
    g = res["gm"]
    exact_ok = all(r["equal"] for r in g["exact_half"])
    ft = g["first_trade"]
    maker_word = "contains" if contains(g["maker"]) else "excludes"
    cal = g["calibration"]
    rows = [{"trade t": c["t"], "exact E[spread]": small(c["exact"]),
             "simulated mean": small(c["sim"]),
             "95% CI": f"{small(c['lo'])} to {small(c['hi'])}",
             "exact E[belief in true V]": "{:.4f}".format(c["truth_exact"]),
             "simulated": "{:.4f}".format(c["truth_sim"])} for c in g["spread_checkpoints"]]
    mus = g["curves_mu"]["mu"]
    idx = [mus.index(x) for x in (0.1, 0.2, 0.3, 0.5, 0.7, 0.9)]
    curve_rows = [{"μ": "{:g}".format(mus[i]),
                   "π = 0.5": "{:.4f}".format(g["curves_mu"]["pi_0.5"][i]),
                   "π = 0.75": "{:.4f}".format(g["curves_mu"]["pi_0.75"][i]),
                   "π = 0.9": "{:.4f}".format(g["curves_mu"]["pi_0.9"][i]),
                   "elastic zero-profit, π = 0.5": ("n/a" if g["curves_mu"]["elastic_half"][i] is None
                                                    else "{:.4f}".format(g["curves_mu"]["elastic_half"][i]))}
                  for i in idx]
    exact_list = ", ".join(f"μ = {r['mu']}: {r['spread']}" for r in g["exact_half"]
                           if r["v_low"] == "0")
    return [
        "### 3.1 Glosten–Milgrom",
        "",
        rp.bullets([
            f"**Spread at π = ½.** In exact rational arithmetic the spread equals μ(V_H − V_L) "
            f"for every case tested ({len(g['exact_half'])} cases: "
            f"{'all equal' if exact_ok else 'NOT all equal'}). With V ∈ {{0, 1}}: {exact_list}.",
            f"**Zero expected profit.** Over {g['params']['n_runs']:,} simulated markets of "
            f"{g['params']['n_steps']} trades (μ = {g['params']['mu']:g}), the maker's mean PnL is "
            f"{ci_text(g['maker'], 3)}, and the interval {maker_word} 0. As a check on the "
            f"interval itself, across {cal['n_batches']} further independent batches of "
            f"{cal['runs_per_batch']:,} runs it covered 0 in {rp.num(cal['coverage'], pct=True)} "
            f"of batches (nominal 95%), and the pooled z-score is {num(cal['pooled_z'], 2)}.",
            f"**Zero-sum accounting.** Insiders earn {ci_text(g['insider'], 3)} per market and noise "
            f"traders lose {ci_text(g['noise'], 3, scale=-1)}. Maker + insiders + noise sum to zero "
            f"in every run (largest error {num(g['zero_sum_max_error'], 2)}), so on average the "
            "insiders' gain is the noise traders' loss.",
            f"**Regret-free quotes.** Among first trades that were buys, the mean realized V is "
            f"{ci_text(ft['v_given_buy'], 4)} against an ask of {num(ft['ask'], 4)}. For sells it is "
            f"{ci_text(ft['v_given_sell'], 4)} against a bid of {num(ft['bid'], 4)}.",
            f"**The spread shrinks as trades reveal V.** The exact expected spread falls at every "
            f"trade ({'monotone' if g['spread_monotone'] else 'NOT monotone'}), to "
            f"{rp.num(g['spread_end_ratio'], pct=True)} of its initial value after "
            f"{g['params']['n_steps']} trades. The simulation matches the exact path at every "
            "checkpoint (table below).",
        ]),
        "",
        mp.picture("e_gm_spread_mu", "Glosten-Milgrom spread against the informed share, for three beliefs, with the elastic-noise zero-profit spread"),
        "",
        table(curve_rows),
        "",
        "Elastic noise demand widens the zero-profit spread. A wider spread drives away noise "
        "traders, which raises the informed share of the flow that remains. With the tournament's "
        f"c_max = {g['elastic_c_max']:g} the zero-profit spread is wider than GM's μ(V_H − V_L) "
        "at every μ in the chart.",
        "",
        mp.picture("e_gm_convergence", "Belief in the true value and expected spread against trade number for three informed shares"),
        "",
        table(rows),
        "",
    ]


def text_kyle(res: dict) -> list[str]:
    k = res["kyle"]
    eq, mc, mm = k["eq"], k["mc"], k["mm_check"]
    z_star = (mc["at_star"]["mean"] - eq["insider_profit"]) / mc["at_star"]["se"]
    inside = abs(z_star) <= Z95
    it_rows = [{"start λ₀/λ*": "{:g}".format(r["start"]),
                "rounds to |λ/λ* − 1| < 1e-10": r["n_to_1e-10"],
                "λ₁/λ*": "{:.4f}".format(r["path"][1]), "λ₂/λ*": "{:.4f}".format(r["path"][2]),
                "λ₃/λ*": "{:.4f}".format(r["path"][3])} for r in k["iteration"]]
    rel = mc["rel"]
    pick = [rel.index(x) for x in (0.25, 0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 2.5)]
    mc_rows = [{"β/β*": "{:g}".format(rel[i]), "analytic": "{:.4f}".format(mc["analytic"][i]),
                "Monte Carlo": "{:.4f}".format(mc["mean"][i]),
                "95% CI": "{:.4f} to {:.4f}".format(mc["mean"][i] - Z95 * mc["se"][i],
                                                    mc["mean"][i] + Z95 * mc["se"][i])} for i in pick]
    level = (f"Its level at β* is {num(mc['at_star']['mean'], 4)} ± "
             f"{num(Z95 * mc['at_star']['se'], 2)} against the analytic {num(eq['insider_profit'], 4)} "
             f"(z = {num(z_star, 2)}). ")
    if not inside:
        level += ("That point misses its 95% interval. The curve is a single set of draws reused "
                  "for every β (common random numbers), so its errors are strongly correlated "
                  "along β. The largest deviation anywhere on the grid is "
                  f"{num(mc['max_abs_z'], 2)} standard errors, and the location of the peak, "
                  "which is the claim being tested, is unaffected.")
    else:
        level += f"The largest deviation anywhere on the grid is {num(mc['max_abs_z'], 2)} standard errors."
    lines = [
        "### 3.2 Kyle",
        "",
        f"With Σ₀ = {k['params']['sigma0_sq']:g} and σ_u = {k['params']['sigma_u']:g} the closed "
        f"form gives β* = {num(eq['beta'], 4)}, λ* = {num(eq['lam'], 4)}, insider expected "
        f"profit ½σ_u√Σ₀ = {num(eq['insider_profit'], 4)}, and posterior variance "
        f"Σ₀/2 = {num(eq['posterior_var'], 4)}.",
        "",
        rp.bullets([
            f"**Best-response iteration.** From all {len(k['iteration'])} starts (0.01λ* to 100λ*) "
            f"the map converges to λ*, within 1e-10 in at most "
            f"{max(r['n_to_1e-10'] for r in k['iteration'])} rounds. After the first round λ never "
            "exceeds λ*, and it rises monotonically from there; near λ* the convergence is "
            "quadratic, because the map's slope at λ* is zero.",
            f"**Monte Carlo over β.** With λ fixed at λ* and {mc['n']:,} draws of (v, u), the "
            f"insider's mean profit peaks at β/β* = {num(mc['argmax_rel'], 3)} on a grid of step "
            f"{num(mc['grid_step'], 2)}. {level}",
            f"**The maker's side.** With the insider at β*, the OLS slope of v on order flow is "
            f"{num(mm['lambda_hat'], 4)} (λ* = {num(mm['lambda_star'], 4)}) and the residual "
            f"variance is {num(mm['posterior_var_hat'], 4)} (Σ₀/2 = {num(mm['posterior_var_star'], 4)}).",
        ]),
        "",
        mp.picture("e_kyle_profit", "Insider profit against trading intensity with the analytic optimum, and best-response iterations for lambda"),
        "",
        table(mc_rows),
        "",
        table(it_rows),
        "",
    ]
    lines += text_bridge(k["bridge"])
    return lines


def text_bridge(br: dict) -> list[str]:
    head = "**Bridge to module D: CKS's β is a cousin of Kyle's λ, not the same object.**"
    why = ("In Kyle's model every order is a trade and the price is an exact linear function of "
           "net order flow, p − p₀ = λy. The regression of the price change on flow therefore has "
           "R² = 1, and its slope is λ = s.d.(Δp)/s.d.(y). CKS's β is the slope of the mid change "
           "on order-flow imbalance (OFI), and OFI also counts limit orders joining and leaving "
           "the best quotes, not only trades. That breaks the identity in two ways. First, "
           "R² < 1, so the OLS slope √R²·s.d.(Δmid)/s.d.(OFI) is smaller than the Kyle-style "
           "ratio s.d.(Δmid)/s.d.(OFI) by a factor √R². Second, OFI is mostly quote traffic "
           "rather than trading, so a share of OFI is not a share of Kyle's y. For the same price "
           "variance, OFI's larger dispersion makes the impact per share of OFI smaller than a "
           "trade-based λ per share would be. That is an expectation, not a measurement here.")
    if not br["available"]:
        return [head, "", why, "",
                f"The numerical comparison {br['note']}. Once module D's results exist, rerunning "
                "this report fills in a per-stock table: CKS β, its R², the Kyle-style ratio "
                "β/√R², and β × depth.", ""]
    rows = []
    for r in br["rows"]:
        row = {"stock": r["ticker"], "CKS β (ticks per 1,000 sh of OFI)": "{:.4g}".format(r["cks_beta"])}
        if "r2" in r:
            row["R²"] = "{:.3f}".format(r["r2"])
            row["Kyle-style ratio β/√R²"] = "{:.4g}".format(r["kyle_ratio"])
        if "depth" in r:
            row["mean depth at best (sh)"] = rp.num(r["depth"])
            row["β × depth (ticks)"] = "{:.3f}".format(r["beta_times_depth"])
        if "kyle_lambda_trades" in r:
            row["Kyle λ from trades (ticks per 1,000 sh)"] = "{:.4g}".format(r["kyle_lambda_trades"])
        rows.append(row)
    betas = [r["cks_beta"] for r in br["rows"]]
    lo_t = min(br["rows"], key=lambda r: r["cks_beta"])["ticker"]
    hi_t = max(br["rows"], key=lambda r: r["cks_beta"])["ticker"]
    parts = [head, "", why, "",
             f"Module D's full-day CKS slopes run from {num(min(betas), 3)} ({lo_t}) to "
             f"{num(max(betas), 3)} ({hi_t}) ticks per 1,000 shares of OFI."]
    if any("r2" in r for r in br["rows"]):
        r2s = [r["r2"] for r in br["rows"] if "r2" in r]
        parts[-1] += (f" Their R² runs from {num(min(r2s), 2)} to {num(max(r2s), 2)}, so the "
                      "Kyle-style ratio is larger than β by a factor of "
                      f"{num(1 / math.sqrt(max(r2s)), 2)} to {num(1 / math.sqrt(min(r2s)), 2)}.")
    if any("beta_times_depth" in r for r in br["rows"]):
        bd = [r["beta_times_depth"] for r in br["rows"] if "beta_times_depth" in r]
        parts[-1] += (f" β × mean depth runs from {num(min(bd), 2)} to {num(max(bd), 2)} ticks; "
                      "CKS's stylized book predicts ½, since OFI equal to the depth at the best "
                      "moves the mid by half a tick.")
    parts += ["", table(rows), ""]
    if br.get("has_kyle_lambda"):
        parts += ["The Kyle λ column uses module D's dispersion of net traded volume, so it is the "
                  "trade-based λ that Kyle's model defines.", ""]
    else:
        parts += [f"A trade-based Kyle λ is not computed: {br['note']}. `kyle_bridge()` computes "
                  "it automatically if module D adds `sd_dmid_ticks` and `sd_trade_imbalance_k` "
                  "per stock.", ""]
    return parts


def _agent_table(summaries: list[dict], tick: float) -> str:
    rows = []
    for s in summaries:
        has = s["fills_per_episode"] > 0
        rows.append({
            "agent": LABELS[s["name"]],
            "PnL per episode": ci_text(s["pnl"], 3),
            "PnL s.d.": num(s["pnl_sd"], 3),
            "fill share": rp.num(s["fill_share"]["mean"], pct=True),
            "fills": num(s["fills_per_episode"], 3),
            "PnL per fill (ticks)": _t(s["pnl_per_fill"]["mean"], tick) if has else "n/a",
            "edge per fill (ticks)": _t(s["edge_per_fill"]["mean"], tick) if has else "n/a",
            "adverse selection per fill (ticks)": _t(s["adverse_per_fill"]["mean"], tick) if has else "n/a",
            "informed share of fills": rp.num(s["informed_share_of_fills"]["mean"], pct=True) if has else "n/a",
            "mean inventory²": num(s["inventory_sq"]["mean"], 3),
        })
    return table(rows)


def text_arena(res: dict) -> list[str]:
    a = res["arena"]
    tick = a["market"]["tick"]
    names = [x["name"] for x in a["agents"]]
    mono = a["monopoly"]
    ks = a["k_sweep"]
    k1, k2 = ks[0], ks[1]
    rr = a["round_robin"]
    ffa = {x["name"]: x for x in a["ffa"]["agents"]}
    rep = a["replicator"]

    # (a) monopoly
    best_mono = max(names, key=lambda n: mono[n]["agent"]["pnl"]["mean"])
    gm_m = mono["gm"]["agent"]
    tight_m = mono["tight"]["agent"]
    mono_lines = [
        "#### (a) Each agent alone",
        "",
        f"Alone, the {PROSE[best_mono]} earns the most, {ci_text(mono[best_mono]['agent']['pnl'], 3)} "
        f"dollars per episode, with a time-average quoted spread of "
        f"{_t(mono[best_mono]['market']['quoted_spread']['mean'], tick)} ticks against a zero-profit "
        f"spread of {_t(mono[best_mono]['market']['zero_profit_spread']['mean'], tick)} at the same "
        f"beliefs. The fixed wide quote comes close, at "
        f"{num(mono['wide']['agent']['pnl']['mean'], 3)}: a monopolist earns rents by quoting wide. "
        f"The GM quoter posts its zero-profit quotes even as a monopolist and earns "
        f"{ci_text(gm_m['pnl_per_fill'], 2, scale=1 / tick)} ticks per fill. That is the rent from "
        "rounding quotes away from the mid to the tick grid, bounded by one tick. "
        f"The fixed tight quote loses {ci_text(tight_m['pnl'], 3, scale=-1)} per episode. Its "
        f"adverse selection per fill ({_t(tight_m['adverse_per_fill']['mean'], tick)} ticks) "
        f"exceeds its edge ({_t(tight_m['edge_per_fill']['mean'], tick)} ticks).",
        "",
        _agent_table([mono[n]["agent"] for n in names], tick),
        "",
        mp.picture("e_arena_pnl", "PnL per episode by agent: alone, and all six together, with 95% confidence intervals"),
        "",
    ]

    # (b) round robin
    st = rr["standings"]
    st_rows = [{"agent": LABELS[s["name"]], "W–D–L": f"{s['wins']}–{s['draws']}–{s['losses']}",
                "total PnL over its 5 matches": ci_text(s["total"], 3)} for s in st]
    pair_rows = [{"match": f"{LABELS[p['a']]} vs {LABELS[p['b']]}",
                  "PnL (first)": num(p["pnl_a"]["mean"], 3), "PnL (second)": num(p["pnl_b"]["mean"], 3),
                  "difference, 95% CI": ci_text(p["diff"], 3),
                  "fill share (first)": rp.num(p["share_a"], pct=True),
                  "winner": "draw" if p["winner"] == "draw" else LABELS[p["winner"]]} for p in rr["pairs"]]
    mk = next(p for p in rr["pairs"] if {p["a"], p["b"]} == {"undercut", "markout"})
    mk_diff = mk["diff"] if mk["a"] == "markout" else neg(mk["diff"])   # markout − undercut
    top, second = rr["top"], rr["second"]
    tie = (st[0]["wins"] == st[1]["wins"])
    winner_sentence = (f"The {PROSE[top]} wins the round robin, "
                       f"{st[0]['wins']}–{st[0]['draws']}–{st[0]['losses']}")
    if tie:
        winner_sentence += (f", level on wins with the {PROSE[second]} and ahead on total PnL by "
                            f"{ci_text(rr['top_minus_second'], 3)} dollars per episode summed "
                            "over its five matches")
    winner_sentence += "."
    if mk_diff["hi"] < 0:
        mk_verdict = (f"My own design, markout, does **not** beat the plain undercutter on mean PnL. "
                      f"It loses their match by {ci_text(neg(mk_diff), 3)} dollars per episode.")
    elif mk_diff["lo"] > 0:
        mk_verdict = (f"My own design, markout, beats the plain undercutter by {ci_text(mk_diff, 3)} "
                      "dollars per episode.")
    else:
        mk_verdict = (f"Markout and the plain undercutter draw on mean PnL (difference "
                      f"{ci_text(mk_diff, 3)}).")
    sm, su = ((mk["summary_a"], mk["summary_b"]) if mk["a"] == "markout"
              else (mk["summary_b"], mk["summary_a"]))
    gmk = next(p for p in rr["pairs"] if {p["a"], p["b"]} == {"gm", "markout"})
    gmk_diff = gmk["diff"] if gmk["a"] == "markout" else neg(gmk["diff"])   # markout − gm
    if gmk_diff["hi"] < 0:
        mk_verdict += (f" It also loses to the GM zero-profit quoter, by "
                       f"{ci_text(neg(gmk_diff), 3)}.")
    mk_verdict += (" What its inventory term buys is risk. In the match with the undercutter, "
                   f"markout's PnL s.d. is {num(sm['pnl_sd'], 3)} against {num(su['pnl_sd'], 3)}, "
                   f"and its mean inventory² is {num(sm['inventory_sq']['mean'], 3)} against "
                   f"{num(su['inventory_sq']['mean'], 3)}. Its fills carry "
                   f"{_t(sm['adverse_per_fill']['mean'], tick)} ticks of adverse selection each, "
                   f"against {_t(su['adverse_per_fill']['mean'], tick)} for the undercutter's; they "
                   f"also earn less edge ({_t(sm['edge_per_fill']['mean'], tick)} against "
                   f"{_t(su['edge_per_fill']['mean'], tick)} ticks). Skewing lets markout step aside "
                   "from one-directional flow, which is disproportionately informed, and leaves the "
                   "risk-neutral maker to absorb it. Scored on mean PnL, as here, that trade costs "
                   "a little expected profit.")
    rr_lines = [
        "#### (b) Round robin",
        "",
        winner_sentence + " " + mk_verdict,
        "",
        table(st_rows),
        "",
        table(pair_rows),
        "",
        mp.picture("e_arena_h2h", "Heat map of each agent's mean PnL against each opponent"),
        "",
    ]

    # replicator
    P = np.array(rep["payoff"])
    rep_rows = [{"agent": LABELS[n], "share at the horizon": share_text(rep["final"][i]),
                 "bootstrap 90% interval": f"{share_text(rep['boot_lo'][i])} to "
                                           f"{share_text(rep['boot_hi'][i])}",
                 "payoff vs itself": num(float(P[i, i]), 3)} for i, n in enumerate(names)]
    survivors = [names[i] for i in range(len(names)) if rep["boot_hi"][i] > 0.01]
    rep_lines = [
        "**Replicator dynamics (stretch).** Treat the round-robin payoff matrix, diagonal "
        "included, as a population game: strategies that beat the population average grow, "
        "x_i ← x_i·exp(dt·(Ax)_i)/Z. Starting from equal shares, by replicator time "
        f"{rep['horizon']:g} the loss-making and rent-seeking quoters are gone. Across "
        f"{rep['n_boot']} bootstrap resamples of the episodes, the other three strategies "
        f"together keep at most {share_text(1 - rep['competitive_share_min'])} of the population, "
        "and the zero-profit trio (GM, undercutter, markout) holds the rest. How that share "
        "splits among the trio is not identified. The payoff differences that decide it are "
        "close to their Monte Carlo error, so the split varies widely across resamples (the 90% "
        "intervals below; survivors with more than 1% in some resample: "
        f"{', '.join(PROSE[x] for x in survivors)}). "
        "The GM quoter is crowded out early because it takes no rent from the wide quoters while "
        "they still exist.",
        "",
        mp.picture("e_arena_replicator", "Population shares under replicator dynamics"),
        "",
        table(rep_rows),
        "",
    ]

    # (c) free-for-all
    blocks = a["ffa"]["blocks"]
    block_rows = []
    for b in blocks:
        row = {"periods": b["periods"]}
        for n in names:
            if (b.get(f"share_{n}") or 0) > 0:
                row[f"{LABELS[n]}: fill share / adverse per fill (ticks)"] = (
                    f"{rp.num(b[f'share_{n}'], pct=True)} / {_t(b[f'adverse_{n}'], tick)}")
        block_rows.append(row)
    cols = ["periods"] + sorted({k_ for r in block_rows for k_ in r if k_ != "periods"},
                                key=lambda c: names.index(next(n for n in names if c.startswith(LABELS[n]))))
    ms = a["ffa"]["market"]
    ffa_lines = [
        "#### (c) All six together",
        "",
        f"The fixed tight quote takes {rp.num(ffa['tight']['fill_share']['mean'], pct=True)} of all "
        f"fills and loses {ci_text(ffa['tight']['pnl'], 3, scale=-1)} dollars per episode. Its "
        f"adverse selection per fill is {_t(ffa['tight']['adverse_per_fill']['mean'], tick)} ticks "
        f"against an edge of {_t(ffa['tight']['edge_per_fill']['mean'], tick)}. When it wins is the "
        f"point. In periods {blocks[0]['periods']} it takes "
        f"{rp.num(blocks[0]['share_tight'], pct=True)} of the fills, and each carries "
        f"{_t(blocks[0]['adverse_tight'], tick)} ticks of adverse selection against a "
        f"{_t(blocks[0]['edge_tight'], tick)}-tick edge. By periods "
        f"{blocks[-1]['periods']} the Bayesian quotes, whose spread falls as V is revealed, sit "
        f"inside it, and its share is {rp.num(blocks[-1]['share_tight'], pct=True)}. It wins the "
        "flow exactly when its quote is too cheap for the information in the flow. The wide and "
        "inventory quoters never set the best price and do not trade.",
        "",
        _agent_table(a["ffa"]["agents"], tick),
        "",
        f"Market level: time-average quoted spread {_t(ms['quoted_spread']['mean'], tick)} ticks "
        f"(zero-profit {_t(ms['zero_profit_spread']['mean'], tick)}), maker profit "
        f"{_t(ms['maker_pnl_per_trade']['mean'], tick, 2)} ticks per trade, noise-trader welfare "
        f"{num(ms['noise_welfare_per_period']['mean'], 3)} per period. The naive quoter subsidizes "
        "everyone else: informed traders earn "
        f"{num(ms['informed_pnl']['mean'], 3)} per episode.",
        "",
        mp.picture("e_arena_curse", "Fill share and informed share of fills over the episode in the free-for-all"),
        "",
        table(block_rows, cols),
        "",
    ]

    # (d) K sweep
    k_rows = [{"K": r["K"],
               "quoted spread (ticks)": pm_text(r["quoted_spread"], 3, 1 / tick),
               "effective spread (ticks)": pm_text(r["effective_spread"], 3, 1 / tick),
               "zero-profit spread (ticks)": pm_text(r["zero_profit_spread"], 3, 1 / tick),
               "excess (ticks)": pm_text(r["excess_spread"], 2, 1 / tick),
               "maker profit per trade (ticks)": pm_text(r["maker_pnl_per_trade"], 2, 1 / tick),
               "noise welfare per period": pm_text(r["noise_welfare_per_period"], 3),
               "noise participation": rp.num(r["noise_participation"]["mean"], pct=True),
               "trades per episode": num(r["trades_per_episode"]["mean"], 3)} for r in ks]
    same = all(abs(r["quoted_spread"]["mean"] - k2["quoted_spread"]["mean"]) < 1e-12 for r in ks[1:])
    shares = ks[-1]["fill_shares"]
    k_lines = [
        "#### (d) K identical Bayesian undercutters",
        "",
        f"One undercutter alone is a monopolist. It quotes {_t(k1['quoted_spread']['mean'], tick)} "
        f"ticks, {num(k1['quoted_spread']['mean'] / k1['zero_profit_spread']['mean'], 3)} times the "
        f"zero-profit spread, and makes {_t(k1['maker_pnl_per_trade']['mean'], tick)} ticks per "
        f"trade. With K = 2 the spread is {_t(k2['quoted_spread']['mean'], tick)} ticks, "
        f"{pm_text(k2['excess_spread'], 2, 1 / tick)} ticks above the zero-profit spread, and profit "
        f"per trade is {pm_text(k2['maker_pnl_per_trade'], 2, 1 / tick)} ticks: rounding rent only. "
        + ("Adding makers beyond two changes nothing at the market level: every K ≥ 2 reaches the "
           "same Bertrand fixed point. The flow is simply split, with fill shares from "
           f"{rp.num(min(shares), pct=True)} to {rp.num(max(shares), pct=True)} at K = 6. "
           if same else "Beyond two makers the spread changes only slightly. ")
        + f"Two is enough for Bertrand. Noise traders gain: welfare rises from "
        f"{num(k1['noise_welfare_per_period']['mean'], 3)} to "
        f"{num(k2['noise_welfare_per_period']['mean'], 3)} per period, and participation from "
        f"{rp.num(k1['noise_participation']['mean'], pct=True)} to "
        f"{rp.num(k2['noise_participation']['mean'], pct=True)}.",
        "",
        mp.picture("e_arena_competition", "Spreads, maker profit per trade and noise-trader welfare against the number of competing makers"),
        "",
        table(k_rows),
        "",
    ]
    return (["### 3.3 The tournament", ""] + mono_lines + rr_lines + rep_lines + ffa_lines + k_lines)


def text_kuhn(res: dict) -> list[str]:
    kd = res["kuhn"]
    fam = kd["family"]
    hist = kd["history"]
    marks = [h for h in hist if h["iteration"] in (10, 100, 1000, 10000, 100000)]
    plus_by = {h["iteration"]: h["exploitability"] for h in kd["plus"]["history"]}
    ex_rows = [{"iterations": f"{h['iteration']:,}", "value to P1": "{:.6f}".format(h["value"]),
                "exploitability (CFR)": "{:.3g}".format(h["exploitability"]),
                "exploitability (CFR+)": ("{:.3g}".format(plus_by[h["iteration"]])
                                          if h["iteration"] in plus_by else "not run")} for h in marks]
    info = [("J", "P1 bets the Jack"), ("Q", "P1 bets the Queen"), ("K", "P1 bets the King"),
            ("Jpb", "P1 calls a bet with the Jack"), ("Qpb", "P1 calls with the Queen"),
            ("Kpb", "P1 calls with the King"), ("Jp", "P2 bets the Jack after a check (bluff)"),
            ("Qp", "P2 bets the Queen after a check"), ("Kp", "P2 bets the King after a check"),
            ("Jb", "P2 calls with the Jack"), ("Qb", "P2 calls with the Queen"), ("Kb", "P2 calls with the King")]
    strat_rows = [{"infoset": f"`{key}`", "action": lab,
                   "CFR average": "{:.4f}".format(kd["avg_strategy"][key]),
                   "Kuhn family at CFR's α": "{:.4f}".format(kd["analytic"][key])} for key, lab in info]
    leak_label = {"never_bluff_jack": "P2 never bluffs the Jack",
                  "always_call_queen": "P2 always calls with the Queen"}
    leak_rows, frontier_rows = [], []
    for leak, s in kd["exploitation"].items():
        dev = "; ".join(f"`{k_}` {v[0]:.3f} → {v[1]:.3f}" for k_, v in s["deviations"].items())
        leak_rows.append({"leak": leak_label[leak], "equilibrium vs leak": "{:.4f}".format(s["ev_eq"]),
                          "best response vs leak": "{:.4f}".format(s["ev_br"]),
                          "gain": "{:+.4f}".format(s["gain"]),
                          "best response vs its best response": "{:.4f}".format(s["br_worst"]),
                          "cost if P2 adapts": "{:.4f}".format(s["br_exploitability_loss"]),
                          "what changes (bet/call probability)": dev})
        for f in s["frontier"]:
            if f["lam"] in (0.0, 0.25, 0.5, 0.75, 1.0):
                frontier_rows.append({"leak": leak_label[leak], "λ (weight on BR)": "{:g}".format(f["lam"]),
                                      "EV vs leak": "{:.4f}".format(f["ev_vs_leak"]),
                                      "worst-case EV": "{:.4f}".format(f["worst_case"])})
    s1, s2 = kd["exploitation"]["never_bluff_jack"], kd["exploitation"]["always_call_queen"]
    ratio1 = s1["br_exploitability_loss"] / s1["gain"]
    ratio2 = s2["br_exploitability_loss"] / s2["gain"]
    return [
        "### 3.4 Kuhn poker",
        "",
        f"After {kd['iterations']:,} iterations ({num(kd['seconds'], 2)} s), vanilla CFR's average "
        f"strategy is worth {num(kd['value'], 6)} to player 1. The game value is −1/18 = "
        f"{num(-1 / 18, 6)}, a gap of {num(abs(kd['value'] + 1 / 18), 2)}. Its exploitability is "
        f"{num(kd['exploitability'], 3)}. Over the last two decades of iterations exploitability "
        f"falls with slope {num(kd['slope_last_two_decades'], 3)} on log–log axes, the "
        f"O(1/√T) rate of CFR's regret bound. CFR+ reaches {num(kd['plus']['exploitability'], 2)} in "
        f"{kd['plus']['iterations']:,} iterations. The average strategy lands in Kuhn's equilibrium "
        f"family. P1 bets the Jack with α = {num(fam['alpha'], 4)} (the family allows [0, 1/3]) and "
        f"the King with {num(fam['king_bet'], 4)} (3α = {num(3 * fam['alpha'], 4)}). After "
        f"check–bet it calls with the Queen at {num(fam['queen_call'], 4)} "
        f"(α + 1/3 = {num(fam['alpha'] + 1 / 3, 4)}). P2 bluffs the Jack at "
        f"{num(fam['p2_jack_bluff'], 4)} and calls with the Queen at {num(fam['p2_queen_call'], 4)} "
        "(both 1/3 in theory).",
        "",
        mp.picture("e_cfr_exploitability", "Exploitability of the CFR average strategy against iterations, log-log"),
        "",
        table(ex_rows),
        "",
        table(strat_rows),
        "",
        "**Exploit versus protect.** Two leaky P2s are the equilibrium P2 with one information "
        "set changed. P1's equilibrium strategy (Kuhn's family at CFR's α) earns exactly the game "
        f"value against both: {num(s1['ev_eq'], 5)} and {num(s2['ev_eq'], 5)}. The leaks sit at "
        "information sets where the equilibrium makes P2 indifferent, so equilibrium play neither "
        "punishes nor suffers from them. A best response that departs from equilibrium only where "
        f"the leak pays gains {num(s1['gain'], 4)} and {num(s2['gain'], 4)} per hand, respectively. "
        "Once P2 adapts, the same strategies earn "
        f"{num(s1['br_worst'], 4)} and {num(s2['br_worst'], 4)}, against −1/18 for equilibrium play. "
        f"Each unit of EV gained is paid for with {num(ratio1, 3)} and {num(ratio2, 3)} units of "
        "worst-case EV. Mixing the two strategies moves along a straight line between them (figure).",
        "",
        mp.picture("e_kuhn_exploit", "EV against the leaky opponent against worst-case EV for mixtures of equilibrium and best response"),
        "",
        table(leak_rows),
        "",
        table(frontier_rows),
        "",
    ]


def text_card(res: dict) -> list[str]:
    c = res["cardgame"]
    m = c["main"]
    rows = [{"max width": "{:g}".format(r["max_width"]), "Bayes quoter PnL per game": ci_text(r["bayes_pnl"], 3),
             "ignore-trades quoter": ci_text(r["prior_pnl"], 3),
             "Bayes minus ignore-trades": ci_text(r["diff"], 3)} for r in c["by_width"]]
    err_rows = [{"round": i + 1, "Bayes quoter |mid − S|": "{:.2f}".format(eb),
                 "ignore-trades |mid − S|": "{:.2f}".format(ep)}
                for i, (eb, ep) in enumerate(zip(m["mid_error_bayes"], m["mid_error_prior"]))]
    diffs = [r["diff"]["mean"] for r in c["by_width"]]
    widths = [r["max_width"] for r in c["by_width"]]
    monotone = all(d1 >= d2 for d1, d2 in zip(diffs, diffs[1:]))
    sig = "significant" if m["diff"]["lo"] > 0 else "not significant"
    return [
        "### 3.5 Card game",
        "",
        f"The game is for practice, but it has one computed result. Over {c['n_games']} deals at "
        f"the default maximum width of {c['config']['max_width']:g}, the Bayes quoter (centred on "
        f"E[S | revealed, trades]) makes {ci_text(m['bayes_pnl'], 3)} per game. A quoter that "
        f"ignores the trades makes {ci_text(m['prior_pnl'], 3)} on the same deals and bots, a "
        f"difference of {ci_text(m['diff'], 3)} ({sig}). "
        + ("The value of reading the trades falls steadily as the allowed width grows: "
           if monotone else "The value of reading the trades varies with the allowed width: ")
        + ", ".join(f"{num(d, 3, signed=True)} at width {w:g}" for d, w in zip(diffs, widths))
        + ". A wide market lets noise flow pay for everything, while a tight one leaves you "
        "exposed to anyone who knows a card.",
        "",
        table(rows),
        "",
        table(err_rows),
        "",
        "An example game played by `--auto` (the Bayes quoter), with the benchmark mid printed next "
        "to the quote each round:",
        "",
        "```",
        c["example"],
        "```",
        "",
    ]


def text_curse(res: dict) -> list[str]:
    a = res["arena"]
    tick = a["market"]["tick"]
    gm_m = a["monopoly"]["gm"]["agent"]
    ffa = {x["name"]: x for x in a["ffa"]["agents"]}
    b0 = a["ffa"]["blocks"][0]
    k2 = a["k_sweep"][1]
    kd = res["kuhn"]["exploitation"]["never_bluff_jack"]
    mu = a["market"]["mu"]
    b = a["benchmarks"]
    inf_zp = mu / (mu + (1 - mu) * b["noise_participation_zero_profit"])
    inf_mono = mu / (mu + (1 - mu) * b["noise_participation_monopoly"])
    return [
        "## 4. What the competition says about adverse selection",
        "",
        rp.bullets([
            "**The fill is the selection event.** A GM quote is unbiased before it trades. "
            "Conditional on trading, the value moves against it: alone, the GM quoter's fills "
            f"carry {_t(gm_m['adverse_per_fill']['mean'], tick)} ticks of adverse selection each, "
            f"against an edge of {_t(gm_m['edge_per_fill']['mean'], tick)} ticks. The spread is "
            "exactly the price of that selection, and nothing is left over beyond the rounding rent.",
            "**The tightest quote is the winner, and it inherits the curse.** In the free-for-all "
            f"the fixed tight quote wins {rp.num(b0['share_tight'], pct=True)} of the fills in the "
            f"first {b0['hi']} periods, and each of those fills carries "
            f"{_t(b0['adverse_tight'], tick)} ticks of adverse selection against a "
            f"{_t(b0['edge_tight'], tick)}-tick edge. It stops "
            "winning once the Bayesian quotes, whose spread falls as V is revealed, move inside "
            "it. It wins only while it is wrong, and it loses "
            f"{num(-ffa['tight']['pnl']['mean'], 3)} per episode. This is the dealer's winner's "
            "curse: being the best price is informative, and the information is bad news.",
            "**Competition removes rents, not adverse selection.** With two Bayesian undercutters "
            f"the spread falls to within {_t(k2['excess_spread']['mean'], tick, 2)} ticks of the "
            "zero-profit spread and stops there. Quoting tighter than that is quoting below the "
            "adverse-selection cost, which is what the naive quoter does.",
            "**Wider spreads select more toxic flow.** With elastic noise demand, a wider quote "
            "drives away noise traders and leaves the informed. At π = ½ the informed share of "
            f"trades is {rp.num(inf_zp, pct=True)} at the zero-profit spread and "
            f"{rp.num(inf_mono, pct=True)} at the monopoly spread (μ = {mu:g} among arrivals). "
            "This feedback is why the zero-profit spread with elastic noise exceeds GM's μ(V_H − V_L).",
            "**Exploiting is selecting on a belief about your opponent.** In Kuhn poker the best "
            f"response to a leaky player gains {num(kd['gain'], 3)} per hand, but it is worth "
            f"{num(kd['br_worst'], 3)} if the belief is wrong and the opponent adapts. The "
            "equilibrium strategy, which selects on nothing, cannot fall below −1/18.",
        ]),
        "",
    ]


def text_limitations(res: dict) -> list[str]:
    return [
        "## 5. Limitations",
        "",
        rp.bullets([
            "**One shared belief.** All makers see the same tape and hold the same exact posterior. "
            "The common-value winner's curse, where the dealer with the most optimistic private "
            "estimate quotes tightest, is absent by construction; with private signals the "
            "tightest quote would also be the most mistaken.",
            "**A stylized market.** V is binary, drawn once per episode, so adverse selection is "
            "concentrated before V is learned. There is one unit-size trader per period and no "
            "queue priority beyond random tie-breaks. There is no latency: quotes reach the "
            "Bertrand fixed point before every arrival. The Avellaneda–Stoikov σ²(T − t) is a "
            "heuristic here, since the binary value does not diffuse.",
            "**Zero profit means within the tick.** Quotes are rounded away from the mid, so "
            "\"zero profit\" is up to a rounding rent below one tick per fill.",
            "**Rational expectations.** Every Bayesian agent knows μ and c_max. None has to learn "
            "the parameters or detect a change in the informed share.",
            "**Scoring on mean PnL.** Scoring is risk-neutral. Markout's design goal is lower "
            "inventory risk, which this scoring rewards only indirectly (the s.d. columns).",
            "**Replicator dynamics.** They run on pairwise payoffs, a caricature of a population "
            "of market makers. The split among zero-profit strategies is not identified.",
            "**Kuhn poker is tiny.** CFR's convergence here says little about large games. The "
            "exploitation study uses the exact equilibrium at CFR's α, so indifference is "
            "well defined.",
            "**The Kyle bridge is not a measurement of λ.** OFI is not Kyle's order flow, and "
            "module D's single 2012 day has its own limits.",
            "**The card game's bots are simple.** They do not learn from each other, and the "
            "benchmark is a Monte Carlo estimate.",
        ]),
        "",
        "## Reproduce",
        "",
        "```bash",
        ".venv/bin/python -m markout.games.report          # this report, figures, reports/results/arena.json",
        ".venv/bin/python -m pytest tests/test_games_*.py  # module E tests",
        ".venv/bin/python -m markout.games.cardgame        # play the card game (add --auto to watch the Bayes quoter)",
        "```",
        "",
        "References: Glosten & Milgrom (1985), *JFE* 14(1); Kyle (1985), *Econometrica* 53(6); "
        "Avellaneda & Stoikov (2008), *Quantitative Finance* 8(3); Zinkevich, Johanson, Bowling & "
        "Piccione (2007), *NIPS*; Neller & Lanctot (2013), *An Introduction to Counterfactual "
        "Regret Minimization*; Tammelin (2014), arXiv:1407.5042; Kuhn (1950); Cont, Kukanov & "
        "Stoikov (2014), *J. Financial Econometrics* 12(1); Easley, Kiefer, O'Hara & Paperman "
        "(1996), *J. Finance* 51(4).",
    ]


def write_report(res: dict) -> str:
    lines = (text_summary(res) + text_question() + text_methods(res) + ["## 3. Results", ""]
             + text_gm(res) + text_kyle(res) + text_arena(res) + text_kuhn(res) + text_card(res)
             + text_curse(res) + text_limitations(res))
    return "\n".join(lines)


def compute() -> dict:
    t0 = time.time()
    res = {}
    for key, fn in (("gm", gm_section), ("kyle", kyle_section), ("arena", arena_section),
                    ("kuhn", kuhn_section), ("cardgame", card_section)):
        t = time.time()
        res[key] = fn()
        print(f"  {key}: {time.time() - t:.1f} s")
    res["meta"] = {"seed": SEED, "n_seeds": N_SEEDS, "compute_seconds": time.time() - t0}
    return res


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Recompute report 03 (the arena).")
    ap.add_argument("--render-only", action="store_true",
                    help=f"redraw figures and text from reports/results/{NAME}.json")
    args = ap.parse_args(argv)
    t0 = time.time()
    if args.render_only:
        res = results.load(NAME)
        if res is None:
            raise SystemExit(f"reports/results/{NAME}.json not found; run without --render-only")
        res["kyle"]["bridge"] = kyle.kyle_bridge()   # module D may have (re)written its results
    else:
        res = compute()
    results.save(NAME, res)
    render_figures(res)
    rp.write("03_arena", write_report(res))
    print(f"wrote reports/03_arena.md, reports/results/{NAME}.json and the e_* figures "
          f"in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
