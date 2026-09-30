"""Report 06: `python -m markout.vol.report` (reads the saved index data; never refetches)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from markout import plotting as mp
from markout import reporting as R
from markout import results
from markout.vol import data, vrp
from markout.vol import straddle as ST
from markout.vol import strategy as S

COMMON = "2006-08-01"  # VIX3M starts in July 2006: every rule is compared on the same months
LONG = "1993-01-01"  # the HAR forecast needs three years of history first
COSTS = (0.0, 0.5, 1.0)  # vol points of strike paid per trade
SHIFTS = (0.0, -1.0, -2.0, -3.0)  # vol points below VIX at which the straddle is priced


def kelly_after_2008(pnl: pd.Series) -> dict:
    """Kelly sizes fitted on 1993-2007 outcomes, then lived through 2008 onward."""
    pre, post = pnl.loc[:"2007-12-31"].to_numpy(), pnl.loc["2008-01-01":].to_numpy()
    k_pre = S.kelly_exact(pre)
    sims = []
    for label, v in [("continuous mu/sigma^2", k_pre["v_continuous"]), ("full Kelly", k_pre["v_star"]),
                     ("half Kelly", 0.5 * k_pre["v_star"]), ("quarter Kelly", 0.25 * k_pre["v_star"])]:
        w = S.wealth_path(post, v)
        busted = bool((1 + v * post).min() <= 0)
        sims.append({"sizing": label, "vega_per_100": 100 * v, "final_wealth": 0.0 if busted else float(w[-1]),
                     "max_drawdown": -1.0 if busted else S.max_drawdown(w), "busted": busted})
    worst_post = pnl.loc["2008-01-01":]
    v_survive = 0.999 / -worst_post.min()
    return {"kelly_pre2008": k_pre, "kelly_after_2008": sims,
            "post2008_worst": {"date": str(worst_post.idxmin().date()), "pnl": float(worst_post.min())},
            "survivable_vega_per_100": 100 * v_survive,
            "survivable_share_of_pre2008_kelly": v_survive / k_pre["v_star"]}


def atm_gap(fr: pd.DataFrame) -> dict | None:
    """VIX minus the ~30-day ATM implied vol on the SPY chain saved for report 04 (one day)."""
    snap = (results.load("options") or {}).get("snapshot") or {}
    if not snap.get("available"):
        return None
    e = min(snap["expiries"], key=lambda e: abs(e["days"] - 30))
    day = pd.Timestamp(snap["meta"]["quote_time_et"][:10])
    if day not in fr.index:
        return None
    vix = float(fr.loc[day, "vix"])
    return {"date": str(day.date()), "days": int(e["days"]), "atm_iv": 100 * e["smile_atm_iv"], "vix": vix,
            "gap": vix - 100 * e["smile_atm_iv"]}


def straddle_analysis(fr: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """The same premium collected with a delta-hedged ATM straddle: same months, same implied
    vol, same 0.5 vol point cost, plus 1 bp per unit of index traded for the hedge."""
    res = ST.run(fr, LONG)
    inst = {}
    for col in ("var_swap", "straddle"):
        inst[col] = {**S.stats(ST.as_strategy(res, col)), "kelly": S.kelly_exact(res[col].to_numpy()),
                     **kelly_after_2008(res[col])}
    g, attr = res["gross"], res["attribution"]

    def month(d, r) -> dict:
        return {"date": str(d.date()), "vix": float(r.vix), "realized": float(r.rv_vol), "eff_vol": float(r.eff_vol),
                "exposure": float(r.exposure), "var_swap": float(r.var_swap), "straddle": float(r.straddle)}

    worse = res[res["straddle"] < res["var_swap"]]  # months the straddle did worse
    rel = worse["straddle"].idxmin() if len(worse) else None
    sim = ((results.load("options") or {}).get("hedging") or {}).get("attribution") or []
    shifts = []
    for sh in SHIFTS:
        rr = res if sh == 0 else ST.run(fr, LONG, vol_shift=sh)
        st = S.stats(ST.as_strategy(rr, "straddle"))
        shifts.append({"shift": sh, "sharpe_ann": st["sharpe_ann"], "mean_pnl": st["mean_pnl"],
                       "worst_pnl": st["worst_pnl"], "v_star": S.kelly_exact(rr["straddle"].to_numpy())["v_star"]})
    out = {"period": [str(res.index.min().date()), str(res.index.max().date())], "rolls": int(len(res)),
           "instruments": inst, "corr": float(np.corrcoef(res["var_swap"], res["straddle"])[0, 1]),
           "attribution_r2": float(1 - np.var(g - attr) / np.var(g)),
           "sim_daily_attribution_r2": next((a["attr_r2"] for a in sim if a["label"] == "1 day"), None),
           "mean_hedge_cost": float(res["hedge_cost"].mean()),
           "worst": [month(d, r) for d, r in res.nsmallest(5, "var_swap").iterrows()],
           "straddle_worse": month(rel, res.loc[rel]) if rel is not None else None,
           "shifts": shifts, "atm_gap": atm_gap(fr)}
    return out, res


def analyse(fr: pd.DataFrame) -> dict:
    from markout.evaluation.bootstrap import sharpe_ci
    from markout.evaluation.dsr import deflated_sharpe

    valid = fr.dropna(subset=["rv"])
    prem = valid["vix"] - valid["rv_vol"]
    out = {"period": [str(valid.index.min().date()), str(valid.index.max().date())],
           "days": int(len(valid)), "mean_premium_vol": float(prem.mean()),
           "median_premium_vol": float(prem.median()), "share_implied_above": float((prem > 0).mean()),
           "corr_vix_rv": float(np.corrcoef(valid["vix"], valid["rv_vol"])[0, 1])}
    runs, rows = {}, []
    for rule in S.RULES:
        res = S.run(fr, rule, COMMON)
        runs[rule.name] = res
        st = S.stats(res)
        ci = sharpe_ci(res["pnl"].to_numpy(), level=0.95, n_boot=5000, seed=0)
        rows.append({"rule": rule.name, **st, "sharpe_lo": ci["lo"] * np.sqrt(S.ROLLS_PER_YEAR),
                     "sharpe_hi": ci["hi"] * np.sqrt(S.ROLLS_PER_YEAR)})
    out["rules"] = rows
    best = max(rows, key=lambda r: r["sharpe_ann"])
    trial_srs = np.array([r["mean_pnl"] / r["sd_pnl"] for r in rows])
    out["best_rule"] = best["rule"]
    out["dsr"] = deflated_sharpe(runs[best["rule"]]["pnl"].to_numpy(), trial_srs=trial_srs, n_trials=len(rows))
    out["costs"] = [{"cost_vol": c, **{r.name: S.stats(S.run(fr, r, COMMON, cost_vol=c))["sharpe_ann"]
                                       for r in (S.RULES[0], S.RULES[1], S.RULES[4])}} for c in COSTS]
    worst = runs["always"].nsmallest(5, "pnl")
    out["worst_months"] = [{"date": str(d.date()), "vix": float(r.vix), "realized": float(r.rv_vol), "pnl": float(r.pnl),
                            "traded_by_best": bool(runs[best["rule"]].loc[d, "traded"])} for d, r in worst.iterrows()]
    # sizing: exact Kelly on the long history, then the same rule under estimation error
    long_res = S.run(fr, S.RULES[0], LONG)
    out["kelly_full"] = S.kelly_exact(long_res["pnl"].to_numpy())
    out.update(kelly_after_2008(long_res["pnl"]))
    out["long_period"] = [str(long_res.index.min().date()), str(long_res.index.max().date())]
    return out, runs, long_res


def figures(fr: pd.DataFrame, a: dict, runs: dict, long_res: pd.DataFrame, sres: pd.DataFrame) -> None:
    m = fr.dropna(subset=["rv"]).resample("ME").last()

    def vix_rv():
        fig, ax = mp.subplots(h=3.4)
        ax.plot(m.index, m["vix"], color=mp.series(0), label="VIX (implied, next 30 days)")
        ax.plot(m.index, m["rv_vol"], color=mp.series(1), label="S&P 500 realized vol, next 21 days")
        ax.set_ylabel("annualized vol, points")
        mp.legend(ax, loc="upper left")
        mp.title(ax, "Implied volatility usually overstates what follows", "Month-end values, 1990 onward")
        return fig

    mp.render("h_vix_vs_realized", vix_rv)

    def hist():
        fig, ax = mp.subplots(w=6.4, h=3.4)
        p = runs["always"]["pnl"].to_numpy()
        ax.hist(p, bins=60, color=mp.series(0), edgecolor=mp.surface(), linewidth=0.6)
        ax.axvline(0, color=mp.baseline_color(), linewidth=0.9)
        ax.set_xlabel("PnL per $1 of vega sold, one month")
        ax.set_ylabel("months")
        mp.title(ax, "Small gains, rare large losses", f"Selling variance every month since {COMMON[:4]}")
        return fig

    mp.render("h_pnl_hist", hist)

    def cum():
        fig, ax = mp.subplots(h=3.6)
        for i, name in enumerate(["always", "contango", "har + contango"]):
            res = runs[name]
            ax.plot(res.index, res["pnl"].cumsum(), color=mp.series(i), label=name)
        mp.zero_line(ax)
        ax.set_ylabel("cumulative PnL per $1 of vega")
        mp.legend(ax, loc="upper left")
        mp.title(ax, "Selling one month of S&P variance at a time", "0.5 vol point cost per trade; rules use "
                 "only what is known at each roll")
        return fig

    mp.render("h_cumulative", cum)

    post = long_res.loc["2008-01-01":, "pnl"].to_numpy()
    dates = long_res.loc["2008-01-01":].index

    def kelly():
        fig, ax = mp.subplots(h=3.6)
        for i, s in enumerate(a["kelly_after_2008"][1:]):
            v = s["vega_per_100"] / 100
            w = S.wealth_path(post, v)
            ax.plot(dates, np.maximum(w, 1e-3), color=mp.series(i), label=f"{s['sizing']} ({s['vega_per_100']:.2f})")
        ax.set_yscale("log")
        ax.set_ylabel("wealth (start = 1, log scale)")
        ax.set_ylim(5e-4, None)
        ax.annotate("wiped out", xy=(0.99, 0.02), xycoords="axes fraction", ha="right", va="bottom",
                    fontsize=8.5, color=mp.ink("secondary"))
        mp.legend(ax, loc="center right", title=r"sizing fitted on 1993–2007 (\$ vega per \$100)", title_fontsize=8.5)
        mp.title(ax, "Kelly sizing chosen before 2008, lived through it", "Always-sell rule, 0.5 vol point cost")
        return fig

    mp.render("h_kelly", kelly)

    def straddle():
        fig, ax = mp.subplots(w=6.4, h=4.6)
        x, y = sres["var_swap"], sres["straddle"]
        lo, hi = min(x.min(), y.min()) - 4, max(x.max(), y.max()) + 4
        ax.plot([lo, hi], [lo, hi], color=mp.ink("muted"), linewidth=1.0, linestyle="--", label="same PnL")
        ax.scatter(x, y, s=14, color=mp.series(0), alpha=0.65, linewidths=0, label="one month")
        worse = sres[sres["straddle"] < sres["var_swap"]]  # the straddle's worst month relative to the swap
        marked = list(sres.nsmallest(4, "var_swap").index) + ([worse["straddle"].idxmin()] if len(worse) else [])
        for d in marked:
            r = sres.loc[d]
            ax.annotate(str(d.date()), xy=(r.var_swap, r.straddle), xytext=(7, 0), textcoords="offset points",
                        va="center", fontsize=8.5, color=mp.ink("secondary"))
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(r"short variance swap: PnL per \$1 of vega, one month")
        ax.set_ylabel("short straddle, delta-hedged daily")
        mp.legend(ax, loc="upper left")
        mp.title(ax, "Same months, same implied vol: the straddle's crashes are milder",
                 f"{sres.index.min().year}–{sres.index.max().year}, both priced at VIX, 0.5 vol point cost; "
                 "labels are roll dates")
        return fig

    mp.render("h_straddle", straddle)


def straddle_section(sd: dict) -> list[str]:
    n = R.num
    vs, st = sd["instruments"]["var_swap"], sd["instruments"]["straddle"]
    w0, sw, gap = sd["worst"][0], sd["straddle_worse"], sd["atm_gap"]
    rows = [{"instrument": label, "Sharpe": s["sharpe_ann"], "skew": s["skew"], "worst month": s["worst_pnl"],
             "worst 5% (mean)": s["cvar5"], "max drawdown": s["max_drawdown"], "Kelly": 100 * s["kelly"]["v_star"],
             "ruin at": 100 * s["kelly"]["v_max"], "mu/sigma^2": 100 * s["kelly"]["v_continuous"]}
            for label, s in (("short variance swap", vs), ("short straddle, delta-hedged", st))]
    fractions = st["kelly_after_2008"][1:]  # full, half, quarter
    alive = [s for s in fractions if not s["busted"]]
    vs_alive = [s for s in vs["kelly_after_2008"][1:] if not s["busted"]]
    if alive:
        busted = [s["sizing"] for s in fractions if s["busted"]]
        survival = (f"Fitted on 1993–2007 alone, the straddle's {alive[0]['sizing']} lives through 2008 onward, with a "
                    f"{-100 * alive[0]['max_drawdown']:.0f}% drawdown on the way"
                    + (f"; {' and '.join(busted)} are wiped out" if busted else "")
                    + ("." if vs_alive else ", as is every fraction of the variance swap's."))
    else:
        survival = ("Fitted on 1993–2007 alone, every fraction down to a quarter Kelly is still wiped out after 2008, "
                    "for the straddle as for the variance swap.")
    if w0["eff_vol"] < w0["realized"]:
        why = f"because the big days came after the index had left the strike (gamma-weighted vol {n(w0['eff_vol'])})"
    else:
        why = f"even though its gamma-weighted vol was {n(w0['eff_vol'])}"
    worse = ""
    if sw and sw["straddle"] < 0:
        worse = (f" The straddle is not always the safer side: from {sw['date']} the big days came while its gamma was "
                 f"high (gamma-weighted vol {n(sw['eff_vol'])} against {n(sw['realized'])} realized), and it lost "
                 f"{n(-sw['straddle'])} to the variance swap's {n(-sw['var_swap'])}.")
    sim = sd["sim_daily_attribution_r2"]
    attribution = (f"On these real paths the gamma–theta attribution explains {n(100 * sd['attribution_r2'])}% of the "
                   "variance of the straddle's monthly PnL"
                   + (f", against {n(100 * sim)}% for the same daily hedge on the lognormal paths of report 04. The "
                      "gap comes from large daily moves, which real prices have far more of." if sim is not None
                      else "."))
    below = next((s for s in sd["shifts"] if s["sharpe_ann"] < vs["sharpe_ann"]), None)
    price = ("The catch is the price. VIX averages the whole smile, including the expensive downside puts, so an "
             "at-the-money straddle trades below it"
             + (f": on the SPY chain saved for report 04 ({gap['date']}), {gap['days']}-day at-the-money implied vol "
                f"was {n(gap['atm_iv'])}% against a VIX close of {n(gap['vix'])}, {n(gap['gap'])} points lower."
                if gap else ".")
             + " Pricing the straddle below VIX leaves its tail about where it was and takes the premium away:")
    after = (f"By a {-below['shift']:g}-point discount its Sharpe ({n(below['sharpe_ann'])}) is below the variance "
             f"swap's ({n(vs['sharpe_ann'])}) over the same months. What the straddle buys is a milder worst month, not "
             "a bigger premium." if below else
             "Even at the largest discount tried its Sharpe stays above the variance swap's over the same months.")
    return [
        "## Collecting the premium with options instead", "",
        "Variance swaps trade over the counter between institutions. Most traders collect the premium by selling "
        f"listed options and delta-hedging them, so the same {sd['rolls']} months ({sd['period'][0][:4]}–"
        f"{sd['period'][1][:4]}) were also run through a short at-the-money straddle. It is priced at the same "
        "implied vol (VIX), hedged at every "
        "close with Black–Scholes deltas at that vol, and charged the same 0.5 vol point cost plus 1 bp per unit of "
        f"index traded for the hedge ({n(sd['mean_hedge_cost'])} vol points a month on average). Dividing by the "
        "straddle's vega at the roll puts both in PnL per $1 of vega.", "",
        "A hedged option is still a bet on realized variance, but each day counts in proportion to the option's "
        "dollar gamma, which peaks near the strike and collapses once the index runs away from it; the variance "
        "swap counts every day the same. " + attribution, "",
        R.table(rows, formats={"Sharpe": "{:.2f}", "skew": "{:.2f}", "worst month": "{:.1f}",
                               "worst 5% (mean)": "{:.1f}", "max drawdown": "{:.1f}", "Kelly": "{:.2f}",
                               "ruin at": "{:.2f}", "mu/sigma^2": "{:.2f}"}), "",
        "PnL is per $1 of vega a month. Kelly (exact), ruin and mu/sigma^2 are dollars of vega per $100 of capital; "
        "ruin is the size at which the worst month takes the whole account.", "",
        mp.picture("h_straddle", "Monthly PnL of the delta-hedged straddle against the variance swap"), "",
        f"The two move together (correlation {n(sd['corr'])}) until the index runs. The five worst months for the "
        "variance swap:", "",
        R.table([{"roll date": w["date"], "VIX": w["vix"], "realized vol": w["realized"],
                  "gamma-weighted vol": w["eff_vol"], "gamma exposure": w["exposure"],
                  "variance swap": w["var_swap"], "straddle": w["straddle"]} for w in sd["worst"]],
                formats={"VIX": "{:.1f}", "realized vol": "{:.1f}", "gamma-weighted vol": "{:.1f}",
                         "gamma exposure": "{:.2f}", "variance swap": "{:.1f}", "straddle": "{:.1f}"}), "",
        "Gamma-weighted vol weights each day's move by the straddle's dollar gamma that day. Gamma exposure is the "
        "month's total dollar gamma relative to the average for a path that moves at the implied vol: about 2 if "
        "the index sat on the strike all month, below 1 if it spent more of the month far from the strike.", "",
        f"In the month from {w0['date']}, VIX was {n(w0['vix'])} and the index then realized {n(w0['realized'])}. The "
        f"variance swap lost {n(-w0['var_swap'])} and the straddle {n(-w0['straddle'])}, {why}.{worse}", "",
        f"The milder tail raises the growth-optimal size {n(st['kelly']['v_star'] / vs['kelly']['v_star'])}x, to "
        f"{n(100 * st['kelly']['v_star'])} dollars of vega per $100 against {n(100 * vs['kelly']['v_star'])}. "
        + survival, "",
        price, "",
        R.table([{"priced at": "VIX" if s["shift"] == 0 else f"VIX {s['shift']:+g}", "Sharpe": s["sharpe_ann"],
                  "mean PnL": s["mean_pnl"], "worst month": s["worst_pnl"], "Kelly": 100 * s["v_star"]}
                 for s in sd["shifts"]],
                formats={"Sharpe": "{:.2f}", "mean PnL": "{:.2f}", "worst month": "{:.1f}", "Kelly": "{:.2f}"}), "",
        after, "",
    ]


def write(a: dict) -> str:
    n, P = R.num, R.prob
    rules = {r["rule"]: r for r in a["rules"]}
    al, best = rules["always"], rules[a["best_rule"]]
    kf, kp = a["kelly_full"], a["kelly_pre2008"]
    dsr = a["dsr"]
    sd = a["straddle"]
    vs, st = sd["instruments"]["var_swap"], sd["instruments"]["straddle"]
    shift2 = next(s for s in sd["shifts"] if s["shift"] == -2.0)
    below = next((s for s in sd["shifts"] if s["sharpe_ann"] < vs["sharpe_ann"]), None)
    lines = [
        "# 06 · The volatility risk premium: when does selling S&P volatility pay?", "",
        "*Generated by `python -m markout.vol.report` from saved Yahoo Finance data (S&P 500, VIX and VIX3M "
        "daily closes); the data itself is not committed.*", "",
        "**Question.** Options on the S&P 500 are usually priced for more movement than actually follows. How "
        "large is that premium, how often does collecting it go badly wrong, and does timing or sizing make it "
        "safe enough to trade?", "",
        f"**Answer.** From {a['period'][0][:4]} to {a['period'][1][:4]}, VIX exceeded the volatility the S&P then "
        f"realized over the next 21 trading days {n(100 * a['share_implied_above'])}% of the time, by "
        f"{n(a['mean_premium_vol'])} vol points on average (median {n(a['median_premium_vol'])}). Selling one month "
        f"of variance every month since {COMMON[:4]} earned an annualized Sharpe of {n(al['sharpe_ann'])} with a "
        f"skew of {n(al['skew'])}: its worst month ({al['worst_date']}) lost {n(-al['worst_pnl'])} per $1 of vega, "
        f"against an average gain of {n(al['mean_pnl'])}. The best of {len(a['rules'])} timing rules, "
        f"*{a['best_rule']}*, raises the Sharpe to {n(best['sharpe_ann'])} (95% CI {n(best['sharpe_lo'])} to "
        f"{n(best['sharpe_hi'])}) and cuts the maximum drawdown from {n(-al['max_drawdown'])} to "
        f"{n(-best['max_drawdown'])}"
        + (f", and it survives the deflation for the {len(a['rules'])} rules tried (Deflated Sharpe {P(dsr['dsr'])})."
           if dsr["dsr"] >= 0.95 else f", but its Deflated Sharpe over the {len(a['rules'])} rules tried is "
           f"{P(dsr['dsr'])}.")
        + f" Sizing matters more than timing: the textbook Kelly fraction mu/sigma^2 asks for "
        f"{n(kf['v_continuous'] / kf['v_star'])}x the growth-optimal size on these fat-tailed outcomes, "
        f"{n(kf['v_continuous'] / kf['v_max'])}x the size at which the worst month wipes out the account. "
        f"Collected with a delta-hedged straddle instead, the worst month of {sd['period'][0][:4]}–"
        f"{sd['period'][1][:4]} loses {n(-st['worst_pnl'])} rather than {n(-vs['worst_pnl'])} per $1 of vega, because "
        "the straddle's exposure fades once the index leaves the strike. But at-the-money options trade below VIX, "
        f"and at a 2-point discount the straddle's Sharpe ({n(shift2['sharpe_ann'])}) is "
        f"{'below' if shift2['sharpe_ann'] < vs['sharpe_ann'] else 'still above'} the variance swap's over the same "
        f"months ({n(vs['sharpe_ann'])}).", "",
        "## Data and method", "",
        R.bullets([
            "Daily closes of the S&P 500 index, VIX (implied volatility of 30-day S&P options, from 1990) and "
            "VIX3M (93-day, from July 2006), from Yahoo Finance.",
            "Implied variance is VIX squared: VIX is built by replicating a 30-day variance swap, so VIX^2 is that "
            "swap's fair strike. Realized variance is the annualized sum of squared daily log returns over the next "
            "21 trading days.",
            "The strategy sells one month of variance every 21 trading days. Per $1 of vega notional, a short "
            "variance swap struck at K that realizes R pays (K^2 - R^2) / (2K): at most K/2, unbounded below. A "
            "cost of 0.5 vol points is charged on every trade (varied below).",
            "Timing rules use only what is known at the roll: the VIX/VIX3M term structure (sell only in "
            "contango), and a HAR forecast of realized variance (Corsi 2009), refitted every month on samples whose "
            "21-day window had already ended.",
            f"Every rule is compared over the same {al['rolls']} monthly rolls from {COMMON[:7]}.",
        ]), "",
        mp.picture("h_vix_vs_realized", "VIX against the realized volatility that followed"), "",
        "## Results", "",
        R.table([{"rule": r["rule"], "traded": f"{r['traded']}/{r['rolls']}", "Sharpe": r["sharpe_ann"],
                  "95% CI": f"{n(r['sharpe_lo'])} to {n(r['sharpe_hi'])}", "skew": r["skew"],
                  "worst month": r["worst_pnl"], "max drawdown": r["max_drawdown"], "total": r["total"]}
                 for r in a["rules"]],
                formats={"Sharpe": "{:.2f}", "skew": "{:.2f}", "worst month": "{:.1f}", "max drawdown": "{:.1f}",
                         "total": "{:.1f}"}), "",
        "PnL is per $1 of vega sold each month; annualized Sharpe uses 12 rolls a year.", "",
        mp.picture("h_cumulative", "Cumulative PnL of the timing rules"), "",
        mp.picture("h_pnl_hist", "Distribution of monthly PnL"), "",
        "**The months that hurt.** The five worst months for the always-sell rule, and whether the best rule was "
        "in the market:", "",
        R.table([{"roll date": w["date"], "VIX at roll": w["vix"], "realized vol": w["realized"], "PnL": w["pnl"],
                  f"{a['best_rule']} traded": "yes" if w["traded_by_best"] else "no"} for w in a["worst_months"]],
                formats={"VIX at roll": "{:.1f}", "realized vol": "{:.1f}", "PnL": "{:.1f}"}), "",
        "**Costs.** Annualized Sharpe by trading cost (vol points of strike per trade):", "",
        R.table(a["costs"], formats={k: "{:.2f}" for k in a["costs"][0] if k != "cost_vol"} | {"cost_vol": "{:g}"},
                headers={"cost_vol": "cost"}), "",
        "**Selection.** " + f"Five rules were tried; the best one's Sharpe is deflated against the best of five "
        f"tries by luck (SR* = {n(dsr['sr_star'] * np.sqrt(S.ROLLS_PER_YEAR))} annualized). Deflated Sharpe "
        f"{P(dsr['dsr'])}, from {dsr['T']} monthly observations with skew {n(dsr['skew'])} and kurtosis "
        f"{n(dsr['kurt'])}: the negative skew is part of why the deflation is severe.", "",
        "## Sizing: Kelly with a fat left tail", "",
        f"On {a['long_period'][0][:4]}–{a['long_period'][1][:4]} monthly outcomes of the always-sell rule, the growth-optimal "
        f"size is {n(100 * kf['v_star'])} dollars of vega per $100 of capital, computed exactly by maximizing the "
        f"mean log growth. The worst month caps any survivable size at {n(100 * kf['v_max'])}. The familiar "
        f"formula mu/sigma^2, which assumes small, roughly normal outcomes, says {n(100 * kf['v_continuous'])}: "
        "far past the point of ruin.", "",
        f"Estimation risk makes it worse. Fitted on 1993–2007 alone, full Kelly is {n(100 * kp['v_star'])} per $100: "
        "that history had no month like the ones to come. "
        + (f"Every fraction of it down to a quarter is wiped out after 2008. " if all(x["busted"] for x in
           a["kelly_after_2008"]) else "")
        + f"The worst month after 2007 ({a['post2008_worst']['date']}) lost {n(-a['post2008_worst']['pnl'])} per $1 of "
        f"vega, so only sizes below {n(a['survivable_vega_per_100'])} per $100, "
        f"{n(100 * a['survivable_share_of_pre2008_kelly'])}% of the pre-2008 Kelly size, would have survived it.", "",
        R.table([{"sizing (fitted on 1993–2007)": s["sizing"], "$ vega per $100": s["vega_per_100"],
                  "final wealth 2008–": "bust" if s["busted"] else s["final_wealth"],
                  "max drawdown": "-100%" if s["busted"] else f"{100 * s['max_drawdown']:.0f}%"}
                 for s in a["kelly_after_2008"]], formats={"$ vega per $100": "{:.2f}", "final wealth 2008–": "{:.2f}"}),
        "", mp.picture("h_kelly", "Wealth paths under Kelly fractions chosen before 2008"), "",
        *straddle_section(sd),
        "## What this says", "",
        R.bullets([
            "The premium is persistent and large, which is why selling index options is a real business. It is "
            "paid for carrying a crash, and the crashes are the whole story of the risk.",
            "Timing helps mainly by being out of the market when the term structure inverts, which is exactly when "
            "the next month is most dangerous; it does not remove the left tail (the worst months still arrive "
            "from calm starts).",
            "Kelly sizing on a payoff like this has to use the actual distribution, not mean and variance, and it "
            "still depends on the worst month you have seen. Fractional Kelly is the price of not knowing the tail.",
            "The instrument shapes the tail. A delta-hedged straddle's exposure fades as the index runs away from "
            f"the strike, which cuts the worst month to {n(100 * st['worst_pnl'] / vs['worst_pnl'])}% of the "
            f"variance swap's and raises the size Kelly allows {n(st['kelly']['v_star'] / vs['kelly']['v_star'])}x. "
            "But at-the-money options trade below VIX"
            + (f", and a {-below['shift']:g}-point discount uses up the straddle's Sharpe advantage." if below
               else ", which eats into the straddle's premium."),
        ]), "",
        "## Limitations", "",
        R.bullets([
            "VIX^2 is close to, not equal to, a tradable variance-swap strike: real strikes include a convexity and "
            "skew adjustment and a dealer margin, which the cost line only approximates.",
            "Realized variance uses daily closes (no intraday data), and index dividends are ignored (they barely "
            "move variance).",
            "Trading this means SPX options or VIX futures, with their own margin, liquidity and roll costs; the "
            "study measures the premium, not an executable strategy.",
            "Five rules were tried on the same history; the Deflated Sharpe above accounts for that, not for the "
            "wider literature this design draws on.",
            "The straddle is priced at VIX (or a fixed discount to it) and hedged once a day at the close with "
            "Black–Scholes deltas, with r = q = 0. A real book would hedge futures during the day, and option spreads "
            "widen in exactly the months that hurt.",
        ]), "",
        "## Reproduce", "", "```bash", "python -m markout.vol.data     # download once", "python -m markout.vol.report",
        "```",
    ]
    return "\n".join(lines)


def main() -> None:
    if not data.available():
        raise SystemExit("index data missing: run `python -m markout.vol.data` first")
    fr = vrp.frame(data.load())
    a, runs, long_res = analyse(fr)
    a["straddle"], sres = straddle_analysis(fr)
    figures(fr, a, runs, long_res, sres)
    results.save("vol_premium", a)
    R.write("06_vol_premium", write(a))
    print("wrote reports/06_vol_premium.md")


if __name__ == "__main__":
    main()
