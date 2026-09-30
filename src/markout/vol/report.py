"""Report 06: `python -m markout.vol.report` (reads the saved index data; never refetches)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from markout import plotting as mp
from markout import reporting as R
from markout import results
from markout.vol import data, vrp
from markout.vol import strategy as S

COMMON = "2006-08-01"  # VIX3M starts in July 2006: every rule is compared on the same months
LONG = "1993-01-01"  # the HAR forecast needs three years of history first
COSTS = (0.0, 0.5, 1.0)  # vol points of strike paid per trade


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
    pre, post = long_res.loc[:"2007-12-31", "pnl"].to_numpy(), long_res.loc["2008-01-01":, "pnl"].to_numpy()
    k_pre = S.kelly_exact(pre)
    out["kelly_pre2008"] = k_pre
    sims = []
    for label, v in [("continuous mu/sigma^2", k_pre["v_continuous"]), ("full Kelly", k_pre["v_star"]),
                     ("half Kelly", 0.5 * k_pre["v_star"]), ("quarter Kelly", 0.25 * k_pre["v_star"])]:
        w = S.wealth_path(post, v)
        busted = bool((1 + v * post).min() <= 0)
        sims.append({"sizing": label, "vega_per_100": 100 * v, "final_wealth": 0.0 if busted else float(w[-1]),
                     "max_drawdown": -1.0 if busted else S.max_drawdown(w), "busted": busted})
    out["kelly_after_2008"] = sims
    worst_post = long_res.loc["2008-01-01":, "pnl"]
    out["post2008_worst"] = {"date": str(worst_post.idxmin().date()), "pnl": float(worst_post.min())}
    v_survive = 0.999 / -worst_post.min()
    out["survivable_vega_per_100"] = 100 * v_survive
    out["survivable_share_of_pre2008_kelly"] = v_survive / k_pre["v_star"]
    out["long_period"] = [str(long_res.index.min().date()), str(long_res.index.max().date())]
    return out, runs, long_res


def figures(fr: pd.DataFrame, a: dict, runs: dict, long_res: pd.DataFrame) -> None:
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


def write(a: dict) -> str:
    n, P = R.num, R.prob
    rules = {r["rule"]: r for r in a["rules"]}
    al, best = rules["always"], rules[a["best_rule"]]
    kf, kp = a["kelly_full"], a["kelly_pre2008"]
    cont = next(s for s in a["kelly_after_2008"] if s["sizing"].startswith("continuous"))
    half = next(s for s in a["kelly_after_2008"] if s["sizing"] == "half Kelly")
    full = next(s for s in a["kelly_after_2008"] if s["sizing"] == "full Kelly")
    dsr = a["dsr"]
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
        f"{n(kf['v_continuous'] / kf['v_max'])}x the size at which the worst month wipes out the account.", "",
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
        "## What this says", "",
        R.bullets([
            "The premium is persistent and large, which is why selling index options is a real business. It is "
            "paid for carrying a crash, and the crashes are the whole story of the risk.",
            "Timing helps mainly by being out of the market when the term structure inverts, which is exactly when "
            "the next month is most dangerous; it does not remove the left tail (the worst months still arrive "
            "from calm starts).",
            "Kelly sizing on a payoff like this has to use the actual distribution, not mean and variance, and it "
            "still depends on the worst month you have seen. Fractional Kelly is the price of not knowing the tail.",
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
    figures(fr, a, runs, long_res)
    results.save("vol_premium", a)
    R.write("06_vol_premium", write(a))
    print("wrote reports/06_vol_premium.md")


if __name__ == "__main__":
    main()
