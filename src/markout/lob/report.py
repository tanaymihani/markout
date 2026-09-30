"""Report 02, the microstructure lab: `python -m markout.lob.report`.

Recomputes everything from the Parquet files (the pipeline is deterministic; the
only randomness, the block bootstrap, uses fixed seeds), saves
reports/results/microstructure.json, renders the d_* figures and writes
reports/02_microstructure.md. Every number in the report is interpolated from the
computed results; prose that depends on a result is chosen from it here.
"""

from __future__ import annotations

import argparse
import gc
import time
from dataclasses import asdict
from typing import Callable

import numpy as np

from markout import plotting as mp
from markout import reporting as rp
from markout import results
from markout.lob import figures as fg
from markout.lob.facts import LARGE_TICK_THRESHOLD, compute_facts
from markout.lob.fills import END_NAMES, MODELS, cpp_available
from markout.lob.imbalance import STEP, fit_imbalance, make_samples
from markout.lob.lobster import (
    DATE,
    LEVELS,
    SESSION,
    SPLIT,
    TICK,
    TICKERS,
    TRIM,
    available,
    check_consistency,
    load_day,
)
from markout.lob.markouts import (
    EVERY,
    HORIZONS,
    MAX_WAIT,
    SIZE,
    SPREAD_HORIZONS,
    by_queue_bucket,
    order_markouts,
    sample_orders,
    spread_decomposition,
    summarize_markouts,
)
from markout.lob.ofi import (
    BUCKET,
    WINDOW,
    beta_depth_slope,
    contemporaneous_vs_lagged,
    full_day_beta,
    make_buckets,
    window_regressions,
)
from markout.lob.postcross import (
    H,
    SIGNAL_LABELS,
    TAKER_FEE_TICKS,
    TAKER_FEE_USD,
    cell_table,
    decisions,
    evaluate_rule,
    grid,
)

NAME = "microstructure"
DATA_SOURCE = ("LOBSTER sample files (Nasdaq TotalView-ITCH), downloaded by "
               "scripts/download_lobster.py from a hash-pinned Hugging Face mirror "
               "(totalorganfailure/lobster-data) because the official lobsterdata.com links no "
               "longer serve the zips; the consistency replay below is the authenticity check")


def clock(t: float) -> str:
    s = int(round(t))
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}"


# --------------------------------------------------------------------------- compute


def analyze(ticker: str, engine: str) -> tuple[dict, dict]:
    """Every per-ticker result, plus arrays needed only for figures."""
    day = load_day(ticker)
    cons = check_consistency(day.messages, day.book)
    facts = compute_facts(day)

    bk = make_buckets(day)
    windows = window_regressions(bk)
    worst = min(windows, key=lambda w: w["r2"])
    in_worst = (bk.start >= worst["start"]) & (bk.start < worst["start"] + WINDOW)
    beta, r2_day = full_day_beta(bk)
    cvl = contemporaneous_vs_lagged(bk)
    imb = fit_imbalance(make_samples(day))

    mk = order_markouts(day, sample_orders(day), engine=engine)
    summary = summarize_markouts(mk)
    ends = {m: {END_NAMES[int(k)]: int(v) for k, v in
                zip(*np.unique(mk.loc[mk["model"] == m, "end_reason"], return_counts=True))}
            for m in MODELS}
    partial = {m: int(((mk["model"] == m) & (mk["filled_qty"] > 0) & (mk["filled_qty"] < SIZE)).sum())
               for m in MODELS}

    pc = decisions(day, engine=engine)
    post_vs_cross = {
        "n_decisions": int((pc["model"] == MODELS[0]).sum()),
        "grid": {m: grid(pc[pc["model"] == m]).tolist() for m in MODELS},
        "cells": {m: cell_table(pc[pc["model"] == m]) for m in MODELS},
        "rules": {m: asdict(evaluate_rule(pc, m, seed=k)) for k, m in enumerate(MODELS)},
        "means": {m: {"p_fill": float((g["fill_frac"] > 0).mean()),
                      "ev_post": float(g["ev_post"].mean()),
                      "ev_cross": float(g["ev_cross"].mean()),
                      "ev_post_bps": float((g["ev_post"] * g["tick_bps"]).mean()),
                      "ev_cross_bps": float((g["ev_cross"] * g["tick_bps"]).mean()),
                      "move": float(g["move"].mean()),
                      "spread": float(g["spread"].mean())}
                  for m, g in pc.groupby("model", sort=False)},
    }
    out = {
        "consistency": {"n_messages": cons.n_messages, "n_eligible": cons.n_eligible,
                        "n_checked": cons.n_checked, "n_pass": cons.n_pass,
                        "pass_rate": cons.pass_rate, "coverage": cons.coverage,
                        "by_type": {str(k): list(v) for k, v in cons.by_type.items()},
                        "n_failures": int(len(cons.failures))},
        "facts": facts.to_dict(),
        "cks": {"windows": windows,
                "mean_r2": float(np.mean([w["r2"] for w in windows])),
                "median_r2": float(np.median([w["r2"] for w in windows])),
                "min_r2": worst["r2"],
                "worst_window": worst,
                "max_abs_ofi_worst_window": float(np.max(np.abs(bk.ofi[in_worst]))),
                "n_buckets": len(bk)},
        "predictive": {"contemporaneous": asdict(cvl["contemporaneous"]),
                       "lagged": asdict(cvl["lagged"]), "imbalance": asdict(imb)},
        "fills": {"summary": summary, "by_queue": by_queue_bucket(mk), "end_reasons": ends,
                  "partial_fills": partial, "n_orders": int((mk["model"] == MODELS[0]).sum())},
        "spread_decomposition": spread_decomposition(day),
        "post_vs_cross": post_vs_cross,
        "kyle_bridge": {"beta_ticks_per_1000_shares": beta * 1000, "r2": r2_day,
                        "beta_usd_per_share": beta * TICK / 1e4,
                        "beta_bps_per_1000_shares": beta * 1000 * facts.tick_bps,
                        "median_window_beta_ticks_per_1000_shares":
                            float(np.median([w["beta"] for w in windows])) * 1000,
                        "mean_depth_best": facts.depth_best,
                        "mean_price_usd": facts.mean_price_usd},
    }
    plot = {"ofi": bk.ofi, "dmid": bk.dmid}
    del day, mk, pc
    gc.collect()
    return out, plot


def run(engine: str = "auto") -> tuple[dict, dict]:
    t0 = time.time()
    engine_used = "cpp" if engine == "cpp" or (engine == "auto" and cpp_available()) else "python"
    per, plots = {}, {}
    for t in TICKERS:
        per[t], plots[t] = analyze(t, engine)
        print(f"  {t}: done ({time.time() - t0:.0f}s)")
    betas = np.array([w["beta"] for t in TICKERS for w in per[t]["cks"]["windows"]])
    depths = np.array([w["depth"] for t in TICKERS for w in per[t]["cks"]["windows"]])
    groups = np.array([t for t in TICKERS for _ in per[t]["cks"]["windows"]])
    res = {
        "meta": {"date": DATE, "tickers": list(TICKERS), "levels": LEVELS, "session": list(SESSION),
                 "trim_seconds": TRIM, "split": SPLIT, "bucket_seconds": BUCKET,
                 "window_seconds": WINDOW, "imbalance_step": STEP, "order_every": EVERY,
                 "max_wait": MAX_WAIT, "order_size": SIZE, "markout_horizons": list(HORIZONS),
                 "spread_horizons": list(SPREAD_HORIZONS), "post_cross_horizon": H,
                 "large_tick_threshold": LARGE_TICK_THRESHOLD, "fill_engine": engine_used,
                 "taker_fee_usd": TAKER_FEE_USD,
                 "kyle_bridge_units": ("beta = OLS slope of the 10 s mid change (ticks) on 10 s OFI "
                                       "(shares) over the trimmed day; *_per_1000_shares scales it "
                                       "by 1,000; usd_per_share = ticks x $0.01; OFI counts limit "
                                       "orders and cancels at the best, not only trades"), "data_source": DATA_SOURCE,
                 "runtime_seconds": None},
        **{k: {t: per[t][k] for t in TICKERS}
           for k in ("consistency", "facts", "predictive", "fills", "spread_decomposition",
                     "post_vs_cross", "kyle_bridge")},
        "cks": {"tickers": {t: per[t]["cks"] for t in TICKERS},
                "depth_slope": asdict(beta_depth_slope(betas, depths)),
                "depth_slope_within": asdict(beta_depth_slope(betas, depths, groups))},
    }
    res["verdict"] = verdict(res)
    res["meta"]["runtime_seconds"] = time.time() - t0
    return res, plots


def classes(res: dict) -> tuple[list[str], list[str]]:
    large = [t for t in res["meta"]["tickers"] if res["facts"][t]["tick_class"] == "large-tick"]
    small = [t for t in res["meta"]["tickers"] if t not in large]
    return large, small


def verdict(res: dict) -> dict:
    """Does the queue-imbalance signal survive realistic (FIFO) fills? Decided from the
    out-of-sample EV of the post-or-cross rule, its block-bootstrap CI and the fee cap."""
    out = {"fee_ticks": TAKER_FEE_TICKS, "tickers": {}}
    for t in res["meta"]["tickers"]:
        r = res["post_vs_cross"][t]["rules"]
        fifo, touch = r["fifo"], r["touch"]
        status = ("positive" if fifo["ci"][0] > 0 else
                  "negative" if fifo["ci"][1] < 0 else "zero")
        out["tickers"][t] = {"fifo_oos": fifo["out_of_sample"], "fifo_ci": fifo["ci"],
                             "fifo_oos_bps": fifo["oos_bps"], "fifo_per_trade": fifo["per_trade"],
                             "fifo_share_traded": fifo["share_post"] + fifo["share_cross"],
                             "fifo_net_of_fee": fifo["net_of_fee"], "fifo_net_ci": fifo["net_ci"],
                             "touch_oos": touch["out_of_sample"], "touch_ci": touch["ci"],
                             "status": status,
                             "clears_fee": status == "positive" and fifo["net_ci"][0] > 0,
                             "touch_positive": touch["ci"][0] > 0}
    out["survives"] = any(v["clears_fee"] for v in out["tickers"].values())
    out["touch_illusion"] = [t for t, v in out["tickers"].items()
                             if v["touch_positive"] and not v["clears_fee"]]
    return out


# --------------------------------------------------------------------------- figures


def render_figures(res: dict, plots: dict) -> None:
    large, small = classes(res)
    busiest = lambda names: max(names, key=lambda t: res["facts"][t]["n_messages"])  # noqa: E731
    shown = [busiest(g) for g in (large, small) if g]
    mp.render("d_tick_regimes", fg.draw_tick_regimes(res))
    mp.render("d_sign_acf", fg.draw_sign_acf(res))
    mp.render("d_cks_scatter", fg.draw_cks_scatter({t: plots[t] for t in shown}, res))
    mp.render("d_cks_beta_depth", fg.draw_beta_depth(res))
    mp.render("d_contemp_vs_lagged", fg.draw_contemp_vs_lagged(res))
    mp.render("d_imbalance_curve", fg.draw_imbalance_curve(res))
    mp.render("d_fill_prob", fg.draw_fill_prob(res))
    mp.render("d_markouts", fg.draw_markouts(res))
    mp.render("d_queue_buckets", fg.draw_queue_buckets(res))
    mp.render("d_spread_decomp", fg.draw_spread_decomp(res))
    if large:
        mp.render("d_post_vs_cross", fg.draw_post_vs_cross(
            res, large, "Post or cross, large-tick stocks: EV(post) − EV(cross) at 30 s"))
    if small:
        mp.render("d_post_vs_cross_small", fg.draw_post_vs_cross(
            res, small, "Post or cross, small-tick stocks: EV(post) − EV(cross) at 30 s"))
    mp.render("d_rule_oos", fg.draw_rule_oos(res))


# --------------------------------------------------------------------------- text


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}%"


def listing(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def each(names: list[str], fmt: Callable[[str], str]) -> str:
    """'x (A), y (B) and z (C)': one formatted item per name."""
    return listing([fmt(t) for t in names])


def mk_row(res: dict, t: str, model: str) -> dict:
    return next(r for r in res["fills"][t]["summary"] if r["model"] == model)


def one_line_answer(res: dict) -> str:
    v = res["verdict"]
    fee = v["fee_ticks"]
    tick = v["tickers"]

    def span(names: list[str], key: str) -> str:
        return each(names, lambda t: f"{t} {tick[t][key]:+.3f}")

    if v["survives"]:
        winners = [t for t, x in tick.items() if x["clears_fee"]]
        text = (f"**Yes, for {listing(winners)}**: under FIFO fills the post-or-cross rule chosen "
                f"before 12:45 still earns {span(winners, 'fifo_net_of_fee')} ticks per decision after "
                f"12:45 net of a {fee:.1f}-tick taker fee on every cross (CI above zero).")
    else:
        neg = [t for t, x in tick.items() if x["status"] == "negative"]
        zero = [t for t, x in tick.items() if x["status"] == "zero"]
        pos = [t for t, x in tick.items() if x["status"] == "positive"]
        details = []
        if neg:
            details.append(f"reliably negative for {listing(neg)}")
        if zero:
            details.append(f"indistinguishable from zero for {listing(zero)}")
        if pos:
            how = each(pos, lambda t: "{} only because it trades on {} of decisions for {:+.2f} ticks "
                                      "each".format(t, pct(tick[t]["fifo_share_traded"], 0),
                                                    tick[t]["fifo_per_trade"]))
            details.append(f"positive for {how}, which a {fee:.1f}-tick taker fee wipes out")
        text = ("**No.** Chosen before 12:45 and scored after, the best post-or-cross rule under "
                f"FIFO fills earns {span(list(tick), 'fifo_oos')} ticks per decision: "
                f"{'; '.join(details)}.")
    if v["touch_illusion"]:
        plural = "s" if len(v["touch_illusion"]) > 1 else ""
        text += (f" Touch fills would have said yes: {span(v['touch_illusion'], 'touch_oos')} ticks "
                 f"per decision, confidence interval{plural} above zero.")
    return text


def section_data(res: dict) -> list[str]:
    meta = res["meta"]
    tickers = meta["tickers"]
    session = f"{clock(meta['session'][0])}–{clock(meta['session'][1])}"
    split = clock(meta["split"])
    return ["## Data and method", "", rp.bullets([
        f"**Data.** {meta['data_source']}. {listing(tickers)} on {meta['date']}, {meta['levels']} "
        "levels: every order-book event with its order id, and the book after it.",
        f"**Trimmed session.** The first and last {meta['trim_seconds'] / 60:.0f} minutes are dropped "
        f"from every analysis ({session}). Book statistics are time-weighted: each state counts for the "
        "seconds it was in effect.",
        f"**Train/test split.** Anything fitted or chosen uses data before {split} and is scored on "
        f"data after it; labels that straddle {split} are dropped.",
        "**Contemporaneous is not predictive.** The Cont–Kukanov–Stoikov (CKS, 2014) regression "
        "explains the mid change over a 10 s bucket with the order-flow imbalance (OFI) of the *same* "
        "bucket; you only know that OFI once the move has happened, so its R² says nothing about "
        "forecasting. The predictive tests are separate: lagged OFI → next bucket's mid change, "
        "and queue imbalance → direction of the next mid move (Gould & Bonart 2016), both scored "
        "out of sample.",
        f"**Hypothetical orders.** {meta['order_size']} shares at the best bid and at the best ask every "
        f"{meta['order_every']:g} s, each cancelled after {meta['max_wait']:g} s or when its price leaves "
        f"the {meta['levels']} visible levels (LOBSTER records nothing deeper). Each order is replayed "
        "alone against the real messages, so it has no market impact.",
        "**Fill models.** *Touch* (optimistic): filled by any print at our price. *FIFO queue* "
        "(realistic): we join behind the displayed depth; later arrivals (tracked by order id) queue "
        "behind us; cancels of anyone else shrink the queue ahead; visible executions consume the "
        "queue ahead first and the rest fills us; hidden executions are ignored because displayed "
        "orders have priority at the same price. *Trade-through* (pessimistic): we are last in line "
        "behind everyone, so only a print strictly beyond our price reaches us. In all three a print "
        "through our price or the opposite quote reaching it fills the order. The spec's extra "
        "trade-through trigger, \"the whole level is executed away\", was dropped: when an aggressor "
        "exactly clears the level, FIFO leaves us first in line but unfilled, so that trigger made the "
        "pessimistic model fill *more* often than FIFO for the small-tick stocks. Without it, "
        "trade-through ⊆ FIFO ⊆ touch holds order by order, and a test checks it.",
        f"**Decision rule.** Every {meta['order_every']:g} s, follow the sign of the queue imbalance I: "
        f"either cross (take the opposite quote) or post at our best quote, marked to the mid "
        f"{meta['post_cross_horizon']:g} s after the decision. Fees and rebates are excluded.",
    ]), ""]


def section_consistency(res: dict) -> list[str]:
    tickers = res["meta"]["tickers"]
    c = res["consistency"]
    rates = [c[t]["pass_rate"] for t in tickers]
    coverage = min(c[t]["coverage"] for t in tickers)
    n_fail = sum(c[t]["n_failures"] for t in tickers)
    rate_text = (f"The pass rate is {pct(rates[0], 4)} for every stock" if min(rates) == max(rates)
                 else f"Pass rates run from {pct(min(rates), 4)} to {pct(max(rates), 4)}")
    cover_text = ("every eligible message could be checked, because LOBSTER only writes events inside "
                  "the requested 10 levels" if coverage == 1.0 else
                  f"at least {pct(coverage, 2)} of eligible messages could be checked (the rest touch a "
                  "price outside the visible levels)")
    fail_text = "" if n_fail == 0 else f" The {rp.num(n_fail)} failures are counted per ticker in the JSON."
    return [
        "### 1. The message and book files agree", "",
        "For every submission, cancel, delete and visible execution whose price is in view before "
        "and after it, the displayed size at that price must change by exactly ± the message size. "
        f"{rate_text}, and {cover_text}. The first message of each file has no book before it and is "
        f"skipped; hidden executions do not touch displayed depth.{fail_text}", "",
        rp.table([{"ticker": t, "messages": rp.num(c[t]["n_messages"]), "checked": rp.num(c[t]["n_checked"]),
                   "passed": rp.num(c[t]["n_pass"]), "pass rate": pct(c[t]["pass_rate"], 4)}
                  for t in tickers]),
        "",
    ]


def section_facts(res: dict) -> list[str]:
    meta = res["meta"]
    tickers = meta["tickers"]
    large, small = classes(res)
    f = res["facts"]
    lag_n = len(f[tickers[0]]["sign_acf"])
    text = (f"A stock is classed *large-tick* when its spread is exactly one tick more than "
            f"{pct(meta['large_tick_threshold'], 0)} of the time. ")
    if large:
        ticks = each(large, lambda t: "{:.1f}".format(f[t]["tick_bps"]))
        ones = each(large, lambda t: pct(f[t]["share_one_tick"]))
        depth = each(large, lambda t: rp.num(f[t]["depth_best"]))
        text += (f"{listing(large)} (one tick = {ticks} bps) sit at a one-tick spread {ones} of the "
                 f"time, with {depth} shares at the best. ")
    if small:
        ticks = each(small, lambda t: "{:.2f}".format(f[t]["tick_bps"]))
        spreads = each(small, lambda t: "{:.1f}".format(f[t]["spread_ticks"]))
        depth = each(small, lambda t: rp.num(f[t]["depth_best"]))
        text += (f"{listing(small)} (one tick = {ticks} bps) quote spreads of {spreads} ticks on "
                 f"average with only {depth} shares at the best. ")
    acf1 = each(tickers, lambda t: "{:.2f}".format(f[t]["sign_acf"][0]))
    acfn = each(tickers, lambda t: "{:.2f}".format(f[t]["sign_acf"][-1]))
    text += ("Everything below splits along this line. Trade signs are strongly persistent in both "
             f"regimes: lag-1 autocorrelation of {acf1}, still {acfn} at lag {lag_n}. Visible "
             "executions sharing a timestamp and side are grouped into one trade, so the lag-1 "
             "figure isn't just one order sweeping several resting orders.")
    rows = [{"ticker": t, "price $": "{:.2f}".format(f[t]["mean_price_usd"]),
             "tick (bps)": "{:.2f}".format(f[t]["tick_bps"]),
             "spread (ticks)": "{:.2f}".format(f[t]["spread_ticks"]),
             "spread (bps)": "{:.2f}".format(f[t]["spread_bps"]),
             "1-tick share": pct(f[t]["share_one_tick"]), "depth at best": rp.num(f[t]["depth_best"]),
             "messages": rp.num(f[t]["n_messages"]), "visible execs": rp.num(f[t]["n_visible_exec"]),
             "hidden execs": rp.num(f[t]["n_hidden_exec"]), "trades": rp.num(f[t]["n_trades"]),
             "class": f[t]["tick_class"]} for t in tickers]
    acf_rows = [{"ticker": t, **{f"lag {k + 1}": "{:.3f}".format(a) for k, a in enumerate(f[t]["sign_acf"])}}
                for t in tickers]
    return ["### 2. Two tick-size regimes", "", text, "",
            mp.picture("d_tick_regimes", "Tick size in bps, mean spread in ticks and depth at the best per stock"),
            "", rp.table(rows), "",
            mp.picture("d_sign_acf", "Trade-sign autocorrelation by lag for each stock"), "",
            rp.table(acf_rows), ""]


def section_cks(res: dict) -> list[str]:
    tickers = res["meta"]["tickers"]
    f = res["facts"]
    cks = res["cks"]
    kb = res["kyle_bridge"]
    slope, within = cks["depth_slope"], cks["depth_slope_within"]
    n_windows = len(cks["tickers"][tickers[0]]["windows"])
    worst_t = min(tickers, key=lambda t: cks["tickers"][t]["min_r2"])
    worst = cks["tickers"][worst_t]["worst_window"]
    big = cks["tickers"][worst_t]["max_abs_ofi_worst_window"]
    w_lo = max(worst["start"], res["meta"]["session"][0])
    w_hi = min(worst["start"] + WINDOW, res["meta"]["session"][1])
    # Diagnosed by reading that window's messages (GOOG 09:35-10:00): a bid re-posted at
    # 576.42/576.44 above a 1-share order, each copy cancelled while it sat at level 2.
    flicker = (worst_t, clock(w_lo)) == ("GOOG", "09:35")
    reason = ("Reading the messages shows a quote flickering between the first two bid levels: "
              "level-1 OFI counts every re-post at the best, but not the cancel one level deeper, "
              "so the imbalance piles up. This is a real blind spot of the level-1 measure."
              if flicker else "Those buckets were not inspected further.")
    mean_r2 = each(tickers, lambda t: "{:.2f} ({})".format(cks["tickers"][t]["mean_r2"], t))
    large, small = classes(res)
    r2 = {t: cks["tickers"][t]["mean_r2"] for t in tickers}
    higher = (", higher for the large-tick stocks, where almost every price change is a queue "
              "being depleted" if large and small and min(r2[t] for t in large) > max(r2[t] for t in small)
              else "")
    text1 = (f"OFI summed over {res['meta']['bucket_seconds']:g} s buckets explains the same bucket's mid "
             f"change. The mean R² across the {n_windows} half-hour windows is {mean_r2}{higher}. "
             f"The lowest window R² is {worst['r2']:.4f} ({worst_t}, {clock(w_lo)}–{clock(w_hi)}). "
             f"That window contains buckets with up to {rp.num(big)} shares of OFI (about "
             f"{big / f[worst_t]['depth_best']:.0f}× the mean depth at the best) and no matching price "
             f"move. {reason}")
    if slope["ci"][0] <= -1 <= slope["ci"][1]:
        verdict_pooled = "That interval contains CKS's −1. "
    else:
        side = "lies below" if slope["ci"][1] < -1 else "lies above"
        verdict_pooled = (f"That interval {side} CKS's −1. The pooled slope mostly compares stocks, "
                          "which differ in more than depth. ")
    verdict_within = (", consistent with −1." if within["ci"][0] <= -1 <= within["ci"][1]
                      else ", not consistent with −1.")
    dropped = f" {slope['n_dropped']} windows with β ≤ 0 were dropped." if slope["n_dropped"] else ""
    text2 = (f"Across windows and stocks, log β on log mean depth has a slope of **{slope['slope']:.2f}** "
             f"(95% CI {slope['ci'][0]:.2f} to {slope['ci'][1]:.2f}, HC3, n = {slope['n']}). "
             f"{verdict_pooled}Within stocks (one intercept per ticker) the slope is "
             f"{within['slope']:.2f} (95% CI {within['ci'][0]:.2f} to {within['ci'][1]:.2f})"
             f"{verdict_within}{dropped}")
    table = rp.table([{"ticker": t, "mean R² (windows)": "{:.3f}".format(cks["tickers"][t]["mean_r2"]),
                       "median R²": "{:.3f}".format(cks["tickers"][t]["median_r2"]),
                       "min R²": "{:.3f}".format(cks["tickers"][t]["min_r2"]),
                       "full-day β (ticks per 1,000 sh)": "{:.4g}".format(kb[t]["beta_ticks_per_1000_shares"]),
                       "full-day R²": "{:.3f}".format(kb[t]["r2"]),
                       "buckets": cks["tickers"][t]["n_buckets"]} for t in tickers])
    windows = rp.table([{"ticker": t, "window": clock(max(w["start"], res["meta"]["session"][0])),
                         "buckets": w["n"],
                         "β (ticks per 1,000 sh)": "{:.4g}".format(1000 * w["beta"]),
                         "R²": "{:.3f}".format(w["r2"]), "mean depth": rp.num(w["depth"])}
                        for t in tickers for w in cks["tickers"][t]["windows"]])
    betas = each(tickers, lambda t: "{} {:.3g}".format(t, kb[t]["beta_ticks_per_1000_shares"]))
    bridge = (f"**Bridge to module E (Kyle's λ).** The full-day CKS β is saved under `kyle_bridge` in "
              f"`reports/results/{NAME}.json`: {betas} ticks per 1,000 shares of OFI. It is a cousin of "
              "Kyle's λ, not the same object: OFI counts limit-order arrivals and cancels at the best as "
              "well as trades.")
    return ["### 3. CKS replication (contemporaneous)", "", text1, "", text2, "",
            mp.picture("d_cks_scatter", "Mid change versus OFI in 10-second buckets for one large-tick and one small-tick stock"),
            "", table, "",
            mp.picture("d_cks_beta_depth", "CKS beta against mean depth per half-hour window, log-log, with the fitted slope"),
            "", "<details><summary>Window-level β and depth (the points in the figure)</summary>", "",
            windows, "", "</details>", "", bridge, ""]


def section_predictive(res: dict) -> list[str]:
    meta = res["meta"]
    tickers = meta["tickers"]
    large, small = classes(res)
    pred = res["predictive"]
    imb = {t: pred[t]["imbalance"] for t in tickers}
    auc = {t: imb[t]["auc"] for t in tickers}
    split = clock(meta["split"])
    contemp = each(tickers, lambda t: "{:.2f}".format(pred[t]["contemporaneous"]["r2_oos"]))
    lagged = each(tickers, lambda t: "{:+.4f}".format(pred[t]["lagged"]["r2_oos"]))
    gap = all(pred[t]["lagged"]["r2_oos"] < 0.1 * pred[t]["contemporaneous"]["r2_oos"] for t in tickers)
    text1 = (f"Fitted before {split} and scored after, the *contemporaneous* CKS regression keeps an "
             f"out-of-sample R² of {contemp}. The same OFI used to forecast the *next* bucket scores "
             f"{lagged} (out-of-sample R² against the training mean; zero or below means no forecasting "
             "value)." + (" That gap is why a high CKS R² is not an alpha." if gap else ""))
    aucs = each(tickers, lambda t: "{:.3f} ({})".format(auc[t], t))
    large_min = min((auc[t] for t in large), default=float("nan"))
    small_max = max((auc[t] for t in small), default=float("nan"))
    if large and small and large_min > small_max:
        order = (f"Every large-tick stock beats every small-tick one ({large_min:.3f} vs at most "
                 f"{small_max:.3f}), as Gould & Bonart report: long queues deplete slowly and visibly, "
                 "while thin small-tick queues are constantly replaced and say little. ")
    else:
        order = "The large-tick/small-tick ordering reported by Gould & Bonart does not hold cleanly here. "
    slopes = each(tickers, lambda t: "{:.2f}".format(imb[t]["coef"]))
    lead = ("Queue imbalance does carry information about the next mid move. " if min(auc.values()) > 0.5
             else "Queue imbalance is not uniformly informative about the next mid move. ")
    text2 = (f"{lead}Out-of-sample AUC is {aucs}. {order}The logistic slope on I is {slopes}. Samples are taken every "
             f"{meta['imbalance_step']:g} s; the label is the direction of the next mid change.")
    table = rp.table([{"ticker": t,
                       "contemp. R² (OOS)": "{:.3f}".format(pred[t]["contemporaneous"]["r2_oos"]),
                       "lagged R² (OOS)": "{:+.4f}".format(pred[t]["lagged"]["r2_oos"]),
                       "lagged R² vs zero": "{:+.4f}".format(pred[t]["lagged"]["r2_oos_zero"]),
                       "imbalance AUC (OOS)": "{:.3f}".format(auc[t]),
                       "AUC (train)": "{:.3f}".format(imb[t]["auc_train"]),
                       "logit slope": "{:.2f}".format(imb[t]["coef"]),
                       "Brier (model)": "{:.4f}".format(imb[t]["brier_test"]),
                       "Brier (base rate)": "{:.4f}".format(imb[t]["brier_base"]),
                       "test samples": imb[t]["n_test"]} for t in tickers])

    def cell(t: str, k: int) -> str:
        c = imb[t]["curve"][k]
        return "n/a" if c["p_up"] is None else "{:.2f} (n={})".format(c["p_up"], c["n"])

    bins = imb[tickers[0]]["curve"]
    curve = rp.table([{"I bin": "{:+.1f} to {:+.1f}".format(c["lo"], c["hi"]),
                       **{t: cell(t, k) for t in tickers}} for k, c in enumerate(bins)])
    return ["### 4. Predictive tests (the ones a trader could use)", "", text1, "", text2, "",
            mp.picture("d_contemp_vs_lagged", "Out-of-sample R-squared, contemporaneous versus predictive, and queue-imbalance AUC per stock"),
            "", table, "",
            mp.picture("d_imbalance_curve", "Probability that the next mid move is up, by queue-imbalance bin, per stock"),
            "", "<details><summary>P(next move up) by imbalance bin, test set (the points in the figure)</summary>",
            "", curve, "", "</details>", ""]


def section_fills(res: dict) -> list[str]:
    meta = res["meta"]
    tickers = meta["tickers"]
    large, _ = classes(res)
    fl = res["fills"]
    row = {(t, m): mk_row(res, t, m) for t in tickers for m in MODELS}
    fifo_neg = [t for t in tickers if row[t, "fifo"]["mk_10"] < 0]
    touch_pos = [t for t in tickers if row[t, "touch"]["mk_10"] > 0]
    p_fifo = each(tickers, lambda t: "{} ({})".format(pct(row[t, "fifo"]["p_fill"], 0), t))
    p_touch = each(tickers, lambda t: pct(row[t, "touch"]["p_fill"], 0))
    p_through = each(tickers, lambda t: pct(row[t, "through"]["p_fill"], 0))
    gap = each(large, lambda t: "{} {}".format(t, pct(row[t, "touch"]["p_fill"] - row[t, "fifo"]["p_fill"], 0)))
    text1 = (f"Across {rp.num(fl[tickers[0]]['n_orders'])} hypothetical orders per stock, FIFO fills "
             f"{p_fifo} of them within {meta['max_wait']:g} s, against {p_touch} under touch and "
             f"{p_through} under trade-through.")
    if large:
        text1 += (f" The gap is largest for the large-tick stocks, where queues are long: {gap} of "
                  "orders are 'filled' by touch but not by the queue.")
    unc = each(tickers, lambda t: "{:.2f}".format(row[t, "fifo"]["uncond_10"]))
    m10 = {m: each(tickers, lambda t, m=m: "{:+.2f}".format(row[t, m]["mk_10"])) for m in MODELS}
    less = all(row[t, m]["mk_10"] < row[t, m]["uncond_10"] for t in tickers for m in MODELS)
    text2 = ("A random fill would earn the half-spread, and the *unconditional* markout "
             f"({unc} ticks) is exactly that, because both sides are quoted at every decision time and "
             "the mid move cancels. " + ("Filled orders earn less under every model. " if less else "")
             + "At 10 s the FIFO markout is "
             f"{m10['fifo']} ticks, touch {m10['touch']}, trade-through {m10['through']}. ")
    both = [t for t in tickers if t in touch_pos and t in fifo_neg]
    if both:
        text2 += (f"Touch fills keep a positive 10 s markout in {listing(both)}, where FIFO fills lose "
                  "money. The passive edge that touch fills show does not survive a realistic queue. ")
    elif fifo_neg:
        text2 += f"FIFO fills lose money at 10 s in {listing(fifo_neg)}. "
    text2 += ("This is adverse selection, the passive trader's winner's curse. The fills a queue "
              "actually gives you are the ones where the price is about to move through your level.")
    rows = [{"ticker": t, "model": m, "P(fill)": "{:.3f}".format(row[t, m]["p_fill"]),
             **{f"{h:g} s": "{:+.3f}".format(row[t, m][f"mk_{h:g}"]) for h in HORIZONS},
             "10 s (bps)": "{:+.3f}".format(row[t, m]["mk_10_bps"]),
             "uncond. (ticks)": "{:.3f}".format(row[t, m]["uncond_10"]),
             "partial fills": fl[t]["partial_fills"][m]} for t in tickers for m in MODELS]
    reasons = sorted({k for t in tickers for m in MODELS for k in fl[t]["end_reasons"][m]})
    ends = [{"ticker": t, **{r: " / ".join(rp.num(fl[t]["end_reasons"][m].get(r, 0)) for m in MODELS)
                             for r in reasons}} for t in tickers]
    queue = [{"ticker": t, "queue ahead": r["label"], "median queue (sh)": rp.num(r["median_queue"]),
              "orders": rp.num(r["n"]), "P(fill)": "{:.3f}".format(r["p_fill"]),
              "10 s markout (ticks)": "{:+.3f}".format(r["mk"])}
             for t in tickers for r in fl[t]["by_queue"] if r["model"] == "fifo"]
    q = {t: sorted((r for r in fl[t]["by_queue"] if r["model"] == "fifo"),
                   key=lambda r: r["queue_bucket"]) for t in tickers}
    fewer = [t for t in tickers if q[t][-1]["p_fill"] < q[t][0]["p_fill"]]
    worse = [t for t in tickers if q[t][-1]["mk"] < q[t][0]["mk"]]
    p_ends = each(fewer, lambda t: "{} {} → {}".format(t, pct(q[t][0]["p_fill"], 0), pct(q[t][-1]["p_fill"], 0)))
    m_ends = each(worse, lambda t: "{} {:+.2f} → {:+.2f}".format(t, q[t][0]["mk"], q[t][-1]["mk"]))
    queue_text = (f"From the shortest to the longest queue bucket, P(fill) falls in {listing(fewer)} "
                  f"({p_ends})" if fewer else "P(fill) does not fall with the queue ahead in any stock")
    others = [t for t in tickers if t not in fewer]
    if fewer and others:
        queue_text += f" but not in {listing(others)}, whose queues at the best are a few round lots"
    queue_text += (f". The 10 s markout of filled orders is worse at the back of a long queue in "
                   f"{listing(worse)} ({m_ends}): waiting longer in line does not buy a better fill, "
                   "because the fills that do arrive are the ones where the price moves through the level."
                   if worse else ". Markouts do not worsen with the queue ahead.")
    return ["### 5. Fill probability and markouts by fill model", "", text1, "", text2, "",
            mp.picture("d_fill_prob", "Probability of a fill within 60 seconds by fill model and stock"), "",
            mp.picture("d_markouts", "Markout of filled orders at 0.1, 1, 10 and 60 seconds by fill model, per stock"), "",
            "Markouts in ticks per share (quantity-weighted over filled orders), side × (mid − fill price):", "",
            rp.table(rows), "", "How orders ended (touch / FIFO / trade-through):", "", rp.table(ends), "",
            queue_text, "",
            mp.picture("d_queue_buckets", "FIFO fill probability and 10-second markout by queue-ahead bucket"), "",
            rp.table(queue), ""]


def section_spread(res: dict) -> list[str]:
    meta = res["meta"]
    tickers = meta["tickers"]
    sd = res["spread_decomposition"]
    last = len(meta["spread_horizons"]) - 1
    eff = each(tickers, lambda t: "{:.2f}".format(sd[t][0]["effective_bps"]))
    imp = each(tickers, lambda t: "{:.2f}".format(sd[t][last]["impact_bps"]))
    real = each(tickers, lambda t: "{:+.2f}".format(sd[t][last]["realized_bps"]))
    neg = [t for t in tickers if sd[t][last]["realized_ticks"] < 0]
    tail = (f"Liquidity providers lose before rebates in {listing(neg)}." if neg else
            "Liquidity providers keep a positive realized spread in every stock.")
    text = ("On the file's own trades (visible executions grouped per aggressive order, "
            f"share-weighted), the effective spread is {eff} bps. At {meta['spread_horizons'][last]:g} s "
            f"the price impact is {imp} bps and the realized spread, what liquidity providers keep, is "
            f"{real} bps. {tail}")
    rows = [{"ticker": t, "horizon (s)": "{:g}".format(r["horizon"]), "trades": r["n_trades"],
             "effective (ticks)": "{:.3f}".format(r["effective_ticks"]),
             "realized (ticks)": "{:+.3f}".format(r["realized_ticks"]),
             "impact (ticks)": "{:.3f}".format(r["impact_ticks"]),
             "effective (bps)": "{:.3f}".format(r["effective_bps"]),
             "realized (bps)": "{:+.3f}".format(r["realized_bps"]),
             "impact (bps)": "{:.3f}".format(r["impact_bps"])} for t in tickers for r in sd[t]]
    return ["### 6. Effective spread = realized spread + price impact", "", text, "",
            mp.picture("d_spread_decomp", "Effective spread, realized spread and price impact in bps at two horizons"),
            "", rp.table(rows), ""]


def section_post_cross(res: dict) -> list[str]:
    meta = res["meta"]
    tickers = meta["tickers"]
    large, small = classes(res)
    pvc = res["post_vs_cross"]
    H_s = "{:g}".format(meta["post_cross_horizon"])
    signal_lo = {label: float(label.split("-")[0]) for label in SIGNAL_LABELS}
    out = ["### 7. Post or cross?", "",
           f"At each decision the trader follows sign(I). EV(cross) = s·(mid(t+{H_s} s) − opposite quote) and "
           "EV(post) = filled fraction × s·(mid(t+H) − our quote), in ticks per share, where s = +1 for a buy. "
           "The heatmaps show EV(post) − EV(cross) by signal strength |I| and by queue ahead (in multiples of "
           "the stock's median queue). Blue cells favour posting, red cells favour crossing, and empty cells "
           "have fewer than 20 decisions.", ""]

    def mean(t: str, m: str, key: str, fmt: str = "{:+.2f}") -> str:
        return fmt.format(pvc[t]["means"][m][key])

    def cells(t: str, m: str) -> np.ndarray:
        return np.array(pvc[t]["grid"][m], dtype=float)

    if large:
        n_cells = cells(large[0], "fifo").size
        move = each(large, lambda t: mean(t, "fifo", "move"))
        cross = each(large, lambda t: mean(t, "fifo", "ev_cross"))
        post_f = each(large, lambda t: mean(t, "fifo", "ev_post"))
        post_t = each(large, lambda t: mean(t, "touch", "ev_post"))
        neg_f = each(large, lambda t: "{} ({})".format(int(np.sum(cells(t, "fifo") < 0)), t))
        neg_t = each(large, lambda t: str(int(np.sum(cells(t, "touch") < 0))))
        strength = [c for t in large for c in pvc[t]["cells"]["fifo"] if c["n"] >= 20]
        win_x = [c for c in strength if c["diff"] < 0]
        win_p = [c for c in strength if c["diff"] >= 0]

        def wavg(cs: list[dict], key: str) -> float:
            return float(np.average([c[key] for c in cs], weights=[c["n"] for c in cs])) if cs else float("nan")

        strong = sum(signal_lo[c["signal_bucket"]] >= 0.4 for c in win_x)
        where = (f"{strong} of the {len(win_x)} cells where crossing wins have |I| above 0.4, and posts there fill "
                 f"{pct(wavg(win_x, 'p_fill'), 0)} of the time under FIFO, against "
                 f"{pct(wavg(win_p, 'p_fill'), 0)} in the cells where posting wins: a strong signal means "
                 "the queue on our side is long relative to the other side, and the price tends to leave "
                 "without us." if win_x and wavg(win_x, "p_fill") < wavg(win_p, "p_fill") else
                 "The cells where crossing wins are not simply the ones where posts fill least.")
        spread_l = each(large, lambda t: mean(t, "fifo", "spread", "{:.2f}"))
        out += [f"**Large-tick ({listing(large)}).** Crossing pays half the spread ({spread_l} ticks on "
                f"average), and the signal's average move over {H_s} s is {move} ticks, so EV(cross) "
                f"averages {cross} ticks. Under FIFO, "
                f"posting averages {post_f} ticks against {post_t} under touch. Crossing wins in {neg_f} of the "
                f"{n_cells} FIFO cells, against {neg_t} under touch. {where}", "",
                mp.picture("d_post_vs_cross", "Heatmaps of EV(post) minus EV(cross) for large-tick stocks under three fill models"),
                ""]
    if small:
        spread = each(small, lambda t: mean(t, "fifo", "spread", "{:.1f}"))
        move = each(small, lambda t: mean(t, "fifo", "move"))
        cross = each(small, lambda t: mean(t, "fifo", "ev_cross", "{:+.1f}"))
        smallest = each(small, lambda t: "{:+.1f}".format(np.nanmin(cells(t, "fifo"))))
        post = each(small, lambda t: mean(t, "fifo", "ev_post"))
        all_pos = all(np.nanmin(cells(t, m)) > 0 for t in small for m in MODELS)
        claim = ("Posting beats crossing in every cell under every fill model" if all_pos
                 else "Posting beats crossing in most cells")
        costly = all(pvc[t]["means"]["fifo"]["ev_cross"] < 0 for t in small)
        loses = all(pvc[t]["means"]["fifo"]["ev_post"] < 0 for t in small)
        out += [f"**Small-tick ({listing(small)}).** Crossing a {spread}-tick spread "
                + ("costs far more than the signal moves" if costly else "is compared with the signal's move")
                + f" ({move} ticks), so EV(cross) is {cross} ticks. {claim} (smallest FIFO advantage "
                f"{smallest} ticks)" + (", so there is no boundary to draw. " if all_pos else ". ")
                + ("Posting still loses on average" if loses else "Posting averages")
                + f": {post} ticks under FIFO.", "",
                mp.picture("d_post_vs_cross_small", "Heatmaps of EV(post) minus EV(cross) for small-tick stocks under three fill models"),
                ""]
    means = rp.table([{"ticker": t, "model": m, "P(fill)": mean(t, m, "p_fill", "{:.3f}"),
                       "EV(post) ticks": mean(t, m, "ev_post", "{:+.3f}"),
                       "EV(cross) ticks": mean(t, m, "ev_cross", "{:+.3f}"),
                       "EV(post) bps": mean(t, m, "ev_post_bps", "{:+.3f}"),
                       "EV(cross) bps": mean(t, m, "ev_cross_bps", "{:+.3f}"),
                       "decisions": rp.num(pvc[t]["n_decisions"])} for t in tickers for m in MODELS])
    cell_rows = rp.table([{"ticker": t, "model": m, "queue ahead": c["queue_bucket"],
                           "|I|": c["signal_bucket"], "n": c["n"], "P(fill)": "{:.2f}".format(c["p_fill"]),
                           "EV(post)": "{:+.3f}".format(c["ev_post"]),
                           "EV(cross)": "{:+.3f}".format(c["ev_cross"]),
                           "diff": "{:+.3f}".format(c["diff"])}
                          for t in tickers for m in MODELS for c in pvc[t]["cells"][m]])

    def rule(t: str, m: str) -> dict:
        r = pvc[t]["rules"][m]
        return {"ticker": t, "model": m, "in-sample": "{:+.3f}".format(r["in_sample"]),
                "out-of-sample": "{:+.3f}".format(r["out_of_sample"]),
                "95% CI": "{:+.3f} to {:+.3f}".format(*r["ci"]), "OOS bps": "{:+.4f}".format(r["oos_bps"]),
                "posts": pct(r["share_post"], 0), "crosses": pct(r["share_cross"], 0),
                "per trade": "{:+.3f}".format(r["per_trade"]),
                "net of taker fee": "{:+.3f}".format(r["net_of_fee"]),
                "always post": "{:+.3f}".format(r["always_post"]),
                "always cross": "{:+.3f}".format(r["always_cross"])}

    gaps = [pvc[t]["rules"][m]["in_sample"] - pvc[t]["rules"][m]["out_of_sample"]
            for t in tickers for m in MODELS]
    split = clock(meta["split"])
    out += ["Mean EV per decision (all cells):", "", means, "",
            "<details><summary>Every heatmap cell (the numbers in the figures)</summary>", "",
            cell_rows, "", "</details>", "",
            "**The rule, scored honestly.** Picking the best action (post, cross or stay out) cell by "
            "cell and scoring it on the same decisions is the optimizer's curse. So the rule is chosen "
            f"on decisions before {split} and scored after:", "",
            mp.picture("d_rule_oos", "In-sample versus out-of-sample EV of the cell-by-cell rule per stock and fill model"),
            "", rp.table([rule(t, m) for t in tickers for m in MODELS]), "",
            f"The in-sample EV exceeds the out-of-sample EV in {sum(g > 0 for g in gaps)} of {len(gaps)} "
            f"ticker × model cases, by {np.mean(gaps):+.3f} ticks per decision on average. EV is per "
            "share, the CIs come from a block bootstrap over 5-minute blocks, and the bps figures use "
            "each decision's own mid.", ""]
    return out


def section_answer(res: dict) -> list[str]:
    meta = res["meta"]
    v = res["verdict"]
    return [
        "## Does the signal survive realistic fills?", "",
        one_line_answer(res) + f" The fee test charges every cross the Reg NMS access-fee cap, "
        f"${meta['taker_fee_usd']:.3f} a share = {v['fee_ticks']:.1f} ticks, and credits posts nothing, "
        "although a maker would collect a rebate of similar size.", "",
        "## Limitations", "",
        rp.bullets([
            f"**One day, 2012.** Five stocks on {meta['date']}. Queue dynamics, fees and the tick-size "
            "regime of these names have changed since, and one day is a small sample for anything intraday.",
            "**No impact of our own orders.** Each hypothetical order is replayed alone. Nobody reacts to "
            "it, and it takes no liquidity away from anyone else. Real queue position also depends on "
            "latency, which is zero here.",
            "**Queue-model assumptions.** The queue ahead is an aggregate share count, not a list of "
            "orders. Any cancel of an id not seen joining after us is assumed to be ahead of us. Every "
            "visible execution at our price consumes the queue ahead first, whichever order it hit. A "
            "print through our price, or the opposite quote reaching it, fills the whole order. Orders "
            f"are cancelled when their price leaves the {meta['levels']} visible levels, because LOBSTER "
            "records nothing deeper.",
            "**Fees, rebates and hidden liquidity.** Exchange fees and maker rebates are excluded. Hidden "
            "executions are ignored by FIFO, and nearly all of them print inside the spread (midpoint orders).",
            "**Level-1 OFI.** Changes behind the best are invisible to e_n, as the flickering-quote window shows.",
        ]), ""]


def build_text(res: dict) -> str:
    head = [
        "# 02 · Microstructure lab: does a short-horizon signal survive realistic fills?", "",
        "**Question.** Queue imbalance at the best quotes predicts the next mid-price move, most "
        "strongly in large-tick stocks. Can a trader monetise that prediction once fills are simulated "
        "from order-level queue positions, instead of assuming a fill whenever the price touches the "
        "order?", "",
        "**Answer.** " + one_line_answer(res), "",
    ]
    body = (section_data(res) + ["## Results", ""] + section_consistency(res) + section_facts(res)
            + section_cks(res) + section_predictive(res) + section_fills(res) + section_spread(res)
            + section_post_cross(res) + section_answer(res))
    return "\n".join(head + body)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Recompute report 02 (microstructure).")
    ap.add_argument("--engine", default="auto", choices=("auto", "python", "cpp"))
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)
    if not available():
        print("LOBSTER Parquet files missing: run `make lobster` first")
        return 1
    res, plots = run(args.engine)
    bench = results.load("cpp_bench")
    if bench:
        res["meta"]["cpp_bench_present"] = True
    results.save(NAME, res)
    if not args.no_figures:
        render_figures(res, plots)
    text = build_text(res)
    if bench:  # the engineering section goes before the closing limitations
        from markout.lob.bench import report_section
        head, sep, tail = text.partition("\n## Limitations")
        text = head + "\n" + report_section(bench).rstrip() + "\n" + sep + tail
    rp.write("02_microstructure", text)
    print(f"wrote reports/02_microstructure.md and reports/results/{NAME}.json "
          f"({res['meta']['runtime_seconds']:.0f}s, fills engine: {res['meta']['fill_engine']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
