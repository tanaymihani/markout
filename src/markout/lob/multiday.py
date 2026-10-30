"""Study 07: does the fill gap hold across days and stocks? NASDAQ ITCH sample days.

Report 02 measured one 2012 day of LOBSTER data. This runs the same experiment on every NASDAQ
ITCH sample day converted by `markout.lob.itch` (2019-2021), for ten stocks: the five of report
02 and five chosen to contrast large and small ticks. For each day and stock:

- the stylized facts (price, spread, depth, tick class);
- the three fill models' fill rates and markouts, on the same grid as report 02 (both sides
  every 5 s, 100 shares, resting up to 60 s); the headline is the touch-minus-FIFO gap in the
  10-s markout, in bps;
- the C++ engine's throughput on the full day (messages replayed per second);
- the L2-only fill model against the FIFO "truth" (report 08): fill rate, markout, and how
  often the two agree on whether an order filled at all;
- a latency sweep: the same decisions with the order reaching the book 1 ms to 500 ms later.

The post-or-cross rule of report 02 is then chosen on every day but one and scored on the day
left out, so each day is out of sample once; intervals resample whole days (`bootstrap.mean_ci`
with block length 1), because decisions within a day share a market.

    python -m markout.lob.multiday        # -> reports/07_many_days.md, 08_l2_and_latency.md
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from markout import plotting as mp
from markout import reporting as R
from markout import results
from markout.evaluation.bootstrap import mean_ci
from markout.lob import itch, lobster
from markout.lob.facts import compute_facts
from markout.lob.fills import ALL_MODELS, MODELS, End, cpp_available, simulate
from markout.lob.markouts import order_markouts, sample_orders, summarize_markouts
from markout.lob.postcross import TAKER_FEE_TICKS, apply, choose, decisions

NAME = "many_days"
TICKERS = ("AAPL", "AMZN", "GOOG", "INTC", "MSFT", "CSCO", "SIRI", "MU", "NFLX", "BKNG")
REPORT_02 = ("AAPL", "AMZN", "GOOG", "INTC", "MSFT")
H = 10.0  # the headline markout horizon, seconds
MODEL_LABEL = {"touch": "Touch", "fifo": "FIFO queue", "through": "Trade-through", "l2": "L2 estimate"}
LATENCIES = (0.0, 0.001, 0.005, 0.02, 0.1, 0.5)  # seconds from the decision to the order reaching the book
LAT_MODELS = ("touch", "fifo", "l2")


def _ci(x: list[float] | np.ndarray) -> dict:
    """Mean and a 95% interval that resamples days (needs 3 or more days)."""
    v = np.asarray([y for y in x if y is not None and np.isfinite(y)], dtype=float)
    out = {"mean": float(v.mean()) if v.size else float("nan"), "n": int(v.size), "lo": None, "hi": None}
    if v.size >= 3:
        c = mean_ci(v, block=1.0, n_boot=5000, seed=0)
        out.update(lo=c["lo"], hi=c["hi"])
    return out


def throughput(day: lobster.Day, orders, engine: str) -> dict:
    """Messages replayed per second by each fill model (one warm-up run, then the best of three)."""
    out: dict[str, Any] = {"messages_replayed": 0, "seconds": {}, "messages_per_s": {}}
    first = int(orders.place_idx.min())
    for model in MODELS:
        best, f = float("inf"), None
        for _ in range(4):
            t0 = time.perf_counter()
            f = simulate(day.messages, day.book, orders, model, engine=engine)
            best = min(best, time.perf_counter() - t0)
        assert f is not None
        replayed = int(f.end_idx.max()) - first + 1
        out["messages_replayed"] = max(out["messages_replayed"], replayed)
        out["seconds"][model] = best
        out["messages_per_s"][model] = replayed / best
    return out


def cell(day: str, ticker: str, engine: str) -> tuple[dict, pd.DataFrame]:
    """Everything for one stock on one day, plus its post-or-cross decision frame."""
    d = lobster.load_day(ticker, day)
    f = compute_facts(d)
    sample = sample_orders(d)
    per_order = order_markouts(d, sample, models=ALL_MODELS, engine=engine)
    rows = {r["model"]: r for r in summarize_markouts(per_order)}
    filled = per_order.assign(f=per_order["filled_qty"] > 0).pivot(index="order", columns="model", values="f")
    out: dict[str, Any] = {
        "day": day,
        "ticker": ticker,
        "rows": len(d),
        "mean_price_usd": f.mean_price_usd,
        "tick_bps": f.tick_bps,
        "spread_ticks": f.spread_ticks,
        "share_one_tick": f.share_one_tick,
        "depth_best": f.depth_best,
        "tick_class": f.tick_class,
        "orders": len(sample.orders),
        "p_fill": {m: rows[m]["p_fill"] for m in ALL_MODELS},
        "mk10": {m: rows[m][f"mk_{H:g}"] for m in ALL_MODELS},
        "mk10_bps": {m: rows[m][f"mk_{H:g}_bps"] for m in ALL_MODELS},
        "l2_agrees": float((filled["l2"] == filled["fifo"]).mean()),  # same filled / not filled outcome
        "l2_fills_fifo_doesnt": float((filled["l2"] & ~filled["fifo"]).mean()),
        "fifo_fills_l2_doesnt": float((filled["fifo"] & ~filled["l2"]).mean()),
        "uncond10_bps": rows["fifo"][f"uncond_{H:g}_bps"],
        "engine": throughput(d, sample.orders, engine),
    }
    out["gap_bps"] = out["mk10_bps"]["touch"] - out["mk10_bps"]["fifo"]
    out["gap_ticks"] = out["mk10"]["touch"] - out["mk10"]["fifo"]
    out["latency"] = []
    for lat in LATENCIES:
        late = sample_orders(d, latency=lat)
        df = order_markouts(d, late, models=LAT_MODELS, engine=engine)
        r = {x["model"]: x for x in summarize_markouts(df)}
        out["latency"].append(
            {
                "latency_s": lat,
                "rejected": float((df.loc[df["model"] == "fifo", "end_reason"] == End.REJECTED).mean()),
                "p_fill": {m: r[m]["p_fill"] for m in LAT_MODELS},
                "mk10_bps": {m: r[m][f"mk_{H:g}_bps"] for m in LAT_MODELS},
            }
        )
    dec = decisions(d, engine=engine).assign(day=day)
    del d
    gc.collect()
    return out, dec


def day_out_of_sample(dec: pd.DataFrame, model: str) -> dict:
    """Choose the post-or-cross rule on all other days, score it on each day in turn."""
    dm = dec[dec["model"] == model]
    days = sorted(dm["day"].unique())
    per = []
    for day in days:
        test, train = dm[dm["day"] == day], dm[dm["day"] != day]
        rule = choose(train)
        ev = apply(rule, test)
        act = np.array(
            [rule.get((int(q), int(s)), "none") for q, s in zip(test["queue_bucket"], test["signal_bucket"])]
        )
        per.append(
            {
                "day": day,
                "oos": float(ev.mean()),
                "net_of_fee": float((ev - TAKER_FEE_TICKS * (act == "cross")).mean()),
                "always_post": float(test["ev_post"].mean()),
                "always_cross": float(test["ev_cross"].mean()),
                "share_post": float(np.mean(act == "post")),
                "share_cross": float(np.mean(act == "cross")),
                "n": int(len(test)),
            }
        )
    summ: dict[str, Any] = {k: _ci([p[k] for p in per]) for k in ("oos", "net_of_fee", "always_post", "always_cross")}
    summ["share_post"] = float(np.mean([p["share_post"] for p in per]))
    summ["share_cross"] = float(np.mean([p["share_cross"] for p in per]))
    return {"per_day": per, **summ}


def audits(days: list[str]) -> dict:
    """The all-symbol checks of `python -m markout.lob.itch --audit`, for the days that have one."""
    out = {}
    for day in days:
        path = itch.audit_path(day)
        if path.exists():
            a = json.loads(path.read_text())
            out[day] = {
                "symbols": int(a["by_type"].get("R", 0)),
                "references": int(sum(a["by_type"].get(k, 0) for k in "ECXDU")),
                "unmatched": int(sum(a["unmatched"].values())),
                "orders_left_at_end": int(a["orders_left_at_end"]),
                "peak_live_orders": int(a["peak_live_orders"]),
            }
    return out


def pool_latency(cells: list[dict], by_ticker: dict) -> dict:
    """The latency sweep averaged over stock-days, per tick class (each stock keeps the class of most of its days).

    Markouts are pooled as the change from no delay: their levels differ by stock far more than latency moves
    them, and one stock with a coarse tick (SIRI) would set the average level on its own.
    """
    zero = LATENCIES.index(0.0)
    latency: dict[str, Any] = {}
    for cls in ("large-tick", "small-tick", "all"):
        cs = [c for c in cells if cls == "all" or by_ticker[c["ticker"]]["tick_class"] == cls]
        if not cs:
            continue
        latency[cls] = [
            {
                "latency_s": lat,
                "rejected": float(np.mean([c["latency"][k]["rejected"] for c in cs])),
                "p_fill": {m: float(np.mean([c["latency"][k]["p_fill"][m] for c in cs])) for m in LAT_MODELS},
                "mk10_bps": {m: float(np.mean([c["latency"][k]["mk10_bps"][m] for c in cs])) for m in LAT_MODELS},
                "dmk10_bps": {
                    m: float(np.mean([c["latency"][k]["mk10_bps"][m] - c["latency"][zero]["mk10_bps"][m] for c in cs]))
                    for m in LAT_MODELS
                },
            }
            for k, lat in enumerate(LATENCIES)
        ]
    return latency


def run(days: list[str], tickers: tuple[str, ...] = TICKERS) -> dict:
    engine = "cpp" if cpp_available() else "python"
    cells: list[dict] = []
    decs: dict[str, list[pd.DataFrame]] = {t: [] for t in tickers}
    info: dict[str, dict] = {}
    for day in days:
        stats = json.loads((lobster.ITCH_DIR / day / "stats.json").read_text())
        info[day] = {
            "itch_messages": stats["itch_messages"],
            "parse_seconds": stats["seconds"],
            "unmatched": int(sum(sum(s["unmatched"].values()) for s in stats["tickers"].values())),
            "rows": {t: s["rows"] for t, s in stats["tickers"].items()},
        }
        for t in tickers:
            if not lobster.available((t,), day):
                continue
            t0 = time.perf_counter()
            c, dec = cell(day, t, engine)
            cells.append(c)
            decs[t].append(dec)
            print(
                f"  {day} {t:5s} {c['rows']:>9,} rows  touch {c['mk10_bps']['touch']:+.2f} fifo "
                f"{c['mk10_bps']['fifo']:+.2f} bps  ({time.perf_counter() - t0:.0f}s)",
                flush=True,
            )
    by_ticker = {}
    for t in tickers:
        cs = [c for c in cells if c["ticker"] == t]
        if not cs:
            continue
        classes = [c["tick_class"] for c in cs]
        by_ticker[t] = {
            "days": len(cs),
            "tick_class": max(set(classes), key=classes.count),
            "mean_price_usd": float(np.mean([c["mean_price_usd"] for c in cs])),
            "spread_ticks": float(np.mean([c["spread_ticks"] for c in cs])),
            "gap_bps": _ci([c["gap_bps"] for c in cs]),
            "share_days_gap_positive": float(np.mean([c["gap_bps"] > 0 for c in cs])),
            "mk10_bps": {m: _ci([c["mk10_bps"][m] for c in cs]) for m in MODELS},
            "p_fill": {m: float(np.mean([c["p_fill"][m] for c in cs])) for m in MODELS},
        }
    oos = {}
    for t, frames in decs.items():
        if len(frames) >= 2:
            dec = pd.concat(frames, ignore_index=True)
            oos[t] = {m: day_out_of_sample(dec, m) for m in ("touch", "fifo")}
    per_day_gap = [float(np.mean([c["gap_bps"] for c in cells if c["day"] == d])) for d in days]
    rates = {m: [c["engine"]["messages_per_s"][m] for c in cells] for m in MODELS}

    def daily(fn) -> list[float]:  # the cross-stock average of fn(cell), one value per day
        return [
            float(np.mean([fn(c) for c in cells if c["day"] == d])) for d in days if any(c["day"] == d for c in cells)
        ]

    l2 = {
        "by_ticker": {
            t: {
                "dp_fill_pp": _ci([100 * (c["p_fill"]["l2"] - c["p_fill"]["fifo"]) for c in cells if c["ticker"] == t]),
                "dmk10_bps": _ci([c["mk10_bps"]["l2"] - c["mk10_bps"]["fifo"] for c in cells if c["ticker"] == t]),
                "agrees": float(np.mean([c["l2_agrees"] for c in cells if c["ticker"] == t])),
                "l2_only": float(np.mean([c["l2_fills_fifo_doesnt"] for c in cells if c["ticker"] == t])),
                "fifo_only": float(np.mean([c["fifo_fills_l2_doesnt"] for c in cells if c["ticker"] == t])),
            }
            for t in by_ticker
        },
        "pooled": {
            "l2_only": float(np.mean([c["l2_fills_fifo_doesnt"] for c in cells])) if cells else float("nan"),
            "fifo_only": float(np.mean([c["fifo_fills_l2_doesnt"] for c in cells])) if cells else float("nan"),
            "dp_fill_pp": _ci(daily(lambda c: 100 * (c["p_fill"]["l2"] - c["p_fill"]["fifo"]))),
            "abs_dp_fill_pp": _ci(daily(lambda c: 100 * abs(c["p_fill"]["l2"] - c["p_fill"]["fifo"]))),
            "dmk10_bps": _ci(daily(lambda c: c["mk10_bps"]["l2"] - c["mk10_bps"]["fifo"])),
            "agrees": _ci(daily(lambda c: c["l2_agrees"])),
        },
    }
    return {
        "meta": {
            "days": days,
            "tickers": list(tickers),
            "engine": engine,
            "levels": lobster.LEVELS,
            "horizon_s": H,
            "source": "NASDAQ TotalView-ITCH 5.0 sample days (emi.nasdaq.com/ITCH/Nasdaq ITCH/)",
        },
        "days": info,
        "cells": cells,
        "by_ticker": by_ticker,
        "pooled": {
            "gap_bps": _ci(per_day_gap),  # the daily cross-stock average, resampled by day
            "share_cells_gap_positive": float(np.mean([c["gap_bps"] > 0 for c in cells])) if cells else float("nan"),
            "median_stock_gap_bps": float(np.median([b["gap_bps"]["mean"] for b in by_ticker.values()]))
            if by_ticker
            else float("nan"),
            "n_cells": len(cells),
        },
        "oos": oos,
        "l2": l2,
        "latency": pool_latency(cells, by_ticker),
        "throughput": {
            "median_messages_per_s": {m: float(np.median(v)) for m, v in rates.items() if v},
            "min_messages_per_s": {m: float(np.min(v)) for m, v in rates.items() if v},
            "max_messages_per_s": {m: float(np.max(v)) for m, v in rates.items() if v},
            "itch_messages_read": int(sum(i["itch_messages"] for i in info.values())),
            "rows": int(sum(c["rows"] for c in cells)),
        },
    }


# ------------------------------------------------------------------------------------- figures
def draw_heatmap(res: dict) -> Callable:
    days, tickers = res["meta"]["days"], [t for t in res["meta"]["tickers"] if t in res["by_ticker"]]
    g = np.full((len(days), len(tickers)), np.nan)
    for c in res["cells"]:
        g[days.index(c["day"]), tickers.index(c["ticker"])] = c["gap_bps"]
    vals = g[np.isfinite(g)]
    # cap the colors at the 90th percentile so one coarse-tick stock does not wash out the rest; cells beyond the
    # cap keep their printed value
    lim = max(float(np.quantile(np.abs(vals), 0.9)) if vals.size else 1.0, 1e-6)
    over = sorted({tickers[j] for (_, j), v in np.ndenumerate(g) if np.isfinite(v) and abs(v) > lim})
    extend = {(False, False): "neither", (True, False): "max", (False, True): "min", (True, True): "both"}[
        (bool((vals > lim).any()), bool((vals < -lim).any()))
    ]

    def draw():
        fig, ax = mp.subplots(w=7.2, h=0.42 * len(days) + 1.9)
        im = ax.imshow(
            np.ma.masked_invalid(g), cmap=mp.diverging_cmap(), norm=mp.diverging_norm(-lim, lim), aspect="auto"
        )
        ax.grid(False)
        ax.set_xticks(range(len(tickers)), tickers, fontsize=8.5)
        ax.set_yticks(range(len(days)), days, fontsize=8.5)
        for (i, j), v in np.ndenumerate(g):
            if np.isfinite(v):
                strong = mp.theme()["name"] == "light" and abs(v) >= 0.6 * lim
                ax.text(
                    j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7, color="#ffffff" if strong else mp.ink()
                )
        cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02, extend=extend)
        cb.set_label("touch − FIFO, 10-s markout (bps)", color=mp.ink("secondary"))
        cb.outline.set_visible(False)
        note = f"; colors stop at ±{lim:.2f} bps ({_join(over)} beyond)" if over else ""
        mp.title(
            ax, "The touch-fill bias, day by day", f"Positive: a touch-fill backtest flatters resting orders{note}"
        )
        return fig

    return draw


def draw_markouts(res: dict) -> Callable:
    bt = res["by_ticker"]
    tickers = sorted(bt, key=lambda t: (bt[t]["tick_class"] != "large-tick", t))

    def draw():
        fig, ax = mp.subplots(w=7.2, h=4.0)
        x = np.arange(len(tickers))
        for k, m in enumerate(MODELS):
            mean = np.array([bt[t]["mk10_bps"][m]["mean"] for t in tickers])
            lo = np.array(
                [bt[t]["mk10_bps"][m]["lo"] if bt[t]["mk10_bps"][m]["lo"] is not None else np.nan for t in tickers]
            )
            hi = np.array(
                [bt[t]["mk10_bps"][m]["hi"] if bt[t]["mk10_bps"][m]["hi"] is not None else np.nan for t in tickers]
            )
            xs = x + (k - 1) * 0.22
            ax.errorbar(
                xs, mean, yerr=[mean - lo, hi - mean], fmt="none", ecolor=mp.series(k), elinewidth=1.4, capsize=0
            )
            ax.plot(xs, mean, "o", ms=6, color=mp.series(k), mec=mp.surface(), mew=1.5, label=MODEL_LABEL[m], zorder=3)
        mp.zero_line(ax)
        ax.set_xticks(x, [f"{t}\n{bt[t]['tick_class'].replace('-tick', '')}" for t in tickers], fontsize=8)
        ax.set_ylabel("10-s markout after a fill (bps)")
        mp.legend(ax, loc="lower left", fontsize=8.5)
        mp.title(ax, "Markouts by stock, averaged over days", "Bars: 95% intervals that resample whole days")
        return fig

    return draw


def draw_l2_error(res: dict) -> Callable:
    bt, l2 = res["by_ticker"], res["l2"]["by_ticker"]
    tickers = sorted(l2, key=lambda t: (bt[t]["tick_class"] != "large-tick", t))

    def draw():
        fig, axes = mp.subplots(1, 2, w=7.2, h=3.6)
        for ax, key, label in (
            (axes[0], "dp_fill_pp", "L2 − FIFO fill probability (pp)"),
            (axes[1], "dmk10_bps", "L2 − FIFO 10-s markout (bps)"),
        ):
            y = np.arange(len(tickers))
            mean = np.array([l2[t][key]["mean"] for t in tickers])
            lo = np.array([l2[t][key]["lo"] if l2[t][key]["lo"] is not None else np.nan for t in tickers])
            hi = np.array([l2[t][key]["hi"] if l2[t][key]["hi"] is not None else np.nan for t in tickers])
            ax.errorbar(mean, y, xerr=[mean - lo, hi - mean], fmt="none", ecolor=mp.series(0), elinewidth=1.4)
            ax.plot(mean, y, "o", ms=6, color=mp.series(0), mec=mp.surface(), mew=1.5, zorder=3)
            ax.axvline(0, color=mp.baseline_color(), linewidth=0.9, zorder=1)
            ax.set_yticks(y, tickers if ax is axes[0] else [""] * len(tickers), fontsize=8.5)
            ax.invert_yaxis()
            ax.set_xlabel(label, fontsize=9)
        mp.title(axes[0], "What an L2-only backtest gets wrong", "Order-level FIFO is the truth; bars resample days")
        return fig

    return draw


def draw_latency(res: dict) -> Callable:
    lat = res["latency"]
    classes = [c for c in ("large-tick", "small-tick") if c in lat]

    def draw():
        fig, axes = mp.subplots(1, 2, w=7.2, h=3.6)
        for k, cls in enumerate(classes):
            xs = [max(r["latency_s"], 1e-4) for r in lat[cls]]
            axes[0].plot(xs, [r["p_fill"]["fifo"] for r in lat[cls]], "-o", ms=4, lw=2, color=mp.series(k), label=cls)
            axes[1].plot(
                xs, [r["dmk10_bps"]["fifo"] for r in lat[cls]], "-o", ms=4, lw=2, color=mp.series(k), label=cls
            )
        mp.zero_line(axes[1])
        for ax, label in ((axes[0], "P(fill), FIFO"), (axes[1], "change in 10-s markout vs no delay, FIFO (bps)")):
            ax.set_xscale("log")
            ax.set_xlabel("latency (s; 0 drawn at 0.1 ms)", fontsize=9)
            ax.set_ylabel(label, fontsize=9)
        mp.legend(axes[0], fontsize=8.5)
        mp.title(axes[0], "What latency costs a resting order", "Same decisions; the order reaches the book later")
        return fig

    return draw


def figures(res: dict) -> None:
    if res["cells"]:
        mp.render("g_gap_heatmap", draw_heatmap(res))
    if res["by_ticker"]:
        mp.render("g_markouts_by_stock", draw_markouts(res))
        mp.render("i_l2_error", draw_l2_error(res))
        mp.render("i_latency", draw_latency(res))


# ------------------------------------------------------------------------------------- report
def _ci_text(c: dict, fmt: str = "{:+.2f}") -> str:
    if c.get("lo") is None:
        return fmt.format(c["mean"])
    return f"{fmt.format(c['mean'])} [{fmt.format(c['lo'])}, {fmt.format(c['hi'])}]"


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _lo(c: dict) -> float:
    return c["lo"] if c.get("lo") is not None else float("nan")


def _vs_2012(bt: dict) -> str:
    """How these days compare with report 02's single 2012 day (INTC), when that result is on disk."""
    neg = [t for t in bt if bt[t]["mk10_bps"]["touch"]["mean"] < 0]
    m = results.load("microstructure")
    if m is None or "INTC" not in m.get("fills", {}) or "INTC" not in bt or not neg:
        return ""
    ref = {r["model"]: r for r in m["fills"]["INTC"]["summary"]}
    t12, f12 = ref["touch"]["mk_10_bps"], ref["fifo"]["mk_10_bps"]
    return (
        f" On these days a touch-fill backtest already shows a loss after the fill for {len(neg)} of {len(bt)} "
        f"stocks, so the sign flip of report 02 (INTC in 2012: {t12:+.2f} bps under touch fills, {f12:+.2f} under "
        f"FIFO, a gap of {t12 - f12:.2f}; here {bt['INTC']['gap_bps']['mean']:+.2f}) is not the general case. What "
        "holds on every stock-day is the direction: the real queue makes the markout worse."
    )


def _oos_text(oos: dict) -> str:
    """Read the out-of-sample rule table: where does a rule chosen on other days make money?"""
    won = [t for t in oos if _lo(oos[t]["fifo"]["net_of_fee"]) > 0]
    touch_only = [t for t in oos if _lo(oos[t]["touch"]["net_of_fee"]) > 0 and t not in won]
    acts = {t: oos[t]["fifo"]["share_post"] + oos[t]["fifo"]["share_cross"] for t in oos}
    idle = [t for t in oos if acts[t] < 0.01]
    busy = [t for t in oos if t not in idle]
    parts = [
        (
            f"Under FIFO fills the rule makes money out of sample, net of the fee, for {', '.join(won)}."
            if won
            else "Under FIFO fills no stock has a rule that makes money out of sample net of the fee."
        )
    ]
    if idle:
        parts.append(
            f"For {len(idle)} of {len(oos)} stocks the rule learned on the other days acts on fewer than 1% of "
            "decisions: both posting and crossing lose on average, so the best rule is to stay out."
        )
    if busy:
        parts.append(
            "Where it does trade ("
            + _join([f"{t} {100 * acts[t]:.0f}%" for t in busy])
            + " of decisions), it earns between "
            + f"{min(oos[t]['fifo']['oos']['mean'] for t in busy):+.3f} and "
            + f"{max(oos[t]['fifo']['oos']['mean'] for t in busy):+.3f} ticks per decision before the fee"
            + (" and loses after it." if all(oos[t]["fifo"]["net_of_fee"]["mean"] < 0 for t in busy) else ".")
        )
    for t in touch_only:
        tr = oos[t]["touch"]
        parts.append(
            f"Touch fills invent a profitable rule for {t}: it posts on {100 * tr['share_post']:.0f}% of decisions "
            f"and earns {_ci_text(tr['net_of_fee'], '{:+.3f}')} ticks per decision out of sample. Under the real "
            f"queue the rule learned on the same days posts on {100 * oos[t]['fifo']['share_post']:.0f}%."
        )
    return " ".join(parts)


def _audit_text(aud: dict, days: list[str]) -> str:
    if not aud or set(aud) != set(days):
        return ""
    refs, bad = sum(a["references"] for a in aud.values()), sum(a["unmatched"] for a in aud.values())
    left = sum(a["orders_left_at_end"] for a in aud.values())
    sym = [a["symbols"] for a in aud.values()]
    return (
        f" Run over every stock in the files (`--audit`), the same check covers {refs:,} references to earlier "
        f"orders across {min(sym):,} to {max(sym):,} symbols a day: {bad} unmatched"
        + (
            ", and no order is left on the book at the end of any day."
            if left == 0
            else f"; {left:,} orders are still on the book at the end of the day."
        )
    )


def write(res: dict) -> str:
    meta, bt, pool, thr = res["meta"], res["by_ticker"], res["pooled"], res["throughput"]
    days, n_days, n_stocks = meta["days"], len(meta["days"]), len(bt)
    unmatched = sum(i["unmatched"] for i in res["days"].values())
    big = [t for t in bt if bt[t]["tick_class"] == "large-tick"]
    small = [t for t in bt if bt[t]["tick_class"] != "large-tick"]
    mixed = {}  # stocks whose tick class changes from day to day (their price moved a lot)
    for t in bt:
        cls = [c["tick_class"] for c in res["cells"] if c["ticker"] == t]
        if len(set(cls)) > 1:
            mixed[t] = (cls.count(bt[t]["tick_class"]), len(cls))
    lines = [
        "# 07 · Many days: does the fill gap hold?",
        "",
        (
            f"*Generated by `python -m markout.lob.multiday` from {n_days} NASDAQ ITCH sample days "
            f"({days[0]} to {days[-1]}) and {n_stocks} stocks.*"
            if days
            else "*No converted ITCH days yet.*"
        ),
        "",
        "## The question",
        "",
        "Report 02 found, on one day of 2012 LOBSTER data, that a backtest which fills a resting order whenever the "
        "market touches its price credits it with a positive markout, while simulating its place in the queue turns "
        "that negative. One day and five stocks cannot say whether that is a property of limit orders or of that "
        "day. Here the same experiment runs on every NASDAQ ITCH sample day available, for ten stocks.",
        "",
    ]
    if not res["cells"]:
        return "\n".join(lines) + "\n"
    lines += [
        (
            f"**Answer.** Across {pool['n_cells']} stock-days, the touch-fill markout ten seconds after a fill is "
            f"higher than the FIFO-queue markout on {100 * pool['share_cells_gap_positive']:.0f}% of them. The "
            f"median stock's gap is {pool['median_stock_gap_bps']:+.2f} bps; averaged across the stocks each day it "
            f"is {_ci_text(pool['gap_bps'])} bps (mean, and a 95% interval that resamples days)"
            + (", pulled up by SIRI, a $6 stock whose one-cent tick is worth about 16 bps." if "SIRI" in bt else ".")
            + _vs_2012(bt)
        ),
        "",
        "## By stock",
        "",
        R.table(
            [
                {
                    "stock": t,
                    "tick": bt[t]["tick_class"].replace("-tick", ""),
                    "price $": bt[t]["mean_price_usd"],
                    "spread (ticks)": bt[t]["spread_ticks"],
                    "days": bt[t]["days"],
                    "P(fill) touch": bt[t]["p_fill"]["touch"],
                    "P(fill) FIFO": bt[t]["p_fill"]["fifo"],
                    "touch mk10 (bps)": bt[t]["mk10_bps"]["touch"]["mean"],
                    "FIFO mk10 (bps)": bt[t]["mk10_bps"]["fifo"]["mean"],
                    "gap (bps), 95% CI": _ci_text(bt[t]["gap_bps"]),
                    "days gap > 0": f"{100 * bt[t]['share_days_gap_positive']:.0f}%",
                }
                for t in sorted(bt, key=lambda t: (bt[t]["tick_class"] != "large-tick", t))
            ],
            formats={
                "price $": "{:,.0f}",
                "spread (ticks)": "{:.2f}",
                "P(fill) touch": "{:.2f}",
                "P(fill) FIFO": "{:.2f}",
                "touch mk10 (bps)": "{:+.2f}",
                "FIFO mk10 (bps)": "{:+.2f}",
            },
        ),
        "",
        mp.picture("g_gap_heatmap", "Touch-minus-FIFO markout gap for every day and stock"),
        "",
        mp.picture("g_markouts_by_stock", "Average 10-second markouts by stock and fill model"),
        "",
        (
            f"Large-tick stocks ({', '.join(big) or 'none'}) have long queues at the best price, so the queue decides "
            f"which orders fill; small-tick stocks ({', '.join(small) or 'none'}) quote wider spreads in ticks and "
            "thinner queues."
            + (
                " Some stocks sit near the line between the two: "
                + _join([f"{t} is {bt[t]['tick_class']} on {k} of its {n} days" for t, (k, n) in mixed.items()])
                + ". Each is pooled with the class of most of its days."
                if mixed
                else ""
            )
        ),
        "",
        "## The post-or-cross rule, out of sample by day",
        "",
        "Report 02 chose the rule on a morning and scored it on the same afternoon. Here it is chosen on every other "
        "day and scored on the day left out, in ticks per share per decision (crossing pays the 0.3-tick access-fee "
        "cap in the net column).",
        "",
    ]
    if res["oos"]:
        lines += [
            R.table(
                [
                    {
                        "stock": t,
                        "model": MODEL_LABEL[m],
                        "rule, out of sample": _ci_text(res["oos"][t][m]["oos"], "{:+.3f}"),
                        "net of the fee": _ci_text(res["oos"][t][m]["net_of_fee"], "{:+.3f}"),
                        "always post": res["oos"][t][m]["always_post"]["mean"],
                        "always cross": res["oos"][t][m]["always_cross"]["mean"],
                        "posts": f"{100 * res['oos'][t][m]['share_post']:.0f}%",
                        "crosses": f"{100 * res['oos'][t][m]['share_cross']:.0f}%",
                    }
                    for t in res["oos"]
                    for m in ("touch", "fifo")
                ],
                formats={"always post": "{:+.3f}", "always cross": "{:+.3f}"},
            ),
            "",
            _oos_text(res["oos"]),
            "",
        ]
    else:
        lines += ["Needs at least two days.", ""]
    mps = thr["median_messages_per_s"]
    lines += [
        "## The data and the engine",
        "",
        (
            f"NASDAQ publishes a few full days of its TotalView-ITCH 5.0 feed as samples. `markout.lob.itch` streams "
            f"each one ({thr['itch_messages_read'] / 1e9:.2f} billion messages in all) and rebuilds the book of every "
            f"stock above from the first order of the day. On every day, {'every' if unmatched == 0 else 'not every'} "
            f"execution, cancel, delete and replace matched an order added earlier "
            f"({unmatched} unmatched), and the books pass the same replay check as the LOBSTER sample. The converted "
            f"days hold {thr['rows']:,} order-book events in the regular session."
            + _audit_text(res.get("audit", {}), days)
        ),
        "",
        R.table(
            [
                {
                    "day": d,
                    "ITCH messages": i["itch_messages"],
                    "parse (s)": i["parse_seconds"],
                    "unmatched, 10 stocks": i["unmatched"],
                    "rows, 10 stocks": sum(i["rows"].values()),
                    **(
                        {
                            "symbols": res["audit"][d]["symbols"],
                            "references, all symbols": res["audit"][d]["references"],
                            "unmatched, all symbols": res["audit"][d]["unmatched"],
                        }
                        if d in res.get("audit", {})
                        else {}
                    ),
                }
                for d, i in res["days"].items()
            ],
            formats={
                "ITCH messages": "{:,.0f}",
                "parse (s)": "{:.0f}",
                "rows, 10 stocks": "{:,.0f}",
                "symbols": "{:,.0f}",
                "references, all symbols": "{:,.0f}",
            },
        ),
        "",
        (
            "The C++ engine replays a full stock-day at a median of "
            + ", ".join(f"{mps[m] / 1e6:.0f}M ({MODEL_LABEL[m].lower()})" for m in MODELS if m in mps)
            + " messages per second, on a laptop."
        ),
        "",
        "## Limitations",
        "",
        R.bullets(
            [
                "These are NASDAQ's own sample days, not a random sample, all from the 2019-2021 penny-tick "
                "regime, and they show NASDAQ's book only, not the consolidated market.",
                "The hypothetical orders have no market impact, as in report 02.",
                "Hidden orders are invisible until they trade, so their queue position is not modelled.",
                "The day-level intervals rest on few days; they show the spread of results across days, not a "
                "population estimate.",
            ]
        ),
    ]
    return "\n".join(lines) + "\n"


def _size(c: dict) -> dict:
    """A signed mean and interval as a magnitude (for "overstates by" / "understates by")."""
    if c["mean"] >= 0:
        return c
    lo, hi = c.get("lo"), c.get("hi")
    return {**c, "mean": -c["mean"], "lo": None if hi is None else -hi, "hi": None if lo is None else -lo}


def _answer_l2(res: dict) -> str:
    pool, lat = res["l2"]["pooled"], res["latency"].get("all", [])
    dp = pool["dp_fill_pp"]["mean"]
    by = {r["latency_s"]: r["p_fill"]["fifo"] for r in lat}
    cost = (
        f" Latency matters less than queue position: compared with no delay, 20 ms costs "
        f"{100 * (by[0.0] - by[0.02]):.1f} pp of fill probability and 500 ms costs {100 * (by[0.0] - by[0.5]):.1f} pp, "
        "mostly through orders rejected because the market moved before they arrived."
        if {0.0, 0.02, 0.5} <= set(by)
        else ""
    )
    return (
        f"**Answer.** An L2-only backtest is {'optimistic' if dp > 0 else 'pessimistic'}: it "
        f"{'overstates' if dp > 0 else 'understates'} the fill probability by "
        f"{_ci_text(_size(pool['dp_fill_pp']), '{:.1f}')} percentage points against the order-level answer. "
        f"{100 * pool['l2_only']:.1f}% of orders fill under the L2 estimate but not in the queue, against "
        f"{100 * pool['fifo_only']:.1f}% the other way round; the two agree on whether an order fills for "
        f"{100 * pool['agrees']['mean']:.1f}% of orders." + cost
    )


def write_l2(res: dict) -> str:
    """Report 08: how wrong an L2-only backtest is, and what latency costs, on the same days."""
    meta, bt, l2, lat = res["meta"], res["by_ticker"], res["l2"], res["latency"]
    days = meta["days"]
    pool = l2["pooled"]
    lines = [
        "# 08 · How wrong is an L2 backtest, and what does latency cost?",
        "",
        (
            f"*Generated by `python -m markout.lob.multiday` from {len(days)} NASDAQ ITCH sample days "
            f"({days[0]} to {days[-1]}) and {len(bt)} stocks, the data of report 07.*"
        ),
        "",
        _answer_l2(res),
        "",
        "## The question",
        "",
        "Order-level data (every order id, as in ITCH or LOBSTER) lets a backtest know exactly where a resting order "
        "sits in the queue. Most backtests only have L2 data, the size at each price and the trades, and have to "
        "guess: when the size at our price drops without a trade, was the cancelled order ahead of us or behind? "
        "The usual guess is pro rata. Here that L2 estimate runs next to the order-level FIFO model on the same "
        "orders, so its errors can be measured directly. The same grid then runs with latency between the decision "
        "and the order reaching the book.",
        "",
        "## An L2-only backtest",
        "",
        (
            f"Averaged over stocks and days, the L2 fill probability minus the order-level one is "
            f"{_ci_text(pool['dp_fill_pp'], '{:+.1f}')} percentage points (absolute error "
            f"{_ci_text(pool['abs_dp_fill_pp'], '{:.1f}')} pp), and the 10-s markout of its fills differs by "
            f"{_ci_text(pool['dmk10_bps'])} bps"
            + (
                f" ({np.median([l2['by_ticker'][t]['dmk10_bps']['mean'] for t in l2['by_ticker']]):+.2f} for the "
                "median stock; SIRI's coarse tick dominates the average)"
                if "SIRI" in l2["by_ticker"]
                else ""
            )
            + f". It agrees with FIFO on whether an order fills at all for {100 * pool['agrees']['mean']:.1f}% of "
            "orders."
        ),
        "",
        R.table(
            [
                {
                    "stock": t,
                    "tick": bt[t]["tick_class"].replace("-tick", ""),
                    "P(fill) FIFO": bt[t]["p_fill"]["fifo"],
                    "L2 − FIFO, pp (95% CI)": _ci_text(l2["by_ticker"][t]["dp_fill_pp"], "{:+.1f}"),
                    "markout L2 − FIFO, bps": _ci_text(l2["by_ticker"][t]["dmk10_bps"]),
                    "same outcome": f"{100 * l2['by_ticker'][t]['agrees']:.1f}%",
                    "L2 fills, FIFO doesn't": f"{100 * l2['by_ticker'][t]['l2_only']:.1f}%",
                    "FIFO fills, L2 doesn't": f"{100 * l2['by_ticker'][t]['fifo_only']:.1f}%",
                }
                for t in sorted(l2["by_ticker"], key=lambda t: (bt[t]["tick_class"] != "large-tick", t))
            ],
            formats={"P(fill) FIFO": "{:.2f}"},
        ),
        "",
        mp.picture("i_l2_error", "Fill-probability and markout errors of an L2-only backtest, by stock"),
        "",
        "## Latency",
        "",
        "The same decisions every 5 s, with the order reaching the book later: it joins the queue behind whatever "
        "arrived in the meantime, or is rejected if the market has moved through its price. Each stock is pooled "
        "with the tick class of most of its days, and markouts are shown as the change from no delay (their levels, "
        "by stock, are in report 07).",
        "",
    ]
    for cls in ("large-tick", "small-tick"):
        if cls not in lat:
            continue
        rows = lat[cls]
        lines += [
            f"**{cls.capitalize()} stocks.**",
            "",
            R.table(
                [
                    {
                        "latency": f"{1e3 * r['latency_s']:g} ms",
                        "rejected": f"{100 * r['rejected']:.1f}%",
                        "P(fill) FIFO": r["p_fill"]["fifo"],
                        "P(fill) touch": r["p_fill"]["touch"],
                        "markout change, FIFO (bps)": r["dmk10_bps"]["fifo"],
                        "markout change, touch (bps)": r["dmk10_bps"]["touch"],
                    }
                    for r in rows
                ],
                formats={
                    "P(fill) FIFO": "{:.3f}",
                    "P(fill) touch": "{:.3f}",
                    "markout change, FIFO (bps)": "{:+.3f}",
                    "markout change, touch (bps)": "{:+.3f}",
                },
            ),
            "",
        ]
    lines += [
        mp.picture("i_latency", "Fill probability and markout against latency, large- and small-tick stocks"),
        "",
        "## Limitations",
        "",
        R.bullets(
            [
                "Pro rata is one L2 assumption among several; backtesters also use pessimistic (cancels always "
                "behind) or probabilistic queue models, which would land elsewhere.",
                "Latency here delays placement and cancellation equally and leaves the decision's information "
                "unchanged; it does not model a faster rival reacting to the same signal.",
                "The orders have no market impact, and hidden liquidity is invisible until it trades.",
            ]
        ),
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m markout.lob.multiday", description=__doc__.split("\n\n")[0])
    ap.add_argument("--days", nargs="*", help="default: every converted ITCH day")
    ap.add_argument("--tickers", default=",".join(TICKERS))
    ap.add_argument(
        "--from-results", action="store_true", help=f"rewrite the reports from results/{NAME}.json without replaying"
    )
    a = ap.parse_args(argv)
    if a.from_results:
        saved = results.load(NAME)
        if saved is None:
            raise SystemExit("no saved results: run `python -m markout.lob.multiday` first")
        res = {**saved, "latency": pool_latency(saved["cells"], saved["by_ticker"])}
    else:
        days = a.days or lobster.itch_days()
        if not days:
            raise SystemExit("no converted ITCH days: run `python -m markout.lob.itch <file>` first")
        res = run(days, tuple(t.strip().upper() for t in a.tickers.split(",") if t.strip()))
    res["audit"] = audits(res["meta"]["days"])
    figures(res)
    results.save(NAME, res)
    R.write("07_many_days", write(res))
    R.write("08_l2_and_latency", write_l2(res))
    print("wrote reports/07_many_days.md and reports/08_l2_and_latency.md")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
