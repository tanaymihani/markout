"""Figures for report 02 (house style: `markout.plotting`). Each `draw_*` returns a
zero-argument function for `mp.render`, so every figure exists in light and dark."""

from __future__ import annotations

from typing import Callable

import numpy as np

from markout import plotting as mp
from markout.lob.fills import MODELS
from markout.lob.markouts import HORIZONS, QUEUE_LABELS
from markout.lob.postcross import SIGNAL_LABELS

MODEL_LABEL = {"touch": "touch (optimistic)", "fifo": "FIFO queue (realistic)",
               "through": "trade-through (pessimistic)"}
CLASS_ORDER = ("large-tick", "small-tick")


def _ticker_colors(tickers: list[str]) -> dict[str, str]:
    return {t: mp.series(k) for k, t in enumerate(tickers)}


def draw_tick_regimes(res: dict) -> Callable:
    tickers = list(res["facts"])

    def draw():
        fig, axes = mp.subplots(1, 3, w=7.2, h=3.0)
        facts = res["facts"]
        panels = [("tick_bps", "One tick", "bps of the price", False),
                  ("spread_ticks", "Mean quoted spread", "ticks, time-weighted", True),
                  ("depth_best", "Mean depth at the best", "shares, time-weighted", True)]
        for ax, (key, title, sub, log) in zip(axes, panels):
            for k, t in enumerate(tickers):
                cls = facts[t]["tick_class"]
                mp.bars(ax, [k], [facts[t][key]], color=mp.series(CLASS_ORDER.index(cls)))
            ax.set_xticks(range(len(tickers)), tickers, fontsize=8)
            if log:
                ax.set_yscale("log")
            mp.title(ax, title, sub)
        handles = [mp.plt.Rectangle((0, 0), 1, 1, color=mp.series(i)) for i in range(2)]
        axes[0].legend(handles, CLASS_ORDER, loc="upper left")
        return fig

    return draw


def draw_sign_acf(res: dict) -> Callable:
    tickers = list(res["facts"])

    def draw():
        fig, ax = mp.subplots(w=7.2, h=3.4)
        col = _ticker_colors(tickers)
        for t in tickers:
            acf = res["facts"][t]["sign_acf"]
            ax.plot(range(1, len(acf) + 1), acf, marker="o", color=col[t], label=t)
        ax.set_xlabel("lag (trades)")
        ax.set_ylim(bottom=0)
        mp.legend(ax, ncols=5, loc="upper right")
        mp.title(ax, "Trade signs are persistent",
                 "autocorrelation of trade signs (buyer +1, seller -1), visible executions grouped per aggressive order")
        return fig

    return draw


def draw_cks_scatter(scatter: dict[str, dict], res: dict) -> Callable:
    names = list(scatter)

    def draw():
        fig, axes = mp.subplots(1, len(names), w=7.2, h=3.4)
        for k, (ax, t) in enumerate(zip(np.atleast_1d(axes), names)):
            x, y = scatter[t]["ofi"], scatter[t]["dmid"]
            ax.scatter(x / 1000, y, s=5, alpha=0.35, color=mp.series(k), linewidths=0,
                       label="10 s buckets")
            beta = res["kyle_bridge"][t]["beta_ticks_per_1000_shares"]
            xs = np.linspace(np.quantile(x, 0.001), np.quantile(x, 0.999), 50) / 1000
            ax.plot(xs, beta * xs, color=mp.ink("secondary"), lw=1.2,
                    label=f"OLS: {beta:.3g} ticks per 1,000 sh")
            ax.set_xlim(xs[0], xs[-1])
            ax.set_xlabel("OFI (thousand shares)")
            if k == 0:
                ax.set_ylabel("mid change (ticks)")
            mp.legend(ax, loc="lower right")
            cls = res["facts"][t]["tick_class"]
            mp.title(ax, f"{t} ({cls})",
                     f"same-bucket fit, R² = {res['kyle_bridge'][t]['r2']:.2f} over the day")
        return fig

    return draw


def draw_beta_depth(res: dict) -> Callable:
    tickers = list(res["facts"])
    slope = res["cks"]["depth_slope"]

    def draw():
        fig, ax = mp.subplots(w=7.2, h=3.8)
        col = _ticker_colors(tickers)
        xs, ys = [], []
        for t in tickers:
            w = res["cks"]["tickers"][t]["windows"]
            d = np.array([r["depth"] for r in w])
            b = np.array([r["beta"] for r in w])
            ok = b > 0
            ax.scatter(d[ok], b[ok], color=col[t], label=t, s=28, zorder=3,
                       edgecolors=mp.surface(), linewidths=0.8)
            xs += list(np.log(d[ok]))
            ys += list(np.log(b[ok]))
        grid = np.linspace(min(xs), max(xs), 50)
        ax.plot(np.exp(grid), np.exp(slope["intercept"] + slope["slope"] * grid),
                color=mp.ink("secondary"), lw=1.2,
                label=f"pooled fit, slope {slope['slope']:.2f}")
        cx, cy = np.mean(xs), np.mean(ys)
        ax.plot(np.exp(grid), np.exp(cy - (grid - cx)), color=mp.ink("muted"), lw=1.0,
                ls="--", label="CKS prediction, slope −1")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("mean depth at the best in the window (shares)")
        ax.set_ylabel("β (ticks per share of OFI)")
        mp.legend(ax, loc="upper right", ncols=2)
        mp.title(ax, "Price impact of order flow falls with depth",
                 "CKS β per 30-minute window, all five stocks (log-log)")
        return fig

    return draw


def draw_contemp_vs_lagged(res: dict) -> Callable:
    tickers = list(res["facts"])

    def draw():
        fig, axes = mp.subplots(1, 2, w=7.2, h=3.4)
        x = np.arange(len(tickers))
        c = [res["predictive"][t]["contemporaneous"]["r2_oos"] for t in tickers]
        lag = [res["predictive"][t]["lagged"]["r2_oos"] for t in tickers]
        mp.bars(axes[0], x - 0.19, c, width=0.36, color=mp.series(0), label="same bucket (contemporaneous)")
        mp.bars(axes[0], x + 0.19, lag, width=0.36, color=mp.series(1), label="next bucket (predictive)")
        for xi, v in zip(x + 0.19, lag):
            axes[0].annotate(f"{v:+.3f}", (xi, max(v, 0)), xytext=(0, 3), textcoords="offset points",
                             ha="center", va="bottom", fontsize=7, rotation=90,
                             color=mp.ink("secondary"))
        axes[0].set_xticks(x, tickers)
        mp.zero_line(axes[0])
        mp.legend(axes[0], loc="upper left")
        mp.title(axes[0], "OFI explains, but does not forecast",
                 "out-of-sample R² of mid change on OFI, fit before 12:45")
        auc = [res["predictive"][t]["imbalance"]["auc"] for t in tickers]
        for k, t in enumerate(tickers):
            cls = res["facts"][t]["tick_class"]
            mp.bars(axes[1], [k], [auc[k] - 0.5], color=mp.series(CLASS_ORDER.index(cls)))
        axes[1].set_xticks(x, tickers)
        axes[1].set_ylabel("AUC − 0.5")
        handles = [mp.plt.Rectangle((0, 0), 1, 1, color=mp.series(i)) for i in range(2)]
        axes[1].legend(handles, CLASS_ORDER, loc="upper left")
        mp.title(axes[1], "Queue imbalance forecasts the next move",
                 "out-of-sample AUC above a coin flip, after 12:45")
        return fig

    return draw


def draw_imbalance_curve(res: dict) -> Callable:
    tickers = list(res["facts"])

    def draw():
        fig, ax = mp.subplots(w=7.2, h=3.8)
        col = _ticker_colors(tickers)
        for t in tickers:
            imb = res["predictive"][t]["imbalance"]
            cur = [c for c in imb["curve"] if c["p_up"] is not None and c["n"] >= 30]
            mids = [(c["lo"] + c["hi"]) / 2 for c in cur]
            ax.plot(mids, [c["p_up"] for c in cur], marker="o", color=col[t],
                    label=f"{t}  AUC {imb['auc']:.2f}")
        ax.axhline(0.5, color=mp.baseline_color(), lw=0.9, zorder=1)
        ax.set_xlabel("queue imbalance I = (qB − qA)/(qB + qA)")
        ax.set_ylabel("P(next mid move is up)")
        mp.legend(ax, loc="upper left")
        mp.title(ax, "Long bid queue, short ask queue: the price goes up next",
                 "test set (after 12:45), 1 s samples, bins with at least 30 samples")
        return fig

    return draw


def draw_fill_prob(res: dict) -> Callable:
    tickers = list(res["facts"])

    def draw():
        fig, ax = mp.subplots(w=7.2, h=3.4)
        x = np.arange(len(tickers))
        width = 0.26
        for k, model in enumerate(MODELS):
            p = [next(r for r in res["fills"][t]["summary"] if r["model"] == model)["p_fill"]
                 for t in tickers]
            mp.bars(ax, x + (k - 1) * width, p, width=width, color=mp.series(k),
                    label=MODEL_LABEL[model])
        ax.set_xticks(x, tickers)
        ax.set_ylim(0, 1)
        mp.legend(ax, loc="upper left", ncols=3)
        mp.title(ax, "How often a passive order at the best gets filled",
                 "100 shares at the best bid or ask every 5 s, cancelled after 60 s")
        return fig

    return draw


def draw_markouts(res: dict) -> Callable:
    tickers = list(res["facts"])
    order = sorted(tickers, key=lambda t: (CLASS_ORDER.index(res["facts"][t]["tick_class"]), t))

    def draw():
        fig, axes = mp.subplots(2, 3, w=7.2, h=5.4)
        flat = axes.ravel()
        hs = np.array(HORIZONS)
        for ax, t in zip(flat, order):
            rows = {r["model"]: r for r in res["fills"][t]["summary"]}
            for k, model in enumerate(MODELS):
                ax.plot(hs, [rows[model][f"mk_{h:g}"] for h in HORIZONS], marker="o",
                        color=mp.series(k), label=MODEL_LABEL[model])
            ax.plot(hs, [rows["fifo"][f"uncond_{h:g}"] for h in HORIZONS], color=mp.ink("muted"),
                    ls="--", lw=1.2, label="unconditional (random fill)")
            mp.zero_line(ax)
            ax.set_xscale("log")
            ax.set_xticks(hs, ["0.1", "1", "10", "60"])
            ax.minorticks_off()
            mp.title(ax, t, res["facts"][t]["tick_class"])
        for ax in flat[len(order):]:
            ax.axis("off")
        handles, labels = flat[0].get_legend_handles_labels()
        flat[-1].legend(handles, labels, loc="center", fontsize=9)
        for ax in axes[1]:
            ax.set_xlabel("seconds after the fill")
        axes[0, 0].set_ylabel("markout (ticks per share)")
        axes[1, 0].set_ylabel("markout (ticks per share)")
        fig.suptitle("Filled passive orders lose to the mid: adverse selection by fill model",
                     x=0.01, ha="left", fontsize=11.5, fontweight="semibold", color=mp.ink())
        return fig

    return draw


def draw_queue_buckets(res: dict) -> Callable:
    tickers = list(res["facts"])

    def draw():
        fig, axes = mp.subplots(1, 2, w=7.2, h=3.4)
        col = _ticker_colors(tickers)
        x = np.arange(len(QUEUE_LABELS))
        for t in tickers:
            rows = [r for r in res["fills"][t]["by_queue"] if r["model"] == "fifo"]
            rows = {r["queue_bucket"]: r for r in rows}
            ks = sorted(rows)
            axes[0].plot(ks, [rows[k]["p_fill"] for k in ks], marker="o", color=col[t], label=t)
            axes[1].plot(ks, [rows[k]["mk"] for k in ks], marker="o", color=col[t], label=t)
        for ax in axes:
            ax.set_xticks(x, QUEUE_LABELS)
            ax.set_xlabel("queue ahead at placement (× ticker's median)")
        axes[0].set_ylim(0, 1)
        mp.zero_line(axes[1])
        mp.legend(axes[0], loc="lower left", ncols=2)
        mp.title(axes[0], "P(fill), FIFO model", "within 60 s")
        mp.title(axes[1], "10 s markout of filled orders", "ticks per share, FIFO model")
        return fig

    return draw


def draw_spread_decomp(res: dict) -> Callable:
    tickers = list(res["facts"])

    def draw():
        rows0 = res["spread_decomposition"][tickers[0]]
        fig, axes = mp.subplots(1, len(rows0), w=7.2, h=3.4, sharey=True)
        x = np.arange(len(tickers))
        parts = [("effective", "effective spread"), ("realized", "realized spread"),
                 ("impact", "price impact")]
        for j, ax in enumerate(np.atleast_1d(axes)):
            h = rows0[j]["horizon"]
            for k, (key, label) in enumerate(parts):
                v = [res["spread_decomposition"][t][j][f"{key}_bps"] for t in tickers]
                mp.bars(ax, x + (k - 1) * 0.26, v, width=0.26, color=mp.series(k), label=label)
            ax.set_xticks(x, tickers)
            mp.zero_line(ax)
            mp.title(ax, f"Horizon {h:g} s", "bps, share-weighted, effective = realized + impact")
        axes[0].set_ylabel("bps")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncols=3)
        return fig

    return draw


def draw_post_vs_cross(res: dict, tickers: list[str], title: str) -> Callable:
    grids = {t: res["post_vs_cross"][t]["grid"] for t in tickers}
    vals = np.concatenate([np.ravel(np.array(grids[t][m], dtype=float)) for t in tickers
                           for m in MODELS])
    vals = vals[np.isfinite(vals)]
    # symmetric around zero; the 95th percentile keeps one extreme cell from washing
    # out the rest (the printed numbers show the exact values)
    lim = max(float(np.quantile(np.abs(vals), 0.95)), 1e-6) if len(vals) else 1.0
    vmin, vmax = -lim, lim

    def draw():
        fig, axes = mp.subplots(len(tickers), len(MODELS), w=7.2, h=1.9 * len(tickers) + 0.9,
                                squeeze=False)
        cmap, norm = mp.diverging_cmap(), mp.diverging_norm(vmin, vmax)
        im = None
        for r, t in enumerate(tickers):
            for c, model in enumerate(MODELS):
                ax = axes[r, c]
                g = np.array(grids[t][model], dtype=float)
                im = ax.imshow(np.ma.masked_invalid(g), cmap=cmap, norm=norm, origin="lower",
                               aspect="auto")
                ax.grid(False)
                ax.set_xticks(range(len(SIGNAL_LABELS)),
                              SIGNAL_LABELS if r == len(tickers) - 1 else [""] * len(SIGNAL_LABELS),
                              fontsize=7.5)
                ax.set_yticks(range(len(QUEUE_LABELS)),
                              QUEUE_LABELS if c == 0 else [""] * len(QUEUE_LABELS), fontsize=7.5)
                if r == 0:
                    ax.set_title(MODEL_LABEL[model], fontsize=9.5, loc="left")
                if c == 0:
                    ax.set_ylabel(f"{t}\nqueue ahead", fontsize=9)
                for (i, j), v in np.ndenumerate(g):
                    if np.isfinite(v):
                        ax.text(j, i, f"{v:+.2f}" if abs(v) < 10 else f"{v:+.1f}",
                                ha="center", va="center", fontsize=6.5, color=mp.ink())
        for ax in axes[-1]:
            ax.set_xlabel("signal strength |I|", fontsize=9)
        cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
        cb.set_label("EV(post) − EV(cross), ticks per share", color=mp.ink("secondary"))
        cb.outline.set_visible(False)
        fig.suptitle(title, x=0.01, ha="left", fontsize=11.5, fontweight="semibold",
                     color=mp.ink())
        return fig

    return draw


def draw_rule_oos(res: dict) -> Callable:
    tickers = list(res["facts"])

    def draw():
        fig, axes = mp.subplots(1, len(MODELS), w=7.2, h=3.4, sharey=True)
        x = np.arange(len(tickers))
        for ax, model in zip(axes, MODELS):
            ins = [res["post_vs_cross"][t]["rules"][model]["in_sample"] for t in tickers]
            oos = [res["post_vs_cross"][t]["rules"][model]["out_of_sample"] for t in tickers]
            lo = [res["post_vs_cross"][t]["rules"][model]["ci"][0] for t in tickers]
            hi = [res["post_vs_cross"][t]["rules"][model]["ci"][1] for t in tickers]
            mp.bars(ax, x - 0.19, ins, width=0.36, color=mp.series(0), label="chosen and scored before 12:45")
            mp.bars(ax, x + 0.19, oos, width=0.36, color=mp.series(1), label="same rule, after 12:45")
            ax.errorbar(x + 0.19, oos, yerr=[np.subtract(oos, lo), np.subtract(hi, oos)], fmt="none",
                        ecolor=mp.ink("secondary"), elinewidth=1, capsize=2, zorder=4)
            ax.set_xticks(x, tickers, fontsize=8)
            mp.zero_line(ax)
            mp.title(ax, MODEL_LABEL[model].split(" (")[0], "EV per decision, ticks")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncols=2)
        fig.suptitle("The best cell-by-cell rule looks better in the sample that chose it",
                     x=0.01, ha="left", fontsize=11.5, fontweight="semibold", color=mp.ink())
        return fig

    return draw
