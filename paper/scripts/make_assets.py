"""Build the paper's tables (LaTeX) and figures (PDF) from saved results.

Inputs:  reports/results/microstructure.json  (written by `python -m markout.lob.report`)
         reports/results/fill_rebates.json     (written by paper/scripts/rebates.py)
         reports/results/fill_robustness.json  (written by paper/scripts/robustness.py)
Outputs: paper/generated/tab_*.tex, paper/figures/fig_*.pdf (+ PNG previews in paper/generated/)

Run from the repository root:  .venv/bin/python paper/scripts/make_assets.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from markout import results  # noqa: E402
from markout.paths import RESULTS  # noqa: E402

PAPER = Path(__file__).resolve().parents[1]
GEN, FIG = PAPER / "generated", PAPER / "figures"
for d in (GEN, FIG):
    d.mkdir(parents=True, exist_ok=True)
LARGE, SMALL = ("INTC", "MSFT"), ("AAPL", "AMZN", "GOOG")
ORDER = LARGE + SMALL
MODELS = ("touch", "fifo", "through")
NAME = {"touch": "Touch", "fifo": "FIFO", "through": "Trade-through"}
# house palette (light), fixed slot order; line style and marker repeat the encoding for print
COLOR = {"touch": "#2a78d6", "fifo": "#eb6834", "through": "#1baf7a"}
STYLE = {"touch": "-", "fifo": "--", "through": ":"}
MARK = {"touch": "o", "fifo": "s", "through": "^"}
INK, INK2, GRID, BASE = "#0b0b0b", "#52514e", "#e1e0d9", "#9a998f"

m = results.load("microstructure")
rb = json.loads((RESULTS / "fill_rebates.json").read_text())


def rule(ticker: str, model: str, rebate):
    return next(r for r in rb if r["ticker"] == ticker and r["model"] == model and r["rebate_ticks"] == rebate)


def f(x: float, d: int = 3, sign: bool = True) -> str:
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace("-", "$-$") if s.startswith("-") else s


def ci(lo: float, hi: float, d: int = 3) -> str:
    return f"[{f(lo, d, sign=False)}, {f(hi, d, sign=False)}]"


def write(name: str, body: str) -> None:
    (GEN / name).write_text(body.strip() + "\n")
    print("wrote", GEN / name)


# ---------------------------------------------------------------- Table 1: sample
rows = []
for t in ORDER:
    fa, c = (
        m["facts"][t],
        next(r for r in m["consistency"] if r["ticker"] == t)
        if isinstance(m["consistency"], list)
        else m["consistency"][t],
    )
    rows.append(
        f"{t} & {fa['mean_price_usd']:.2f} & {fa['tick_bps']:.2f} & {fa['spread_ticks']:.2f} & "
        f"{100 * fa['share_one_tick']:.1f} & {fa['depth_best']:,.0f} & {fa['n_messages']:,} & "
        f"{fa['n_trades']:,} & {fa['tick_class'].replace('-tick', '')} \\\\"
    )
write(
    "tab_sample.tex",
    r"""
\begin{tabular}{lrrrrrrrl}
\toprule
Stock & Price (\$) & Tick (bps) & Spread (ticks) & One-tick (\%) & Depth at best & Messages & Trades & Regime \\
\midrule
"""
    + "\n".join(rows[:2])
    + "\n\\addlinespace\n"
    + "\n".join(rows[2:])
    + r"""
\bottomrule
\end{tabular}""",
)

# ---------------------------------------------------------------- Table 2: fills and markouts
rows = []
for t in ORDER:
    summ = {r["model"]: r for r in m["fills"][t]["summary"]}
    for k, mod in enumerate(MODELS):
        r = summ[mod]
        lead = f"\\multirow{{3}}{{*}}{{{t}}}" if k == 0 else ""
        rows.append(
            f"{lead} & {NAME[mod]} & {r['p_fill']:.3f} & {f(r['mk_0.1'])} & {f(r['mk_1'])} & {f(r['mk_10'])} & "
            f"{f(r['mk_60'])} & {f(r['mk_10_bps'], 2)} \\\\"
        )
    rows.append(
        f" & \\textit{{Random fill}} & --- & \\multicolumn{{4}}{{c}}{{\\textit{{{summ['touch']['uncond_10']:.3f} at every horizon}}}} & "
        f"\\textit{{{f(summ['touch']['uncond_10_bps'], 2)}}} \\\\"
    )
    if t != ORDER[-1]:
        rows.append("\\addlinespace")
write(
    "tab_fills.tex",
    r"""
\begin{tabular}{llrrrrrr}
\toprule
 & & & \multicolumn{4}{c}{Markout (ticks per share) after the fill} & 10\,s \\
\cmidrule(lr){4-7}
Stock & Fill model & $P$(fill) & 0.1\,s & 1\,s & 10\,s & 60\,s & (bps) \\
\midrule
"""
    + "\n".join(rows)
    + r"""
\bottomrule
\end{tabular}""",
)

# ---------------------------------------------------------------- Table 3: rules out of sample (gross)
rows = []
gaps = []
for t in ORDER:
    for k, mod in enumerate(MODELS):
        r = rule(t, mod, None)
        gaps.append(r["in_sample"] - r["out_of_sample"])
        lead = f"\\multirow{{3}}{{*}}{{{t}}}" if k == 0 else ""
        rows.append(
            f"{lead} & {NAME[mod]} & {f(r['in_sample'])} & {f(r['out_of_sample'])} & {ci(*r['ci'])} & "
            f"{100 * r['share_post']:.0f} & {100 * r['share_cross']:.0f} & {f(r['always_post'])} & "
            f"{ci(*r['always_post_ci'])} \\\\"
        )
    if t != ORDER[-1]:
        rows.append("\\addlinespace")
write(
    "tab_rules.tex",
    r"""
\begin{tabular}{llrrcrrrc}
\toprule
 & & \multicolumn{5}{c}{Rule chosen before 12:45} & \multicolumn{2}{c}{Always post} \\
\cmidrule(lr){3-7}\cmidrule(lr){8-9}
Stock & Fill model & Train & Test & 95\% CI & Post \% & Cross \% & Test & 95\% CI \\
\midrule
"""
    + "\n".join(rows)
    + r"""
\bottomrule
\end{tabular}""",
)
n_worse = sum(g > 0 for g in gaps)
print(f"selection: in-sample > out-of-sample in {n_worse} of {len(gaps)} cases, mean gap {np.mean(gaps):+.3f}")

# ---------------------------------------------------------------- Table 4: fees and rebates
rows = []
for t in ORDER:
    for k, mod in enumerate(("touch", "fifo")):
        r2, r3 = rule(t, mod, 0.2), rule(t, mod, 0.3)
        lead = f"\\multirow{{2}}{{*}}{{{t}}}" if k == 0 else ""
        rows.append(
            f"{lead} & {NAME[mod]} & {f(r2['out_of_sample'])} & {ci(*r2['ci'])} & {f(r3['out_of_sample'])} & "
            f"{ci(*r3['ci'])} & {f(r3['always_post'])} & {ci(*r3['always_post_ci'])} \\\\"
        )
    if t != ORDER[-1]:
        rows.append("\\addlinespace")
write(
    "tab_costs.tex",
    r"""
\begin{tabular}{llrcrcrc}
\toprule
 & & \multicolumn{4}{c}{Rule re-chosen before 12:45 with costs} & \multicolumn{2}{c}{Always post} \\
\cmidrule(lr){3-6}\cmidrule(lr){7-8}
 & & \multicolumn{2}{c}{Rebate 0.2 ticks} & \multicolumn{2}{c}{Rebate 0.3 ticks} & \multicolumn{2}{c}{Rebate 0.3 ticks} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}
Stock & Fill model & Test & 95\% CI & Test & 95\% CI & Test & 95\% CI \\
\midrule
"""
    + "\n".join(rows)
    + r"""
\bottomrule
\end{tabular}""",
)

# ---------------------------------------------------------------- Table 5: horizon robustness (always post)
rob = json.loads((RESULTS / "fill_robustness.json").read_text())


def robust(ticker, model, check, horizon):
    return next(
        r
        for r in rob
        if r["ticker"] == ticker and r["model"] == model and r["check"] == check and r["horizon"] == horizon
    )


rows = []
for t in ORDER:
    for k, mod in enumerate(("touch", "fifo")):
        cells = []
        for h in (10.0, 30.0, 60.0):
            r = robust(t, mod, "horizon", h)
            cells.append(f"{f(r['always_post'])} & {ci(*r['always_post_ci'])}")
        lead = f"\\multirow{{2}}{{*}}{{{t}}}" if k == 0 else ""
        rows.append(f"{lead} & {NAME[mod]} & " + " & ".join(cells) + " \\\\")
    if t != ORDER[-1]:
        rows.append("\\addlinespace")
write(
    "tab_horizons.tex",
    r"""
\begin{tabular}{llrcrcrc}
\toprule
 & & \multicolumn{2}{c}{$H = 10$\,s} & \multicolumn{2}{c}{$H = 30$\,s} & \multicolumn{2}{c}{$H = 60$\,s} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}
Stock & Fill model & Test & 95\% CI & Test & 95\% CI & Test & 95\% CI \\
\midrule
"""
    + "\n".join(rows)
    + r"""
\bottomrule
\end{tabular}""",
)
checked = sum(m["consistency"][t]["n_checked"] for t in ORDER)
passed = sum(m["consistency"][t]["n_pass"] for t in ORDER)
print(f"consistency: {passed:,} of {checked:,} checked messages pass")

# ---------------------------------------------------------------- figure style
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7.5,
        "axes.edgecolor": BASE,
        "axes.linewidth": 0.6,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)


def save(fig, name: str) -> None:
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(GEN / f"{name}_preview.png", dpi=220)
    plt.close(fig)
    print("wrote", FIG / f"{name}.pdf")


# ---------------------------------------------------------------- Figure 1: markouts by fill model
H = (0.1, 1, 10, 60)
fig, axes = plt.subplots(2, 3, figsize=(6.5, 3.9), constrained_layout=True)
for ax, t in zip(axes.flat, ORDER):
    summ = {r["model"]: r for r in m["fills"][t]["summary"]}
    for mod in MODELS:
        y = [summ[mod][f"mk_{h:g}"] for h in H]
        ax.plot(
            H,
            y,
            STYLE[mod],
            color=COLOR[mod],
            marker=MARK[mod],
            markersize=3.2,
            linewidth=1.2,
            label=f"{NAME[mod]} fills",
        )
    ax.axhline(
        summ["touch"]["uncond_10"],
        color=INK2,
        linewidth=0.8,
        linestyle=(0, (5, 2, 1, 2)),
        label="Random fill (half spread)",
    )
    ax.axhline(0, color=BASE, linewidth=0.6)
    ax.set_xscale("log")
    ax.set_xticks(H, ["0.1", "1", "10", "60"])
    ax.minorticks_off()
    ax.set_title(f"{t} ({'large' if t in LARGE else 'small'} tick)", loc="left")
for ax in axes[:, 0]:
    ax.set_ylabel("Markout (ticks per share)")
for ax in axes[1, :2]:
    ax.set_xlabel("Seconds after the fill")
axes[0, 2].set_xlabel("Seconds after the fill")
handles, labels = axes[0, 0].get_legend_handles_labels()
axes[1, 2].axis("off")
axes[1, 2].legend(handles, labels, loc="center", frameon=False, handlelength=3.2)
save(fig, "fig_markouts")

# ---------------------------------------------------------------- Figure 2: always post, out of sample
fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6), layout="constrained", gridspec_kw={"width_ratios": [1, 1.25]})
series = [
    ("touch", None, True, "Touch"),
    ("touch", 0.3, False, "Touch + 0.3-tick rebate"),
    ("fifo", None, True, "FIFO"),
    ("fifo", 0.3, False, "FIFO + 0.3-tick rebate"),
]
offsets = (0.27, 0.09, -0.09, -0.27)
for ax, group, title in ((axes[0], LARGE, "Large tick"), (axes[1], SMALL, "Small tick")):
    for (mod, reb, filled, lab), off in zip(series, offsets):
        for j, t in enumerate(group):
            r = rule(t, mod, 0.3 if reb else None)
            x, (lo, hi) = r["always_post"], r["always_post_ci"]
            y = len(group) - 1 - j + off
            ax.plot([lo, hi], [y, y], color=COLOR[mod], linewidth=1.1, solid_capstyle="butt")
            ax.plot(
                [x],
                [y],
                MARK[mod],
                color=COLOR[mod],
                markersize=4.2,
                markerfacecolor=COLOR[mod] if filled else "white",
                markeredgewidth=1.0,
                label=lab if j == 0 else None,
            )
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.set_yticks(range(len(group)), list(reversed(group)))
    ax.set_ylim(-0.6, len(group) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_title(title, loc="left")
    ax.set_xlabel("Out-of-sample EV per decision (ticks per share)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False, fontsize=7.5, handlelength=1.2)
save(fig, "fig_always_post")
