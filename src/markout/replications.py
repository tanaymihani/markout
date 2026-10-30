"""Published results this repository reproduces, next to the numbers it gets (the README's replications table).

Each row pairs a value printed in a paper, or in a public repository, with the same quantity computed here, and
names what differs between the two setups. The published values were read from the sources in `PUBLISHED`; the
repository's values come from the saved results of reports 01 and 02, from the Deflated Sharpe formulas, and from
the S&P 500 and VIX data of report 06.

    python -m markout.replications     # -> reports/results/replications.json, and prints the table

The Carr-Wu row needs the Yahoo data in data/ (`python -m markout.vol.data`); without it the last saved value is
kept.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from markout import results

NAME = "replications"

PUBLISHED: dict[str, dict[str, Any]] = {
    "dsr": {
        "cite": "Bailey & López de Prado (2014), *J. Portfolio Management* 40(5), the numerical example",
        "url": "https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf",
        "sr0": 0.1132,
        "dsr_100": 0.9004,
        "dsr_46": 0.9505,
        "dsr_88_normal": 0.9505,
    },
    "cks": {
        "cite": "Cont, Kukanov & Stoikov (2014), *J. Financial Econometrics* 12(1), Tables 2 and 3",
        "url": "https://arxiv.org/abs/1011.6402",
        "mean_r2": 0.65,  # Table 2, average over 50 stocks of the per-window R2 (10-s changes, half-hour windows)
        "lam": 0.98,  # Table 3 grand mean of lambda in log(beta) = a - lambda log(depth), fitted per stock
        "lam_ci": (0.88, 1.08),
        "data": "50 NYSE stocks, TAQ quotes and trades, April 2010",
    },
    "gb": {
        "cite": "Gould & Bonart (2016), *Market Microstructure and Liquidity* 2(2), Tables 2 and 4",
        "url": "https://arxiv.org/abs/1512.03492",
        "auc_large": (0.752, 0.805),  # out-of-sample AUC, MSFT INTC MU CSCO ORCL
        "auc_small": (0.581, 0.642),  # GOOG AMZN TSLA PCLN NFLX
        "slope_large": (2.03, 2.73),  # logistic coefficient on the imbalance
        "slope_small": (0.50, 0.85),
        "data": "10 Nasdaq stocks, LOBSTER, all of 2014",
    },
    "optiver": {
        "cite": "a public repository",
        "url": "https://github.com/andrew-somerset/optiver-trading-at-the-close",
        "median": 5.815,
        "gbm": 5.708,
    },
    "carr_wu": {
        "cite": "Carr & Wu, *Variance Risk Premia*, May 2004 working paper (published in *RFS* 22(3), 2009), "
        "Tables 4 and 5, SPX row",
        "url": "https://econwpa.ub.uni-muenchen.de/econ-wp/fin/papers/0409/0409015.pdf",
        "period": ("1996-01-04", "2003-02-28"),
        "rv_vol": 18.82,
        "sw_vol": 24.41,
        "rp": -2.279,  # mean of 100 (RV - SW)
        "rp_t": -7.194,
        "lrp": -0.594,  # mean of ln(RV / SW)
        "lrp_t": -9.484,
    },
}


def newey_west_t(x: np.ndarray, lag: int = 30) -> float:
    """t-statistic of the mean with a Bartlett-kernel Newey-West variance (Carr and Wu use 30 lags)."""
    x = np.asarray(x, dtype=float)
    n, e = len(x), x - x.mean()
    s = float(e @ e) / n
    for k in range(1, min(lag, n - 1) + 1):
        s += 2.0 * (1.0 - k / (lag + 1.0)) * float(e[k:] @ e[:-k]) / n
    return float(x.mean() / np.sqrt(s / n))


def dsr_row() -> dict:
    from markout.evaluation.dsr import expected_max_sr, psr

    sr, var, T = 2.5 / np.sqrt(250), 0.5 / 250, 1250  # the paper's inputs, annual figures turned daily
    return {
        "sr0": expected_max_sr(var, 100),
        "dsr_100": psr(sr, expected_max_sr(var, 100), T, -3.0, 10.0),
        "dsr_46": psr(sr, expected_max_sr(var, 46), T, -3.0, 10.0),
        "dsr_88_normal": psr(sr, expected_max_sr(var, 88), T, 0.0, 3.0),
    }


def lob_rows(m: dict) -> dict:
    """CKS and Gould-Bonart quantities from report 02's saved results (one LOBSTER day, five stocks)."""
    large, small = ("INTC", "MSFT"), ("AAPL", "AMZN", "GOOG")
    cks, pred = m["cks"], m["predictive"]
    r2 = {t: cks["tickers"][t]["mean_r2"] for t in cks["tickers"]}
    auc = {t: pred[t]["imbalance"]["auc"] for t in pred}
    slope = {t: pred[t]["imbalance"]["coef"] for t in pred}
    within, pooled = cks["depth_slope_within"], cks["depth_slope"]
    return {
        "cks": {
            "mean_r2": float(np.mean(list(r2.values()))),
            "r2_range": (min(r2.values()), max(r2.values())),
            "lam_within": -within["slope"],
            "lam_within_ci": (-within["ci"][1], -within["ci"][0]),
            "lam_pooled": -pooled["slope"],
            "lam_pooled_ci": (-pooled["ci"][1], -pooled["ci"][0]),
        },
        "gb": {
            "auc_large": (min(auc[t] for t in large), max(auc[t] for t in large)),
            "auc_small": (min(auc[t] for t in small), max(auc[t] for t in small)),
            "slope_large": (min(slope[t] for t in large), max(slope[t] for t in large)),
            "slope_small": (min(slope[t] for t in small), max(slope[t] for t in small)),
        },
    }


def carr_wu_row() -> dict | None:
    """Carr and Wu's SPX statistics over their sample, with VIX^2 as the 30-day variance-swap rate."""
    from markout.vol import data, vrp

    if not data.available():
        return None
    lo, hi = PUBLISHED["carr_wu"]["period"]
    f = vrp.frame(data.load()).loc[lo:hi].dropna(subset=["rv", "iv2"])
    rp = (100.0 * (f["rv"] - f["iv2"])).to_numpy()
    lrp = np.log(f["rv"] / f["iv2"]).to_numpy()
    return {
        "days": len(f),
        "rv_vol": float(f["rv_vol"].mean()),
        "sw_vol": float(f["vix"].mean()),
        "rp": float(rp.mean()),
        "rp_t": newey_west_t(rp),
        "lrp": float(lrp.mean()),
        "lrp_t": newey_west_t(lrp),
    }


def compute() -> dict:
    out: dict[str, Any] = {"published": PUBLISHED, "here": {"dsr": dsr_row()}}
    m = results.load("microstructure")
    if m is not None:
        out["here"].update(lob_rows(m))
    a = results.load("auction")
    if a is not None and a.get("holdout"):
        out["here"]["optiver"] = {"median": a["holdout"]["mae_median"], "model": a["holdout"]["mae_model"]}
    cw = carr_wu_row()
    if cw is None:
        saved = results.load(NAME)
        cw = saved["here"].get("carr_wu") if saved else None
    if cw is not None:
        out["here"]["carr_wu"] = cw
    return out


def _r(lo_hi, fmt: str = "{:.3f}") -> str:
    lo, hi = lo_hi
    return f"{fmt.format(lo)}–{fmt.format(hi)}"


def table(res: dict) -> str:
    """The README table: what was published, what this repository gets, and where the setups differ."""
    p, h = res["published"], res["here"]
    rows = []
    if "dsr" in h:
        d, q = p["dsr"], h["dsr"]
        rows.append(
            (
                f"Deflated Sharpe worked example ([{d['cite']}]({d['url']}))",
                f"SR₀ {d['sr0']:.4f}; DSR {d['dsr_100']:.4f} at N = 100, {d['dsr_46']:.4f} at N = 46, "
                f"{d['dsr_88_normal']:.4f} for normal returns at N = 88",
                f"{q['sr0']:.4f}; {q['dsr_100']:.4f}, {q['dsr_46']:.4f}, {q['dsr_88_normal']:.4f} "
                "([tests](tests/test_evaluation_dsr.py))",
                "Same inputs; equal to the printed four decimals.",
            )
        )
    if "cks" in h:
        c, q = p["cks"], h["cks"]
        rows.append(
            (
                f"Order-flow imbalance explains the same 10 s price change ([{c['cite']}]({c['url']}))",
                f"average R² {c['mean_r2']:.2f} ({c['data']})",
                f"{q['mean_r2']:.2f}, from {_r(q['r2_range'], '{:.2f}')} by stock "
                "([report 02](reports/02_microstructure.md))",
                "Five NASDAQ stocks on one LOBSTER day (2012); same buckets and half-hour windows.",
            )
        )
        rows.append(
            (
                "Price impact is inversely proportional to depth (same paper)",
                f"λ = {c['lam']:.2f} [{_r(c['lam_ci'], '{:.2f}')}] in log β = a − λ log depth, fitted stock by stock",
                f"{q['lam_within']:.2f} [{_r(q['lam_within_ci'], '{:.2f}')}] within stocks; "
                f"{q['lam_pooled']:.2f} [{_r(q['lam_pooled_ci'], '{:.2f}')}] pooled across them "
                "([report 02](reports/02_microstructure.md))",
                "They fit each stock across its windows, so the within-stock slope is the comparable one: its interval "
                "and theirs contain 1, the pooled one does not.",
            )
        )
    if "gb" in h:
        g, q = p["gb"], h["gb"]
        rows.append(
            (
                f"Queue imbalance predicts the next mid-price move, better for large-tick stocks "
                f"([{g['cite']}]({g['url']}))",
                f"out-of-sample AUC {_r(g['auc_large'])} large-tick, {_r(g['auc_small'])} small-tick; logistic slope "
                f"{_r(g['slope_large'], '{:.2f}')} vs {_r(g['slope_small'], '{:.2f}')} ({g['data']})",
                f"AUC {_r(q['auc_large'])} vs {_r(q['auc_small'])}; slope {_r(q['slope_large'], '{:.2f}')} vs "
                f"{_r(q['slope_small'], '{:.2f}')} ([report 02](reports/02_microstructure.md))",
                "Same imbalance and target. One 2012 day, sampled every second and split by time of day; the "
                "ordering holds, the levels are lower.",
            )
        )
    if "optiver" in h:
        o, q = p["optiver"], h["optiver"]
        rows.append(
            (
                f"Optiver *Trading at the Close*, MAE on the same holdout days ([{o['cite']}]({o['url']}))",
                f"median baseline {o['median']:.3f}, gradient boosting {o['gbm']:.3f} bps",
                f"median {q['median']:.3f}, LightGBM {q['model']:.3f} ([report 01](reports/01_auction.md))",
                "The matching baseline checks the data and the split.",
            )
        )
    if "carr_wu" in h:
        w, q = p["carr_wu"], h["carr_wu"]
        rows.append(
            (
                f"S&P 500 variance risk premium, {w['period'][0][:7]} to {w['period'][1][:7]} "
                f"([{w['cite']}]({w['url']}))",
                f"realized vol {w['rv_vol']:.2f} vs swap rate {w['sw_vol']:.2f}; mean 100·(RV − SW) {w['rp']:.3f} "
                f"(t = {w['rp_t']:.2f}); mean ln(RV/SW) {w['lrp']:.3f} (t = {w['lrp_t']:.2f})",
                f"{q['rv_vol']:.2f} vs {q['sw_vol']:.2f}; {q['rp']:.3f} (t = {q['rp_t']:.2f}); {q['lrp']:.3f} "
                f"(t = {q['lrp_t']:.2f}) ([code](src/markout/replications.py), data of [report 06](reports/06_vol_premium.md))",
                "VIX² stands in for their option-built swap rate; realized variance over the next 21 trading days; "
                "Newey–West t with 30 lags, as theirs.",
            )
        )
    if not rows:
        return ""
    head = "| result | published | here | setup |\n|---|---|---|---|\n"
    return head + "\n".join("| " + " | ".join(r) + " |" for r in rows) + "\n"


def main() -> int:
    res = compute()
    results.save(NAME, res)
    print(table(res))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
