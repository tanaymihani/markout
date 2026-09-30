"""Predictions Cup decision journal:  python -m markout.decision.journal init|add|resolve|score

One row per trade, written *before* the trade is placed:

    timestamp, market, question, side, p_belief, price_q, kelly_fraction, stake,
    bankroll_before, reason, outcome, resolved_at

* p_belief is your probability that the market resolves YES; price_q is the YES price at
  entry (a NO share costs 1 - price_q). Both are in (0, 1); percent or cents (e.g. 63)
  are accepted by `add` and divided by 100.
* kelly_fraction is computed, never typed: (p - q)/(1 - q) for YES, (q - p)/q for NO,
  and 0 when your belief favours the other side (markout.decision.kelly).
* outcome is 1 (YES) or 0 (NO), blank until the market resolves (`resolve`).

`score` compares your beliefs with the market's prices on the resolved rows:
* Brier score mean (p - y)^2 and log loss -mean[y ln p + (1 - y) ln(1 - p)], yours vs the
  market's, with a paired standard error ("did you beat the market?");
* a calibration table and plot;
* realized PnL against the PnL your beliefs predicted. The difference is luck *if* your
  beliefs were right. A persistent shortfall is the optimizer's curse (Smith & Winkler
  2006): the bets you choose are the ones where you most disagree with the market, so
  they over-represent your own errors;
* your stakes as a multiple of the Kelly fraction.
It then regenerates reports/05_predictions_cup.md (Part 2).
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from markout.decision import kelly
from markout.paths import DATA, ROOT

COLUMNS = ["timestamp", "market", "question", "side", "p_belief", "price_q", "kelly_fraction", "stake",
           "bankroll_before", "reason", "outcome", "resolved_at"]
TEMPLATE = ROOT / "configs" / "journal_template.csv"
JOURNAL = DATA / "journal.csv"
CLIP = 1e-6                     # log loss clip
MIN_ROWS_TO_JUDGE = 10          # below this, "did you beat the market?" is not answered


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def as_prob(x: float | str) -> float:
    """A probability in (0, 1); values in [1, 100) are read as percent or cents."""
    v = float(x)
    if 1.0 <= v < 100.0:
        v /= 100.0
    if not 0.0 < v < 1.0:
        raise ValueError(f"expected a probability in (0, 1) or a percent in (0, 100), got {x!r}")
    return v


# --------------------------------------------------------------------------- writing

def init(path: Path = JOURNAL, template: Path = TEMPLATE, force: bool = False) -> Path:
    """Copy the template (header only) to `path`; refuses to overwrite a journal unless forced."""
    path = Path(path)
    if path.exists() and not force:
        raise FileExistsError(f"{path} exists; pass --force to overwrite it")
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(template, path)
    return path


def add(path: Path, market: str, side: str, p: float | str, q: float | str, stake: float, bankroll: float,
        question: str = "", reason: str = "", timestamp: str | None = None) -> dict:
    """Append one trade, computing its Kelly fraction. Returns the row written."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `journal init` first")
    p, q = as_prob(p), as_prob(q)
    side = side.strip().upper()
    stake, bankroll = float(stake), float(bankroll)
    if bankroll <= 0 or stake < 0 or stake > bankroll:
        raise ValueError("need bankroll > 0 and 0 <= stake <= bankroll")
    row = {"timestamp": timestamp or _now(), "market": market, "question": question, "side": side,
           "p_belief": f"{p:.4f}", "price_q": f"{q:.4f}", "kelly_fraction": f"{float(kelly.kelly_fraction(p, q, side)):.4f}",
           "stake": f"{stake:.12g}", "bankroll_before": f"{bankroll:.12g}", "reason": reason, "outcome": "",
           "resolved_at": ""}
    with path.open("a", newline="") as fh:
        csv.DictWriter(fh, fieldnames=COLUMNS).writerow(row)
    return row


def resolve(path: Path, market: str, outcome: int, resolved_at: str | None = None) -> int:
    """Set the outcome (1 = YES, 0 = NO) on every unresolved row of `market`; returns the count."""
    if int(outcome) not in (0, 1):
        raise ValueError("outcome must be 1 (YES) or 0 (NO)")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    hit = (df["market"] == market) & (df["outcome"] == "")
    df.loc[hit, "outcome"] = str(int(outcome))
    df.loc[hit, "resolved_at"] = resolved_at or _now()
    df[COLUMNS].to_csv(path, index=False)
    return int(hit.sum())


def load(path: Path = JOURNAL) -> pd.DataFrame:
    """The journal with numeric columns parsed (outcome is NaN until resolved)."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"journal is missing columns {sorted(missing)}")
    for c in ("p_belief", "price_q", "kelly_fraction", "stake", "bankroll_before"):
        df[c] = pd.to_numeric(df[c])
    df["outcome"] = pd.to_numeric(df["outcome"].replace("", np.nan))
    df["side"] = df["side"].str.upper()
    return df


# --------------------------------------------------------------------------- scoring

def pnl_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Realized PnL, the PnL your belief predicted, and its variance under your belief."""
    out = df.copy()
    yes = out["side"] == "YES"
    p, q, y, s = out["p_belief"], out["price_q"], out["outcome"], out["stake"]
    out["pnl"] = s * np.where(yes, y / q - 1, (1 - y) / (1 - q) - 1)
    out["pnl_predicted"] = s * np.where(yes, p / q - 1, (1 - p) / (1 - q) - 1)
    out["pnl_var"] = s**2 * p * (1 - p) / np.where(yes, q, 1 - q) ** 2
    out["frac"] = s / out["bankroll_before"]
    return out


def calibration_table(forecast: np.ndarray, outcome: np.ndarray, n_bins: int) -> list[dict]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(forecast, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            rows.append({"bin": f"{edges[b]:.1f}–{edges[b + 1]:.1f}", "n": int(m.sum()),
                         "mean_forecast": float(forecast[m].mean()), "observed": float(outcome[m].mean())})
    return rows


def _paired(d: np.ndarray) -> tuple[float, float, float]:
    n = len(d)
    mean = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    return mean, se, (mean / se if se and math.isfinite(se) and se > 0 else float("nan"))


def score(df: pd.DataFrame) -> dict | None:
    """All Part 2 metrics on the resolved rows; None when nothing has resolved yet."""
    res = df[df["outcome"].isin([0, 1])]
    if res.empty:
        return None
    r = pnl_columns(res)
    y, p, q = r["outcome"].to_numpy(float), r["p_belief"].to_numpy(float), r["price_q"].to_numpy(float)
    pc, qc = np.clip(p, CLIP, 1 - CLIP), np.clip(q, CLIP, 1 - CLIP)
    brier_you, brier_mkt = (p - y) ** 2, (q - y) ** 2
    ll_you = -(y * np.log(pc) + (1 - y) * np.log(1 - pc))
    ll_mkt = -(y * np.log(qc) + (1 - y) * np.log(1 - qc))
    db, db_se, db_t = _paired(brier_you - brier_mkt)
    dl, dl_se, dl_t = _paired(ll_you - ll_mkt)
    n = len(r)
    n_bins = 10 if n >= 100 else 5
    k = r["kelly_fraction"].to_numpy(float)
    frac = r["frac"].to_numpy(float)
    pos = k > 0
    ratio = frac[pos] / k[pos]
    stake_total = float(r["stake"].sum())
    luck = float(r["pnl"].sum() - r["pnl_predicted"].sum())
    sd = math.sqrt(float(r["pnl_var"].sum()))
    return {
        "n_resolved": n, "n_logged": int(len(df)), "n_markets": int(r["market"].nunique()),
        "brier_you": float(brier_you.mean()), "brier_market": float(brier_mkt.mean()),
        "brier_skill": 1 - float(brier_you.mean()) / float(brier_mkt.mean()) if brier_mkt.mean() > 0 else float("nan"),
        "brier_diff": db, "brier_diff_se": db_se, "brier_diff_t": db_t,
        "logloss_you": float(ll_you.mean()), "logloss_market": float(ll_mkt.mean()),
        "logloss_diff": dl, "logloss_diff_se": dl_se, "logloss_diff_t": dl_t,
        "calibration_you": calibration_table(p, y, n_bins), "calibration_market": calibration_table(q, y, n_bins),
        "pnl_realized": float(r["pnl"].sum()), "pnl_predicted": float(r["pnl_predicted"].sum()),
        "luck": luck, "luck_sd": sd, "luck_z": luck / sd if sd > 0 else float("nan"),
        "stake_total": stake_total,
        "return_realized": float(r["pnl"].sum()) / stake_total if stake_total > 0 else float("nan"),
        "return_predicted": float(r["pnl_predicted"].sum()) / stake_total if stake_total > 0 else float("nan"),
        "median_kelly_multiple": float(np.median(ratio)) if pos.any() else float("nan"),
        "share_over_kelly": float((ratio > 1).mean()) if pos.any() else float("nan"),
        "n_against_belief": int(((k == 0) & (frac > 0)).sum()),
        "median_stake_fraction": float(np.median(frac)),
        "hit_rate": float(np.mean(np.where(r["side"] == "YES", y, 1 - y))),
    }


def verdict(s: dict) -> str:
    """Result-dependent answer to 'did you beat the market?'."""
    if s["n_resolved"] < MIN_ROWS_TO_JUDGE:
        return (f"Too few resolved trades to tell ({s['n_resolved']}, fewer than {MIN_ROWS_TO_JUDGE}); "
                f"the difference below is noise-dominated.")
    t = s["brier_diff_t"]
    if not math.isfinite(t) or abs(t) < 2:
        return (f"Not distinguishable from the market: the Brier difference is {s['brier_diff']:+.4f} "
                f"± {s['brier_diff_se']:.4f} (t = {t:.1f}).")
    better = s["brier_diff"] < 0
    return (f"{'Yes' if better else 'No'}: your Brier score is {'lower' if better else 'higher'} than the market's "
            f"by {abs(s['brier_diff']):.4f} ± {s['brier_diff_se']:.4f} (t = {t:.1f}).")


# ----------------------------------------------------------------- figures and text

def figures(df: pd.DataFrame, s: dict) -> None:
    from markout import plotting as mp

    res = pnl_columns(df[df["outcome"].isin([0, 1])]).sort_values("timestamp")

    def calib():
        fig, ax = mp.subplots(w=5.6, h=4.6)
        ax.plot([0, 1], [0, 1], color=mp.ink("muted"), linestyle="--", linewidth=1.0, label="perfect calibration")
        for i, (key, lab) in enumerate((("calibration_market", "market price"), ("calibration_you", "your belief"))):
            rows = s[key]
            ax.plot([r["mean_forecast"] for r in rows], [r["observed"] for r in rows], "o-", color=mp.series(i),
                    label=lab)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("forecast probability of YES (bin mean)")
        ax.set_ylabel("observed YES frequency")
        mp.legend(ax, loc="upper left")
        mp.title(ax, "Calibration", f"{s['n_resolved']} resolved trades")
        return fig

    def pnl():
        fig, ax = mp.subplots()
        x = np.arange(1, len(res) + 1)
        mp.zero_line(ax)
        ax.plot(x, res["pnl_predicted"].cumsum(), color=mp.series(0), label="predicted by your beliefs")
        ax.plot(x, res["pnl"].cumsum(), color=mp.series(1), label="realized")
        ax.set_xlabel("resolved trade, in entry order")
        ax.set_ylabel("cumulative PnL (SUSQies)")
        mp.legend(ax, loc="upper left")
        mp.title(ax, "Realized vs predicted PnL", "the gap is luck if your beliefs were right")
        return fig

    def sizing():
        fig, ax = mp.subplots(w=5.6, h=4.6)
        lim = max(float(res["kelly_fraction"].max()), float(res["frac"].max()), 0.05) * 1.1
        ax.plot([0, lim], [0, lim], color=mp.ink("muted"), linestyle="--", linewidth=1.0, label="stake = Kelly")
        ax.plot([0, lim], [0, lim / 2], color=mp.ink("muted"), linestyle=":", linewidth=1.0, label="half Kelly")
        ax.plot(res["kelly_fraction"], res["frac"], "o", color=mp.series(0), label="trade")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("Kelly fraction of bankroll")
        ax.set_ylabel("stake / bankroll")
        mp.legend(ax, loc="upper left")
        mp.title(ax, "Stakes vs Kelly", f"median stake = {s['median_kelly_multiple']:.2f} × Kelly")
        return fig

    mp.render("p_journal_calibration", calib)
    mp.render("p_journal_pnl", pnl)
    mp.render("p_journal_kelly", sizing)


def part2_markdown(s: dict | None, n_logged: int, status: str, assumed_skill: float | None = None) -> str:
    """Part 2 of report 05. `status` is 'pending', 'interim' or 'final'."""
    from markout import plotting as mp
    from markout import reporting as rp

    head = "## Part 2: Results (generated by `journal score` after Nov 4)\n\n"
    if s is None:
        return head + (f"**Pending.** {n_logged} trade{'s' if n_logged != 1 else ''} logged in `data/journal.csv`, "
                       "none resolved yet. Run `python -m markout.decision.journal score` once markets resolve; "
                       "this section is regenerated from the journal and nothing in it is typed by hand.\n")
    cal = [{"your bin": a["bin"], "n": a["n"], "your mean p": f"{a['mean_forecast']:.2f}", "observed": f"{a['observed']:.2f}"}
           for a in s["calibration_you"]]
    calm = [{"market bin": a["bin"], "n": a["n"], "mean price": f"{a['mean_forecast']:.2f}", "observed": f"{a['observed']:.2f}"}
            for a in s["calibration_market"]]
    skill_txt = ""
    if assumed_skill is not None:
        skill_txt = (f" Part 1 assumed a player whose Brier score beats the market's by {assumed_skill:.1%}; "
                     f"yours is {s['brier_skill']:+.1%}.")
    label = "Interim results (the contest is still running)." if status == "interim" else "Final results."
    return head + f"""**{label}** {s['n_resolved']} resolved trades on {s['n_markets']} markets
(of {s['n_logged']} logged).

**Did you beat the market?** {verdict(s)}{skill_txt}

{rp.table([
    {"score": "Brier (lower is better)", "you": f"{s['brier_you']:.4f}", "market at entry": f"{s['brier_market']:.4f}",
     "you − market": f"{s['brier_diff']:+.4f} ± {s['brier_diff_se']:.4f}"},
    {"score": "log loss (lower is better)", "you": f"{s['logloss_you']:.4f}", "market at entry": f"{s['logloss_market']:.4f}",
     "you − market": f"{s['logloss_diff']:+.4f} ± {s['logloss_diff_se']:.4f}"},
])}

These are the markets you chose to trade, the ones where you disagreed most with the price. If part
of that disagreement was your own error, these rows are where it concentrates (the optimizer's curse).
So this comparison is harder on you than one over all markets would be, and it is the one that decides
your PnL.

{mp.picture('p_journal_calibration', 'Calibration of your beliefs and of market prices')}

{rp.table(cal)}

{rp.table(calm)}

**PnL: belief vs luck.** Realized {s['pnl_realized']:+,.1f} SUSQies against {s['pnl_predicted']:+,.1f} predicted
by your beliefs, a gap of {s['luck']:+,.1f} ({s['luck_z']:+.1f} standard deviations of the luck your own beliefs
imply). Per unit staked you expected {s['return_predicted']:+.1%} and got {s['return_realized']:+.1%}.
{"The shortfall is larger than luck alone explains: evidence that your edge was smaller than you believed." if s['luck_z'] < -2 else "The gap is within what luck explains under your own beliefs."}

{mp.picture('p_journal_pnl', 'Cumulative realized and belief-predicted PnL')}

**Stakes vs Kelly.** Median stake {s['median_stake_fraction']:.1%} of bankroll, a median
{s['median_kelly_multiple']:.2f}× the Kelly fraction; {s['share_over_kelly']:.0%} of trades were above full Kelly
and {s['n_against_belief']} went against your own stated belief. Hit rate {s['hit_rate']:.0%}.

{mp.picture('p_journal_kelly', 'Stake as a fraction of bankroll against the Kelly fraction')}
"""


# ------------------------------------------------------------------------------ CLI

def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m markout.decision.journal", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--journal", default=str(JOURNAL), help="journal CSV (default data/journal.csv)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s_init = sub.add_parser("init", help="copy configs/journal_template.csv to the journal path")
    s_init.add_argument("--force", action="store_true")
    s_add = sub.add_parser("add", help="log a trade before placing it")
    s_add.add_argument("--market", required=True)
    s_add.add_argument("--question", default="")
    s_add.add_argument("--side", required=True, choices=["YES", "NO", "yes", "no"])
    s_add.add_argument("--p", required=True, help="your P(YES), 0-1 or percent")
    s_add.add_argument("--q", required=True, help="YES price at entry, 0-1 or cents")
    s_add.add_argument("--stake", required=True, type=float)
    s_add.add_argument("--bankroll", required=True, type=float, help="bankroll before this trade")
    s_add.add_argument("--reason", default="")
    s_res = sub.add_parser("resolve", help="record a market's outcome")
    s_res.add_argument("--market", required=True)
    s_res.add_argument("--outcome", required=True, type=int, choices=[0, 1])
    sub.add_parser("score", help="score resolved rows and regenerate reports/05_predictions_cup.md")
    a = ap.parse_args(argv)
    path = Path(a.journal)
    if a.cmd == "init":
        print(f"created {init(path, force=a.force)}")
    elif a.cmd == "add":
        row = add(path, a.market, a.side, a.p, a.q, a.stake, a.bankroll, a.question, a.reason)
        k = float(row["kelly_fraction"])
        print(f"logged {row['market']} {row['side']}: p={row['p_belief']} q={row['price_q']} "
              f"Kelly {k:.1%} of bankroll, stake {float(row['stake']) / a.bankroll:.1%}")
        if k == 0:
            print("note: your belief favours the other side; Kelly says no bet", file=sys.stderr)
    elif a.cmd == "resolve":
        print(f"resolved {resolve(path, a.market, a.outcome)} row(s) of {a.market}")
    else:
        from markout.decision import contest
        report = contest.write_report(journal_path=path)
        print(f"wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
