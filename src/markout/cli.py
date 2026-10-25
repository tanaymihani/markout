"""The `markout` command: the toolkit on your own data, and the project's own builds.

    markout status                               what is on disk: data, the C++ extension, results
    markout fills MESSAGES.csv BOOK.csv          queue-position fills and markouts for any LOBSTER day
    markout fills --ticker INTC                  ... or one of the bundled sample days (make lobster)
    markout sharpe pnl.csv --trials 40 --sr-var 0.0004
                                                 Sharpe with a bootstrap CI, PSR and the Deflated Sharpe
    markout pbo trials.csv                       probability of backtest overfitting (CSCV)
    markout kelly --p 0.62 --price 0.55          Kelly stake for a binary contract
    markout report microstructure                regenerate one report (or `all`)
    markout site | markout readme                rebuild docs/index.html or README.md from the results
    markout desk serve --paper                   the Predictions Cup desk (same as python -m markout.cup)

Commands that compute something print a table, or JSON with --json.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

REPORTS = {
    "auction": "markout.auction.report",
    "microstructure": "markout.lob.report",
    "arena": "markout.games.report",
    "options": "markout.options.report",
    "contest": "markout.decision.contest",
    "vol": "markout.vol.report",
}


def _emit(obj: dict, as_json: bool, rows: list[tuple[str, Any]] | None = None) -> None:
    if as_json:
        print(json.dumps(obj, indent=2, default=_jsonable))
        return
    width = max(len(k) for k, _ in rows or [("", 0)])
    for k, v in rows or []:
        print(f"{k:<{width}}  {_fmt(v)}")


def _jsonable(o: Any) -> Any:
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def _fmt(v: Any) -> str:
    if isinstance(v, bool) or v is None:
        return {True: "yes", False: "no", None: "n/a"}[v]
    if isinstance(v, float):
        return "n/a" if not math.isfinite(v) else f"{v:.4g}"
    return str(v)


def _clock(s: str) -> float:
    """Seconds after midnight from '09:35', '09:35:30' or '34500'."""
    if ":" not in s:
        return float(s)
    h, m, sec = ([float(x) for x in s.split(":")] + [0.0])[:3]
    return 3600 * h + 60 * m + sec


def _frame(path: str):
    import pandas as pd

    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


def _series(path: str, column: str | None):
    import numpy as np

    df = _frame(path)
    num = df.select_dtypes("number")
    if column:
        if column not in df.columns:
            raise SystemExit(f"{path}: no column {column!r} (columns: {', '.join(map(str, df.columns))})")
        return df[column].to_numpy(dtype=float)
    if num.shape[1] != 1:
        raise SystemExit(f"{path}: {num.shape[1]} numeric columns, pick one with --column")
    return np.asarray(num.iloc[:, 0], dtype=float)


# --------------------------------------------------------------------------- commands
def cmd_status(a) -> int:
    from markout import __version__
    from markout.lob import lobster
    from markout.lob.fills import cpp_available
    from markout.paths import PROCESSED, RAW, RESULTS

    out: dict[str, Any] = {
        "version": __version__,
        "lobster_sample": lobster.available(),
        "optiver_auction": (PROCESSED / "optiver" / "train.parquet").exists(),
        "sp500_vix_history": (RAW / "vol" / "spx.parquet").exists(),
        "cpp_extension": cpp_available(),
        "results": sorted(p.stem for p in RESULTS.glob("*.json")),
    }
    rows = [
        ("markout", out["version"]),
        ("LOBSTER sample days (make lobster)", out["lobster_sample"]),
        ("Optiver auction data (make optiver)", out["optiver_auction"]),
        ("S&P 500 and VIX history (make vol)", out["sp500_vix_history"]),
        ("C++ fill simulator (make cpp)", out["cpp_extension"]),
        ("results on disk", ", ".join(out["results"]) or "none (make report)"),
    ]
    _emit(out, a.json, rows)
    return 0


def cmd_fills(a) -> int:
    from markout.lob import lobster
    from markout.lob.fills import MODELS
    from markout.lob.markouts import HORIZONS, order_markouts, sample_orders, summarize_markouts

    models = tuple(m.strip() for m in a.models.split(","))
    if bad := [m for m in models if m not in MODELS]:
        raise SystemExit(f"unknown fill model {bad[0]!r}; choose from {', '.join(MODELS)}")
    if a.ticker:
        t = a.ticker.upper()
        if not lobster.available((t,)):
            raise SystemExit(f"no sample day for {t}: run `make lobster` (bundled: {', '.join(lobster.TICKERS)})")
        day = lobster.load_day(t)
    elif a.messages and a.book:
        day = lobster.read_csv(a.messages, a.book)
    else:
        raise SystemExit("give MESSAGES.csv BOOK.csv, or --ticker for a bundled sample day")
    t0, t1 = float(day.messages.time[0]), float(day.messages.time[-1])
    start, end = max(_clock(a.start), t0), min(_clock(a.end), t1)
    sample = sample_orders(day, every=a.every, max_wait=a.wait, size=a.size, session=(start, end))
    if len(sample.orders) == 0:
        need = a.wait + max(HORIZONS)
        raise SystemExit(
            f"no decision times fit: the window {start:.0f}-{end:.0f} s must exceed wait + {max(HORIZONS):g} s ({need:g} s)"
        )
    df = order_markouts(day, sample, models=models, engine=a.engine)
    if a.out:
        df.to_csv(a.out, index=False)
    rows = summarize_markouts(df)
    out = {
        "ticker": day.ticker,
        "messages": len(day),
        "levels": day.book.levels,
        "window_s": [start, end],
        "orders": len(sample.orders),
        "every_s": a.every,
        "max_wait_s": a.wait,
        "size": a.size,
        "summary": rows,
    }
    if a.json:
        _emit(out, True)
        return 0
    print(
        f"{day.ticker}: {len(day):,} messages, {day.book.levels} levels; {len(sample.orders):,} orders "
        f"(both sides every {a.every:g} s, {a.size} shares, resting up to {a.wait:g} s)"
    )
    print("markouts in ticks, quantity-weighted over filled orders; 'random' = an instant fill at the decision time")
    hz = [h for h in HORIZONS if h >= 1]
    head = f"{'model':<8} {'P(fill)':>8} " + " ".join(f"{f'mk {h:g}s':>8}" for h in hz) + f" {'random 10s':>11}"
    print(head)
    for r in rows:
        mk = " ".join(f"{r[f'mk_{h:g}']:>+8.3f}" for h in hz)
        print(f"{r['model']:<8} {r['p_fill']:>8.3f} {mk} {r['uncond_10']:>+11.3f}")
    if a.out:
        print(f"per-order rows written to {a.out}")
    return 0


def cmd_sharpe(a) -> int:
    from markout.evaluation import bootstrap, dsr

    r = _series(a.path, a.column)
    trial_srs = _series(a.trial_sharpes, None) if a.trial_sharpes else None
    if trial_srs is None and a.trials > 1 and a.sr_var is None:
        raise SystemExit("--trials > 1 needs --sr-var (variance of the trials' per-period Sharpes) or --trial-sharpes")
    d = dsr.deflated_sharpe(
        r, trial_srs=trial_srs, var_sr=a.sr_var, n_trials=None if trial_srs is not None else a.trials
    )
    ci = bootstrap.sharpe_ci(r, n_boot=a.boot, seed=a.seed)
    k = math.sqrt(a.periods)
    out = {
        "T": d["T"],
        "sharpe": d["sr"],
        "sharpe_annualized": d["sr"] * k,
        "ci95_annualized": [ci["lo"] * k, ci["hi"] * k],
        "block_length": ci["block"],
        "skew": d["skew"],
        "kurtosis": d["kurt"],
        "psr_0": d["psr_0"],
        "n_trials": d["n_trials"],
        "sr_star_annualized": d["sr_star"] * k,
        "dsr": d["dsr"],
    }
    rows = [
        ("observations", out["T"]),
        (f"Sharpe, annualized (x sqrt {a.periods:g})", out["sharpe_annualized"]),
        ("95% CI (stationary bootstrap)", f"[{_fmt(out['ci95_annualized'][0])}, {_fmt(out['ci95_annualized'][1])}]"),
        ("skew / kurtosis", f"{_fmt(out['skew'])} / {_fmt(out['kurtosis'])}"),
        ("P(true Sharpe > 0), PSR", out["psr_0"]),
        ("trials searched", out["n_trials"]),
        ("best Sharpe expected by luck, annualized", out["sr_star_annualized"]),
        ("Deflated Sharpe (P > luck)", out["dsr"]),
    ]
    _emit(out, a.json, rows)
    return 0


def cmd_pbo(a) -> int:
    from markout.evaluation import pbo

    df = _frame(a.path)
    if a.index:
        df = df.set_index(a.index)
    res = pbo(df.select_dtypes("number"), n_blocks=a.blocks)
    top = res["selection_freq"].sort_values(ascending=False).head(5)
    out = {
        "pbo": res["pbo"],
        "prob_oos_loss": res["prob_oos_loss"],
        "degradation_slope": res["degradation_slope"],
        "n_strategies": res["n_strategies"],
        "T": res["T"],
        "n_combinations": res["n_used"],
        "most_selected": {str(k): float(v) for k, v in top.items()},
    }
    rows = [
        ("strategies x periods", f"{out['n_strategies']} x {out['T']}"),
        ("CSCV splits", out["n_combinations"]),
        ("PBO: P(in-sample winner below the OOS median)", out["pbo"]),
        ("P(in-sample winner loses out of sample)", out["prob_oos_loss"]),
        ("OOS vs IS performance slope", out["degradation_slope"]),
        ("most often selected", ", ".join(f"{k} ({v:.0%})" for k, v in out["most_selected"].items())),
    ]
    _emit(out, a.json, rows)
    return 0


def cmd_kelly(a) -> int:
    from markout.decision import kelly

    if not (0 < a.price < 1 and 0 <= a.p <= 1):
        raise SystemExit("need 0 < price < 1 and 0 <= p <= 1")
    side, full = kelly.best_side(a.p, a.price)
    f = float(kelly.fractional(full, a.fraction))
    out = {
        "side": side if full > 0 else None,
        "kelly": full,
        "fraction": a.fraction,
        "stake_share": f,
        "edge_per_unit": float(kelly.expected_return(a.p, a.price, side)),
        "log_growth_per_bet": float(kelly.growth_rate(f, a.p, a.price, side)),
        "stake": f * a.bankroll if a.bankroll else None,
    }
    rows = [
        ("buy", f"{side} at {a.price if side == 'YES' else 1 - a.price:.4g}" if full > 0 else "nothing (no edge)"),
        ("full Kelly, share of wealth", full),
        (f"{a.fraction:g} x Kelly", f),
        ("expected profit per unit staked", out["edge_per_unit"]),
        ("expected log growth per bet", out["log_growth_per_bet"]),
    ]
    if a.bankroll:
        rows.append((f"stake out of {a.bankroll:g}", out["stake"]))
    _emit(out, a.json, rows)
    return 0


def cmd_report(a) -> int:
    names = list(REPORTS) if a.name == "all" else [a.name]
    for n in names:
        print(f"== {n}", flush=True)
        r = subprocess.run([sys.executable, "-m", REPORTS[n]])
        if r.returncode:
            return r.returncode
    return 0


def cmd_site(a) -> int:
    from markout import site

    site.main()
    return 0


def cmd_readme(a) -> int:
    from markout import readme

    readme.main()
    return 0


def cmd_desk(a) -> int:
    from markout.cup.__main__ import main as desk

    return desk(a.args or ["--help"])


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="markout", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--version", action="store_true", help="print the version and exit")
    sub = ap.add_subparsers(dest="cmd")

    s = sub.add_parser("status", help="what is on disk: data, the C++ extension, results")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_status)

    f = sub.add_parser("fills", help="queue-position fills and markouts for a LOBSTER day")
    f.add_argument("messages", nargs="?", help="LOBSTER message file (CSV)")
    f.add_argument("book", nargs="?", help="LOBSTER order-book file (CSV), row-aligned with the messages")
    f.add_argument("--ticker", help="a bundled sample day instead (AAPL, AMZN, GOOG, INTC, MSFT)")
    f.add_argument("--models", default="touch,fifo,through", help="comma-separated: touch, fifo, through")
    f.add_argument("--every", type=float, default=5.0, help="seconds between decision times (default 5)")
    f.add_argument("--wait", type=float, default=60.0, help="seconds an order may rest (default 60)")
    f.add_argument("--size", type=int, default=100, help="shares per order (default 100)")
    f.add_argument("--start", default="09:35", help="first decision time, HH:MM[:SS] or seconds (default 09:35)")
    f.add_argument("--end", default="15:55", help="end of the window, HH:MM[:SS] or seconds (default 15:55)")
    f.add_argument("--engine", default="auto", choices=["auto", "python", "cpp"])
    f.add_argument("--out", help="write one row per order and model to this CSV")
    f.add_argument("--json", action="store_true")
    f.set_defaults(fn=cmd_fills)

    sh = sub.add_parser("sharpe", help="Sharpe with a bootstrap CI, PSR and the Deflated Sharpe")
    sh.add_argument("path", help="CSV or Parquet of per-period PnL or returns")
    sh.add_argument("--column", help="which column (default: the only numeric one)")
    sh.add_argument("--periods", type=float, default=252, help="periods per year, for display (default 252)")
    sh.add_argument("--trials", type=int, default=1, help="how many variants were tried before this one")
    sh.add_argument("--sr-var", type=float, help="variance of the trials' per-period Sharpe ratios")
    sh.add_argument("--trial-sharpes", help="CSV with one per-period Sharpe per trial (sets trials and variance)")
    sh.add_argument("--boot", type=int, default=5000, help="bootstrap replications (default 5000)")
    sh.add_argument("--seed", type=int, default=0)
    sh.add_argument("--json", action="store_true")
    sh.set_defaults(fn=cmd_sharpe)

    pb = sub.add_parser("pbo", help="probability of backtest overfitting (CSCV)")
    pb.add_argument("path", help="CSV or Parquet: one row per period, one numeric column per strategy")
    pb.add_argument("--index", help="a date or period column to drop from the matrix")
    pb.add_argument("--blocks", type=int, default=16, help="even number of CSCV blocks (default 16)")
    pb.add_argument("--json", action="store_true")
    pb.set_defaults(fn=cmd_pbo)

    k = sub.add_parser("kelly", help="Kelly stake for a binary contract")
    k.add_argument("--p", type=float, required=True, help="your probability that the event happens")
    k.add_argument("--price", type=float, required=True, help="price of the YES contract (pays 1)")
    k.add_argument("--fraction", type=float, default=1.0, help="fractional Kelly multiplier (default 1)")
    k.add_argument("--bankroll", type=float, help="print the stake in currency")
    k.add_argument("--json", action="store_true")
    k.set_defaults(fn=cmd_kelly)

    r = sub.add_parser("report", help="regenerate a report from its data")
    r.add_argument("name", choices=[*REPORTS, "all"])
    r.set_defaults(fn=cmd_report)
    sub.add_parser("site", help="rebuild docs/index.html").set_defaults(fn=cmd_site)
    sub.add_parser("readme", help="rebuild README.md").set_defaults(fn=cmd_readme)
    d = sub.add_parser("desk", help="the Predictions Cup desk (python -m markout.cup)", add_help=False)
    d.add_argument("args", nargs=argparse.REMAINDER)
    d.set_defaults(fn=cmd_desk)
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = parser()
    a = ap.parse_args(argv)
    if a.version:
        from markout import __version__

        print(__version__)
        return 0
    if not a.cmd:
        ap.print_help()
        return 0
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
