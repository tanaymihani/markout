"""Build README.md from the generated reports and results: `python -m markout.readme`.

Each module's summary is lifted verbatim from its generated report, so the README can
never disagree with the reports, and every number in it traces back to code.
"""

from __future__ import annotations

import re
import subprocess
import sys

from markout import plotting as mp
from markout import results
from markout.paths import FIGURES, REPORTS, ROOT


def _read(name: str) -> str | None:
    p = REPORTS / f"{name}.md"
    return p.read_text() if p.exists() else None


def _paragraph_starting(md: str, prefix: str) -> str | None:
    for block in re.split(r"\n\s*\n", md):
        b = block.strip()
        if b.startswith(prefix):
            return b
    return None


def _first_paragraph_after(md: str, heading: str) -> str | None:
    i = md.find(heading)
    if i < 0:
        return None
    for block in re.split(r"\n\s*\n", md[i + len(heading):]):
        b = block.strip()
        if b and not b.startswith(("#", "|", "<", "*Part", "*Generated", "*First")):
            return b
    return None


def _bullets_after_title(md: str) -> str | None:
    lines, out, started = md.splitlines(), [], False
    for ln in lines[1:]:
        if ln.startswith("- "):
            out.append(ln)
            started = True
        elif started and ln.strip() == "":
            break
        elif started:
            out.append(ln)
    return "\n".join(out) if out else None


def _fig(name: str, alt: str) -> str:
    return mp.picture(name, alt, prefix="reports/figures/") if (FIGURES / f"{name}.png").exists() else ""


def _n_tests() -> str:
    try:
        out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"], cwd=ROOT,
                             capture_output=True, text=True, timeout=300).stdout
        m = re.search(r"(\d+) tests? collected", out)
        if m:
            return m.group(1)
        per_file = [int(x) for x in re.findall(r"^\S+\.py: (\d+)$", out, flags=re.M)]  # -qq output
        return str(sum(per_file)) if per_file else "?"
    except Exception:  # noqa: BLE001
        return "?"


def section_auction() -> list[str]:
    md = _read("01_auction")
    out = ["### 01 · Closing-auction alpha: does it survive costs? · [report](reports/01_auction.md)", ""]
    if md is None:
        out += ["**Pending real data.** The pipeline (audit, causal features, walk-forward LightGBM, the "
                "cost-aware decision rule, the trial registry, Deflated Sharpe, PBO and the one-shot holdout) "
                "is built and runs end to end on synthetic data with the Optiver schema "
                "(`python -m markout.auction.report --synthetic`). The real run needs the Kaggle data: "
                "`kaggle auth login`, accept the competition rules, then `make optiver && python -m "
                "markout.auction.report`.", ""]
        return out
    ans = _paragraph_starting(md, "**Answer.**")
    out += [ans or "", "", _fig("b_voi", "Value of information: net edge vs forecast IC"), ""]
    return out


def section_micro() -> list[str]:
    md = _read("02_microstructure") or ""
    return ["### 02 · Microstructure: does a signal survive realistic fills? · "
            "[report](reports/02_microstructure.md)", "",
            _paragraph_starting(md, "**Answer.**") or "", "",
            _fig("d_markouts", "Markouts of filled orders by fill model"), ""]


def section_arena() -> list[str]:
    md = _read("03_arena") or ""
    lead = next((b.strip() for b in re.split(r"\n\s*\n", md) if b.strip().startswith("**")), "")
    return ["### 03 · Arena: game theory under simulated competition · [report](reports/03_arena.md)", "",
            lead, "", _fig("e_arena_competition", "Spread and maker profit vs number of competing makers"), ""]


def section_options() -> list[str]:
    md = _read("04_options") or ""
    return ["### 04 · Options: Black–Scholes, delta hedging, the SPY smile · [report](reports/04_options.md)", "",
            _bullets_after_title(md) or "", "", _fig("f_hedge_frontier", "Hedging frequency vs cost frontier"), ""]


def section_cup() -> list[str]:
    md = _read("05_predictions_cup") or ""
    lead = _first_paragraph_after(md, "### What it takes to win")
    return ["### 05 · SIG Predictions Cup: sizing for a winner-take-most contest · "
            "[report](reports/05_predictions_cup.md)", "",
            "A sizing study written and committed before the contest opens (Oct 1 2026), plus a decision "
            "journal that scores every bet afterwards (Brier and log score against the market, calibration, "
            "skill vs luck). " + (lead or ""), "", _fig("p_sizing", "Contest outcomes by Kelly fraction"), ""]


def section_cpp() -> list[str]:
    b = results.load("cpp_bench")
    if not b:
        return []
    return ["### C++ core · [section in report 02](reports/02_microstructure.md#c-core)", "",
            f"The order-book replay and FIFO queue simulator is sequential and stateful, the one part worth "
            f"porting. `cpp/queue_sim.cpp` (C++17, pybind11) is {b['speedup_min']:.0f}–{b['speedup_max']:.0f}× "
            f"faster than the Python reference (median {b['speedup_median']:.0f}×) and "
            f"{'bit-identical to it on every run' if b['all_identical'] else 'NOT identical to it'} "
            "(checked on the five real days and on randomized `hypothesis` event streams).", "",
            _fig("g_throughput", "C++ vs Python throughput"), ""]


def section_desk() -> list[str]:
    demo = ROOT / "docs" / "demo" / "DEMO.md"
    shot = ROOT / "docs" / "demo" / "desk.png"
    out = ["## The product: Markout Desk for the SIG Predictions Cup", "",
           "A trading desk for SIG's student prediction-market contest (Oct 1 – Nov 4 2026), whose rules allow bots "
           "(one account, individual participation, the platform's rate and position limits). It runs on a laptop "
           "and uses only free public data. Each step of the research above appears in it:", "",
           "- **Reference prices** from real-money markets (Polymarket, Kalshi), option-implied probabilities for "
           "price questions and earnings-beat histories. Every contest-to-reference mapping must be confirmed by a "
           "human (matching is where things go silently wrong).",
           "- **Estimates** pool the references with the contest's own price in log-odds, so the bot never treats a "
           "reference as the truth: the optimizer's-curse correction from module 01.",
           "- **Proposals** need the whole uncertainty band to clear the ask, are sized with Kelly as a target "
           "*exposure* (the policy pre-registered in report 05, or a steady 0.5x Kelly mode), and are sized to the "
           "book: whatever the book can't fill now is not bought, never left resting (module 02's adverse "
           "selection).",
           "- **Nothing trades without a click.** Every approved or declined proposal is written to the decision "
           "journal, which scores the beliefs against the market once markets resolve (report 05, Part 2).", ""]
    if shot.exists():
        out += ["![The approval desk](docs/demo/desk.png)", ""]
    if demo.exists():
        out += ["[Demo walkthrough](docs/demo/DEMO.md): the whole loop on fictional markets, offline "
                "(`make demo`). Run the desk yourself with `make desk` and open http://127.0.0.1:8765.", ""]
    dry = ROOT / "docs" / "demo" / "DRYRUN.md"
    if dry.exists():
        out += ["[Matcher dry run on live markets](docs/demo/DRYRUN.md): stand-in contests built from live "
                "Polymarket and Kalshi questions, with every cross-venue suggestion listed "
                "(`python -m markout.cup dryrun`). The first version's top suggestions were mostly look-alikes "
                "(the other team, another stat line, a different threshold); those cases are now regression tests.",
                ""]
    return out


def build() -> str:
    a = results.load("auction") or {}
    hold = (a.get("holdout") or {})
    guard = (f"accessed {hold['accessed']} time{'s' if hold.get('accessed') != 1 else ''}"
             if hold.get("accessed") else "sealed, not yet run")
    n_tests = _n_tests()
    lines = [
        "# Markout", "",
        "**Does a short-horizon edge survive costs, fills, and competition?**", "",
        "A research project that takes trading signals from prediction to PnL and measures what kills "
        "them at each step: the spread, selection bias across many backtests, realistic queue-position "
        "fills, and competing traders. It spans ML forecasting of the NASDAQ closing auction, order-level "
        "microstructure (LOBSTER), game theory (Glosten–Milgrom, Kyle, a market-making tournament, "
        "Kuhn-poker CFR), options (delta-hedging PnL, the SPY smile) and decision science (value of "
        "information, calibration, Kelly vs tournament sizing), with a C++ core.", "",
        "**The thread through every module is the winner's curse: whatever you select on looks better "
        "than it is.** A *markout*, the price move right after a trade, is how market makers measure it.", "",
        "| Where it shows up | Form of the curse | Where Markout measures it |",
        "|---|---|---|",
        "| Picking the best of N backtests | selection bias under multiple testing | 01: Deflated Sharpe, PBO |",
        "| Acting on your largest forecasts | the optimizer's curse | 01: calibration and shrinkage |",
        "| The passive order that got filled | adverse selection | 02: queue-position fills, markouts |",
        "| Quoting tighter than every rival | the dealer's winner's curse | 03: Glosten–Milgrom, tournament |",
        "| Betting where you most disagree with the market | selecting on your own errors | 05: journal |",
        "",
    ]
    lines += section_desk()
    lines += [
        "## Results", "",
        "Every number below is lifted from a generated report; none is typed by hand.", "",
    ]
    lines += section_auction() + section_micro() + section_arena() + section_options() + section_cup() \
        + section_cpp()
    lines += [
        "## What keeps it honest", "",
        f"- **{n_tests} tests** (`make test`), including a perturbation test that corrupts the future and "
        "requires every feature of the past to stay identical, with a negative control that proves the "
        "test catches a leak.",
        "- **Walk-forward by day** with a calibration block, never a random split. Purging is unnecessary "
        "because labels never cross a day; the reasoning is in `src/markout/auction/cv.py`.",
        "- **Every trial is logged** (`data/registry/trials.jsonl`, with git SHA and config hash), losers "
        "included, so the Deflated Sharpe and PBO see how much searching was done.",
        f"- **The holdout is guarded**: `HoldoutGuard` refuses a second look and audits forced ones "
        f"(holdout {guard}).",
        "- **Replications are checked against closed forms**: the DSR paper's worked example, GM's spread "
        "at π = ½, Kyle's λ, Kuhn poker's −1/18, Black–Scholes parity and Greeks.",
        "- **Every report is generated** by `make report` from fixed seeds; prose that depends on a result "
        "is chosen by code.", "",
        "## Reproduce", "",
        "```bash",
        "make install     # editable install into .venv",
        "make lobster     # LOBSTER sample day (5 stocks, 10 levels)",
        "make optiver     # Optiver closing-auction data (needs `kaggle auth login` + accepted rules)",
        "make cpp         # build the C++ extension (optional; pure-Python fallback)",
        "make test        # the test suite",
        "make report      # regenerate every report, figure and results file",
        "make demo        # the desk's offline demo -> docs/demo/",
        "make desk        # the desk on a paper contest: http://127.0.0.1:8765",
        "python -m markout.readme   # regenerate this README",
        "```", "",
        "Other entry points: `python -m markout.backtest.report` (daily PnL report), "
        "`python -m markout.games.cardgame` (market-making practice game), "
        "`python -m markout.decision.journal init|add|score` (Predictions Cup journal), `make bench`.", "",
        "## Layout", "",
        "```",
        "src/markout/auction/     closing-auction data, audit, causal features, walk-forward models, report 01",
        "src/markout/backtest/    cost model, decision rule, sizing, daily PnL",
        "src/markout/decision/    calibration, value of information, Kelly, contest study, journal",
        "src/markout/evaluation/  trial registry, Deflated Sharpe, PBO, bootstrap, holdout guard, Diebold–Mariano",
        "src/markout/lob/         LOBSTER parser, OFI, fill simulator, markouts, post-vs-cross, report 02",
        "src/markout/games/       Glosten–Milgrom, Kyle, market-making arena, Kuhn CFR, card game, report 03",
        "src/markout/options/     Black–Scholes, delta hedging, SPY smile, report 04",
        "src/markout/cup/         Predictions Cup desk: sources, matcher, engine, sizing, paper exchange, web desk",
        "cpp/                     C++17 queue simulator + pybind11 bindings",
        "reports/                 generated reports, figures (light + dark) and results JSON",
        "tests/                   pytest suite",
        "```", "",
        "## Data", "",
        "- **Optiver *Trading at the Close*** (Kaggle): 200 NASDAQ stocks, 481 days, 10-second snapshots of "
        "the closing auction. Kaggle's rules forbid redistribution, so it is downloaded, never committed.",
        "- **LOBSTER sample files**: AAPL, AMZN, GOOG, INTC, MSFT on 2012-06-21 at 10 levels. The official "
        "download links stopped working when lobsterdata.com was rebuilt, so `scripts/download_lobster.py` "
        "tries them first and otherwise uses a pinned, hash-verified mirror; a 100% message/book consistency "
        "replay is the authenticity check. Not committed.",
        "- **SPY option chain**: one snapshot taken with yfinance on 2026-09-29 after the close (see report 04).", "",
        "## Limitations", "",
        "Each report ends with its own. The big ones: the auction target is relative to a synthetic index, so "
        "PnL assumes a hedge; the microstructure study is a single 2012 day with no impact from our own "
        "orders; the arena is a stylized dealer market with one shared belief; the contest study's rules, "
        "field and markets are assumptions.", "",
    ]
    return "\n".join(ln for ln in lines if ln is not None) + "\n"


def main() -> None:
    (ROOT / "README.md").write_text(build())
    print("wrote README.md")


if __name__ == "__main__":
    main()
