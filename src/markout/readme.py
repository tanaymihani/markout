"""Build README.md from the generated reports: `python -m markout.readme`.

The results paragraphs are copied from the reports, so the README never disagrees with
them and every number traces back to code.
"""

from __future__ import annotations

import re
import subprocess
import sys

from markout import plotting as mp
from markout import replications, results
from markout.paths import FIGURES, REPORTS, ROOT
from markout.site import DESK_URL


def _read(name: str) -> str:
    p = REPORTS / f"{name}.md"
    return p.read_text() if p.exists() else ""


def _blocks(md: str) -> list[str]:
    return [b.strip() for b in re.split(r"\n\s*\n", md) if b.strip()]


def _starting(md: str, prefix: str) -> str:
    return next((b for b in _blocks(md) if b.startswith(prefix)), "")


def _after(md: str, heading: str) -> str:
    i = md.find(heading)
    if i < 0:
        return ""
    return next(
        (
            b
            for b in _blocks(md[i + len(heading) :])
            if not b.startswith(("#", "|", "<", "*Part", "*Generated", "*First"))
        ),
        "",
    )


def _first_bullets(md: str) -> str:
    out, started = [], False
    for ln in md.splitlines()[1:]:
        if ln.startswith("- "):
            out.append(ln)
            started = True
        elif started and not ln.strip():
            break
        elif started:
            out.append(ln)
    return "\n".join(out)


def _fig(name: str, alt: str) -> str:
    return mp.picture(name, alt, prefix="reports/figures/") if (FIGURES / f"{name}.png").exists() else ""


def _n_tests() -> str:
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        ).stdout
        m = re.search(r"(\d+) tests? collected", out)
        if m:
            return m.group(1)
        per_file = [int(x) for x in re.findall(r"^\S+\.py: (\d+)$", out, flags=re.M)]
        return str(sum(per_file)) if per_file else "?"
    except Exception:  # noqa: BLE001
        return "?"


def _answer(md: str) -> str:
    return re.sub(r"^\*\*Answer\.\*\*\s*", "", _starting(md, "**Answer.**"))


def results_section() -> list[str]:
    out = ["## Results", "", "The paragraphs below are copied from the reports.", ""]
    auction = _read("01_auction")
    if auction:
        out += [
            "**01 · Closing auction.** " + _answer(auction),
            "",
            _fig("b_voi", "Net edge per trade against forecast quality"),
            "",
        ]
    else:
        out += ["**01 · Closing auction.** Waiting for the Kaggle data (`make optiver`).", ""]
    micro = _read("02_microstructure")
    out += [
        "**02 · Microstructure.** Queue imbalance predicts the next price move, but does trading on it pay "
        "once fills follow queue priority? " + _answer(micro).replace("**No.** ", "No. "),
        "",
        _fig("d_markouts", "Markouts of filled orders under each fill model"),
        "",
    ]
    arena = _read("03_arena")
    lead = next((b for b in _blocks(arena) if b.startswith("In short:")), "")
    out += [
        "**03 · Arena.** " + lead.replace("In short: ", "").capitalize()[:1] + lead.replace("In short: ", "")[1:],
        "",
        _fig("e_arena_competition", "Spread and market-maker profit as competitors are added"),
        "",
    ]
    out += ["**04 · Options.**", "", _first_bullets(_read("04_options")), ""]
    cup = _after(_read("05_predictions_cup"), "### What it takes to win")
    if cup:
        out += ["**05 · Predictions Cup sizing.** " + cup, ""]
    vol = _read("06_vol_premium")
    if vol:
        out += [
            "**06 · Volatility premium.** " + _answer(vol),
            "",
            _fig("h_kelly", "Kelly sizing chosen before 2008, lived through it"),
            "",
        ]
    many = _read("07_many_days")
    if many:
        out += [
            "**07 · Many days.** Does the gap from 02 survive ten more days and ten stocks? " + _answer(many),
            "",
            _fig("g_gap_heatmap", "Touch-minus-FIFO markout gap for every day and stock"),
            "",
        ]
    l2 = _read("08_l2_and_latency")
    if l2:
        out += [
            "**08 · L2 backtests and latency.** " + _answer(l2),
            "",
            _fig("i_l2_error", "Fill-probability and markout errors of an L2-only backtest, by stock"),
            "",
        ]
    b = results.load("cpp_bench")
    if b:
        out += [
            (
                f"**C++.** The order-book replay and queue simulator from part 02 is sequential and keeps state, "
                f"so it was the one piece worth porting. `cpp/queue_sim.cpp` (C++17 with pybind11) runs "
                f"{b['speedup_min']:.0f}–{b['speedup_max']:.0f}× faster than the Python version and gives "
                f"{'identical output' if b['all_identical'] else 'different output (see report 02)'}, checked on "
                "all five stocks and on randomized event streams."
            ),
            "",
            _fig("g_throughput", "C++ against Python throughput"),
            "",
        ]
    return out


def desk_section() -> list[str]:
    out = [
        "## The Predictions Cup desk",
        "",
        (
            "SIG runs a free prediction-market contest for students (Oct 1 to Nov 4, 2026), and its rules allow "
            "bots as long as each person keeps one account and stays within the platform's limits. I built this "
            "desk for it. It runs on a laptop, uses only free public data, and never trades without a click."
        ),
        "",
        (
            "- It looks up reference prices for each contest market: real-money prices from Polymarket and Kalshi, "
            "option-implied probabilities for stock-price questions, and earnings-beat history. Every suggested "
            "match between a contest market and a reference has to be confirmed by hand, because matching is where "
            "things quietly go wrong."
        ),
        (
            "- It pools those references with the contest's own price in log-odds, so it never treats a single "
            "reference as the truth."
        ),
        (
            "- It proposes a trade only when the whole uncertainty band clears the ask. Size comes from Kelly as a "
            "target exposure (either the pre-registered contest policy from part 05, or a steadier half-Kelly), "
            "capped by what the order book can fill right now. Nothing is left resting on the book."
        ),
        (
            "- Every decision, including the ones I skip, goes into a journal that scores my forecasts against the "
            "market once the questions resolve."
        ),
        "",
    ]
    if (ROOT / "docs" / "demo" / "desk.png").exists():
        out += ["![The approval desk](docs/demo/desk.png)", ""]
    out += [
        (
            "The [demo](docs/demo/DEMO.md) runs the whole loop offline on made-up markets (`make demo`); `make desk` "
            "starts it on a paper contest at http://127.0.0.1:8765. The [matcher dry run](docs/demo/DRYRUN.md) "
            "tests the matching against live Polymarket and Kalshi questions. Its first version mostly suggested "
            "look-alikes (the other team, another stat line, a different threshold); those cases are now tests."
        ),
        "",
        f"**[Open the desk in your browser]({DESK_URL}).** It runs on a fictional 8-market scenario: Pyodide "
        "(Python compiled to WebAssembly) executes the desk's own Python in the tab, so every visitor gets a private "
        "paper-trading sandbox, with its "
        "own exchange, journal and clock, and nothing is sent anywhere. `python -m markout.cup public` (or the Docker "
        "image) serves the same sandboxes from one server, with header sessions, eviction and expiry, rate limits and "
        "a strict Content-Security-Policy; [ARCHITECTURE.md](ARCHITECTURE.md) has the details and "
        "[deploy/README.md](deploy/README.md) the deployment.",
        "",
    ]
    return out


def brief() -> list[str]:
    """The headline numbers, for readers who stop after the first screen."""
    out = []
    a = results.load("auction")
    if a:
        s, h = a["selected"]["summary"], (a.get("holdout") or {}).get("adapted_1x", {})
        two = next(r for r in a["robustness"]["cost_rows"] if r["cost_mult"] == 2.0)["adapted"]["sharpe_annualized"]
        share = a["diebold_mariano"]["share_days_better"]
        days = "every out-of-sample day" if share == 1 else f"{100 * share:.0f}% of out-of-sample days"
        out.append(
            f"**Closing auction.** A LightGBM model beats a per-stock baseline on {days}. Over {s['days']} "
            f"walk-forward days, traded net of costs at a size the order book can absorb, it has an annualized "
            f"Sharpe of {s['sharpe_annualized']:.1f} but nets only ${s['total_pnl_usd']:,.0f}, because costs take "
            f"{s['cost_usd'] / s['gross_usd']:.0%} of the gross; at twice the assumed costs the Sharpe is {two:.1f}. "
            f"On a {h.get('days', 0)}-day holdout I opened once, the frozen model scored "
            f"{h.get('sharpe_annualized', float('nan')):.1f} (${h.get('total_pnl_usd', float('nan')):,.0f}): "
            "real, but thin."
        )
    m = results.load("microstructure")
    if m:
        intc = {r["model"]: r for r in m["fills"]["INTC"]["summary"]}
        out.append(
            f"**Fills.** Resting orders in INTC look profitable if any trade at your price fills you "
            f"({intc['touch']['mk_10_bps']:+.2f} bps ten seconds later). Simulating queue position order by "
            f"order turns that into {intc['fifo']['mk_10_bps']:+.2f} bps: the fills you actually get are the "
            "ones you didn't want."
        )
    md = results.load("many_days")
    if md and md.get("cells"):
        pool, l2p = md["pooled"], md["l2"]["pooled"]
        n_days, n_stocks = len(md["meta"]["days"]), len(md["by_ticker"])
        bt = md["by_ticker"]
        losing = sum(b["mk10_bps"]["touch"]["mean"] < 0 for b in bt.values())
        out.append(
            f"**Fills across days.** On {n_days} NASDAQ ITCH sample days of 2019–2021 and {n_stocks} stocks "
            f"({md['throughput']['itch_messages_read'] / 1e9:.1f} billion messages), the real queue makes the markout "
            f"worse than a touch-fill backtest on {100 * pool['share_cells_gap_positive']:.0f}% of stock-days, by "
            f"{pool['median_stock_gap_bps']:.2f} bps for the median stock. The 2012 sign flip is not general: on "
            f"these days touch fills already show a loss for {losing} of the {n_stocks} stocks. A backtest that only "
            f"sees aggregated depth {'overstates' if l2p['dp_fill_pp']['mean'] > 0 else 'understates'} the fill "
            f"rate by {abs(l2p['dp_fill_pp']['mean']):.1f} points on average."
        )
    ar = results.load("arena")
    if ar:
        k = {r["K"]: r for r in ar["arena"]["k_sweep"]}
        out.append(
            f"**Competition.** A lone market maker quotes a {100 * k[1]['quoted_spread']['mean']:.0f}-tick "
            f"spread; one identical rival brings it to {100 * k[2]['quoted_spread']['mean']:.1f}, next to the "
            f"zero-profit {100 * k[2]['zero_profit_spread']['mean']:.1f}. Competition removes the rent, not the "
            "cost of trading against informed flow."
        )
    v = results.load("vol_premium")
    if v:
        kf, sd = v["kelly_full"], v.get("straddle")
        text = (
            f"**Volatility premium.** VIX sat above the volatility that followed "
            f"{100 * v['share_implied_above']:.0f}% of the time since 1990, but the textbook Kelly formula asks for "
            f"{kf['v_continuous'] / kf['v_star']:.1f}x the growth-optimal size"
            + (
                ", and every Kelly fraction fitted on 1993–2007 was wiped out after 2008."
                if all(x["busted"] for x in v["kelly_after_2008"])
                else "."
            )
        )
        if sd:
            vs, st = sd["instruments"]["var_swap"]["worst_pnl"], sd["instruments"]["straddle"]["worst_pnl"]
            text += (
                f" Selling delta-hedged straddles instead of variance cuts the worst month from {-vs:.0f} to "
                f"{-st:.0f} per $1 of vega, though at-the-money options carry less of the premium."
            )
        out.append(text)
    c = (results.load("predictions_cup") or {}).get("contest")
    if c:
        cfg, k1 = c["config"], next(r for r in c["base"] if r["policy"] == "1x Kelly")
        best = max(c["base"], key=lambda r: r["p_top3"])
        placed = "never finishes" if k1["p_top3"] == 0 else f"finishes {100 * k1['p_top3']:.1f}% of the time"
        out.append(
            f"**Contest sizing.** In a simulated {cfg['n_players']:,}-player contest that pays only the top 3, "
            f"1x Kelly {placed} in the money over {cfg['n_sims']:,} runs. The best policy tried "
            f"({best['policy']}) gets there {100 * best['p_top3']:.1f}% of the time, "
            f"{best['p_top3'] / (3 / cfg['n_players']):.0f}x a random player's odds, and busts in "
            f"{100 * best['p_bust']:.0f}% of runs."
        )
    return (["## In brief", ""] + [f"- {b}" for b in out] + [""]) if out else []


def _itch_audit() -> str:
    """The all-symbol check of the ITCH days, when report 07's results carry it."""
    aud = (results.load("many_days") or {}).get("audit") or {}
    if not aud:
        return ""
    refs, bad = sum(a["references"] for a in aud.values()), sum(a["unmatched"] for a in aud.values())
    return (
        f" (checked for every symbol in the files too: {refs / 1e9:.1f} billion references, "
        f"{'none' if bad == 0 else f'{bad:,}'} unmatched)"
    )


def replications_section() -> list[str]:
    rep = results.load("replications")
    body = replications.table(rep) if rep else ""
    if not body:
        return []
    return [
        "### Replications",
        "",
        (
            "Published numbers next to the ones this code gets on its own data (`python -m markout.replications`). "
            "Where the setups differ, the last column says how."
        ),
        "",
        body,
    ]


def build() -> str:
    a = results.load("auction") or {}
    times = (a.get("holdout") or {}).get("accessed")
    lines = [
        "# Markout",
        "",
        (
            "[![tests](https://github.com/tanaymihani/markout/actions/workflows/tests.yml/badge.svg)]"
            "(https://github.com/tanaymihani/markout/actions/workflows/tests.yml) "
            "[![codeql](https://github.com/tanaymihani/markout/actions/workflows/codeql.yml/badge.svg)]"
            "(https://github.com/tanaymihani/markout/actions/workflows/codeql.yml) "
            "[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)"
        ),
        "",
        (
            "Does a short-horizon trading edge survive the spread, realistic fills, competing traders, and the bias "
            "that comes from trying many strategies? This repo is my attempt to find out, step by step, on real "
            "market data where I could get it."
        ),
        "",
        "**Live labs:** [tanaymihani.github.io/markout](https://tanaymihani.github.io/markout/) has four "
        f"interactive labs, a market-making game and the [trading desk]({DESK_URL}), all running in the browser.",
        "",
        (
            "[![The Markout lab: a simulated order waits in the queue while a touch-fill backtest has already counted "
            "it as filled](docs/media/markout-demo.gif)](https://tanaymihani.github.io/markout/)"
        ),
        "",
        (
            "The name comes from the *markout*, the price move right after a trade, which market makers use to check "
            "whether they got picked off. The same trap shows up at every stage here: the best of many backtests "
            "looks better than it is, and so do the order that happened to get filled and the bet where you disagree "
            "most with the market. Each part tries to measure that gap and correct for it."
        ),
        "",
        *brief(),
        "## Contents",
        "",
        "| Part | Question | Data |",
        "|---|---|---|",
        (
            "| [01 Closing auction](reports/01_auction.md) | Can a model predict 60-second moves in the NASDAQ closing "
            "auction well enough to pay the spread? | Optiver *Trading at the Close* (Kaggle): 200 stocks, 481 days |"
        ),
        (
            "| [02 Microstructure](reports/02_microstructure.md) | Does an order-book signal survive realistic "
            "queue-position fills? | LOBSTER sample: five NASDAQ stocks, one day, every order |"
        ),
        (
            "| [03 Arena](reports/03_arena.md) | What does competition between market makers do to spreads and "
            "profits? | Simulated markets: Glosten–Milgrom, Kyle, a market-making tournament, Kuhn poker |"
        ),
        (
            "| [04 Options](reports/04_options.md) | What does delta hedging actually earn, and how often should you "
            "hedge? | Simulation, plus one SPY option chain |"
        ),
        (
            "| [05 Predictions Cup](reports/05_predictions_cup.md) | How should you size bets when only the top three "
            "places get paid? | Simulation of SIG's student contest |"
        ),
        (
            "| [06 Volatility premium](reports/06_vol_premium.md) | How big is the premium in S&P options, when does "
            "selling it blow up, and how much should you sell? | S&P 500, VIX and VIX3M daily closes, 1990 onward |"
        ),
        (
            "| [07 Many days](reports/07_many_days.md) | Does the fill gap from 02 hold across days and stocks? | "
            "NASDAQ TotalView-ITCH 5.0 sample days, 2019–2021: ten days, ten stocks |"
        ),
        (
            "| [08 L2 and latency](reports/08_l2_and_latency.md) | How wrong is a backtest that only sees aggregated "
            "depth, and what does latency cost? | The days of 07 |"
        ),
        "",
        (
            "There is also a trading desk for SIG's Predictions Cup, and a C++ version of the order-book replay "
            "from part 02."
        ),
        "",
    ]
    lines += results_section()
    lines += desk_section()
    lines += [
        "## How I tried to keep it honest",
        "",
        (
            f"- {_n_tests()} tests (`make test`). One of them corrupts every future row of the auction data and checks "
            "that no feature of the past changes; a deliberately leaky feature fails it, so the test does catch leaks."
        ),
        (
            "- Walk-forward splits by day, never random rows. Labels never cross a day, so there is nothing to purge; "
            "`src/markout/auction/cv.py` explains why."
        ),
        (
            "- Every variant I tried is logged with its config, daily PnL and git commit, including the losers, so "
            "the Deflated Sharpe and PBO count all of the searching. (The 32 auction trials were logged from a working "
            "copy whose commit predates this repository's public history, so their SHAs don't resolve on GitHub.)"
        ),
        "- The auction holdout sits behind a guard that refuses a second look"
        + (f" (it has been opened {times} time{'s' if times != 1 else ''})." if times else "."),
        (
            "- The tests check known answers from theory: the Glosten–Milgrom spread, Kyle's λ, the value of Kuhn "
            "poker, put–call parity and the Greeks. Published empirical results are in the table below."
        ),
        (
            "- `make report` regenerates every report and figure from fixed seeds, so the numbers in the text are "
            "never typed by hand."
        ),
        "",
        *replications_section(),
        "## Use it on your own data",
        "",
        (
            "The tools under the reports work on other data too. `markout fills` runs the three fill models on any "
            "LOBSTER day, and `markout sharpe` and `markout pbo` check a backtest against the search that produced it:"
        ),
        "",
        "```bash",
        "markout fills AAPL_2012-06-21_34200000_57600000_message_10.csv AAPL_2012-06-21_34200000_57600000_orderbook_10.csv",
        "markout sharpe pnl.csv --trials 50 --sr-var 0.0009   # bootstrap CI, PSR and the Deflated Sharpe",
        "markout pbo trials.csv --index date                  # probability of backtest overfitting",
        "markout kelly --p 0.62 --price 0.55 --fraction 0.5",
        "```",
        "",
        (
            "[USAGE.md](USAGE.md) covers the command line and the Python API (fill simulation, the Deflated Sharpe, PBO, "
            "the trial registry, the holdout guard, Kelly, Black–Scholes and hedging). Its examples run in the test "
            "suite, so they stay correct. The [API reference](https://tanaymihani.github.io/markout/api/) documents "
            "every module (`make api-docs`)."
        ),
        "",
        "## Running it",
        "",
        "```bash",
        "make install     # editable install into .venv",
        "make lobster     # LOBSTER sample day (5 stocks, 10 levels)",
        "make optiver     # Optiver data (needs a Kaggle login and the competition rules accepted)",
        "make vol         # S&P 500 and VIX history for report 06",
        "make itch        # ten NASDAQ ITCH sample days (about 45 GB) for reports 07 and 08",
        "make cpp         # C++ extension (optional; there is a pure-Python fallback)",
        "make test",
        "make report      # regenerate every report, figure and results file",
        "make demo        # the desk's offline demo",
        "make desk        # the desk on a paper contest, http://127.0.0.1:8765",
        "make public-desk # the hosted desk, one sandbox per visitor, http://127.0.0.1:7860",
        "make docker      # the hosted desk's image (docker run --rm -p 7860:7860 markout-desk)",
        "make demo-video  # re-record the site's demo video (needs the demo extra)",
        "```",
        "",
        (
            "Also: `python -m markout.backtest.report` prints the daily PnL of the chosen auction strategy, "
            "`python -m markout.games.cardgame` is a market-making practice game, and "
            "`python -m markout.decision.journal init|add|score` manages the contest journal."
        ),
        "",
        "## Development",
        "",
        "```bash",
        "make check       # what CI runs: ruff, mypy, the tests and the browser tests",
        "make cov         # coverage report (htmlcov/)",
        "make lock        # re-resolve requirements.lock after changing dependencies",
        "pre-commit install",
        "```",
        "",
        (
            "CI installs the exact versions in `requirements.lock` and runs lint, types, the tests with a coverage "
            "floor, browser tests in Chrome and a smoke test of the desk's container; CodeQL scans the code. "
            "[ARCHITECTURE.md](ARCHITECTURE.md) explains how the pieces fit together and how each is tested, and "
            "[CHANGELOG.md](CHANGELOG.md) lists the releases."
        ),
        "",
        "## Layout",
        "",
        "```",
        "src/markout/auction/     closing-auction data, audit, features, walk-forward models, report 01",
        "src/markout/backtest/    costs, decision rule, sizing, daily PnL",
        "src/markout/decision/    calibration, value of information, Kelly, contest study, journal",
        "src/markout/evaluation/  trial registry, Deflated Sharpe, PBO, bootstrap, holdout guard, Diebold–Mariano",
        "src/markout/lob/         LOBSTER parser, order-flow imbalance, fill simulator, markouts, report 02",
        "src/markout/games/       Glosten–Milgrom, Kyle, market-making arena, Kuhn CFR, card game, report 03",
        "src/markout/options/     Black–Scholes, delta hedging, SPY smile, report 04",
        "src/markout/vol/         S&P 500 / VIX data, variance risk premium, HAR forecast, sizing, hedged straddles, report 06",
        "src/markout/cup/         Predictions Cup desk: sources, matcher, estimates, sizing, paper exchange, local and hosted desk",
        "src/markout/cli.py       the markout command",
        "cpp/                     C++17 queue simulator and pybind11 bindings",
        "reports/                 generated reports, figures (light and dark) and results",
        "docs/                    the site (GitHub Pages), the desk in the browser (docs/desk/), the API reference",
        "deploy/                  the hosted desk's pinned requirements and Hugging Face Space card",
        "tests/                   pytest suite; tests/e2e/ drives a browser, tests/js/ runs the site's labs in Node",
        "```",
        "",
        "## Data",
        "",
        (
            "- **Optiver *Trading at the Close*** (Kaggle). Kaggle's rules don't allow redistributing it, so it is "
            "downloaded by `make optiver` and never committed."
        ),
        (
            "- **LOBSTER sample files** for AAPL, AMZN, GOOG, INTC and MSFT on 2012-06-21. The official download "
            "links stopped working when lobsterdata.com was rebuilt, so the script tries them first and otherwise "
            "uses a pinned, hash-checked mirror. A replay that checks every message against the book is the "
            "authenticity check. Not committed."
        ),
        "- **One SPY option chain**, saved with yfinance on 2026-09-29 after the close.",
        "- **S&P 500, VIX and VIX3M daily closes** from Yahoo Finance (`make vol`). Not committed.",
        (
            "- **NASDAQ TotalView-ITCH 5.0 sample days** from emi.nasdaq.com: ten full days of 2019–2021 "
            "(`make itch`, about 45 GB), converted to LOBSTER format for ten stocks by `markout.lob.itch`, with every "
            "execution, cancel, delete and replace matched to an earlier order" + _itch_audit() + ". Not committed."
        ),
        "",
        "## Limitations",
        "",
        (
            "Each report ends with its own. The main ones: the auction target is measured against a synthetic index, "
            "so the PnL assumes a hedge I can't observe; report 02 is a single day from 2012 (07 adds ten NASDAQ "
            "sample days of 2019–2021, on NASDAQ's book only) and my own orders have no market impact; the arena is a stylized dealer market where everyone shares one "
            "belief; and the contest study has to assume the field size and the number of markets."
        ),
        "",
        "## License",
        "",
        "MIT, see [LICENSE](LICENSE). The datasets keep their own terms and are not redistributed here.",
        "",
    ]
    return "\n".join(ln for ln in lines if ln is not None) + "\n"


def main() -> None:
    (ROOT / "README.md").write_text(build())
    print("wrote README.md")


if __name__ == "__main__":
    main()
