# Markout

**Does a short-horizon edge survive costs, fills, and competition?**

A research project that takes trading signals from prediction to PnL and measures what kills them at each step: the spread, selection bias across many backtests, realistic queue-position fills, and competing traders. It spans ML forecasting of the NASDAQ closing auction, order-level microstructure (LOBSTER), game theory (Glosten–Milgrom, Kyle, a market-making tournament, Kuhn-poker CFR), options (delta-hedging PnL, the SPY smile) and decision science (value of information, calibration, Kelly vs tournament sizing), with a C++ core.

**The thread through every module is the winner's curse: whatever you select on looks better than it is.** A *markout*, the price move right after a trade, is how market makers measure it.

| Where it shows up | Form of the curse | Where Markout measures it |
|---|---|---|
| Picking the best of N backtests | selection bias under multiple testing | 01: Deflated Sharpe, PBO |
| Acting on your largest forecasts | the optimizer's curse | 01: calibration and shrinkage |
| The passive order that got filled | adverse selection | 02: queue-position fills, markouts |
| Quoting tighter than every rival | the dealer's winner's curse | 03: Glosten–Milgrom, tournament |
| Betting where you most disagree with the market | selecting on your own errors | 05: journal |

## The product: Markout Desk for the SIG Predictions Cup

A trading desk for SIG's student prediction-market contest (Oct 1 – Nov 4 2026), whose rules allow bots (one account, individual participation, the platform's rate and position limits). It runs on a laptop and uses only free public data. Each step of the research above appears in it:

- **Reference prices** from real-money markets (Polymarket, Kalshi), option-implied probabilities for price questions and earnings-beat histories. Every contest-to-reference mapping must be confirmed by a human (matching is where things go silently wrong).
- **Estimates** pool the references with the contest's own price in log-odds, so the bot never treats a reference as the truth: the optimizer's-curse correction from module 01.
- **Proposals** need the whole uncertainty band to clear the ask, are sized with Kelly as a target *exposure* (the policy pre-registered in report 05, or a steady 0.5x Kelly mode), and are sized to the book: whatever the book can't fill now is not bought, never left resting (module 02's adverse selection).
- **Nothing trades without a click.** Every approved or declined proposal is written to the decision journal, which scores the beliefs against the market once markets resolve (report 05, Part 2).

![The approval desk](docs/demo/desk.png)

[Demo walkthrough](docs/demo/DEMO.md): the whole loop on fictional markets, offline (`make demo`). Run the desk yourself with `make desk` and open http://127.0.0.1:8765.

[Matcher dry run on live markets](docs/demo/DRYRUN.md): stand-in contests built from live Polymarket and Kalshi questions, with every cross-venue suggestion listed (`python -m markout.cup dryrun`). The first version's top suggestions were mostly look-alikes (the other team, another stat line, a different threshold); those cases are now regression tests.

## Results

Every number below is lifted from a generated report; none is typed by hand.

### 01 · Closing-auction alpha: does it survive costs? · [report](reports/01_auction.md)

**Answer.** The best model (lgbm[all, 63 leaves]) cuts out-of-sample MAE by 2.18% versus the per-stock median (IC 0.184; Diebold–Mariano p < 0.001). Turned into trades at 1x cost, the selected policy's research-period PnL is positive and its 95% bootstrap CI excludes zero: annualized Sharpe 4.81 (95% CI 3.34 to 6.28), 22.3 trades a day, 2.27 bps net per trade. The edge is about one spread wide: at 2x cost the same rule's Sharpe is -0.209 and it stops paying. It is also small: the displayed-depth cap binds on 99.5% of trades, the average trade is $1,683, and the research period nets $2,041 in total. Deflated for 18 effective trials (of 32), the Deflated Sharpe Ratio is 0.853 and PBO is 0.059: suggestive, but below the usual 0.95 bar once the search is accounted for. On the holdout (days 421–480, scored once), the same frozen policy nets $1,101 over 60 days (annualized Sharpe 4.74).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="reports/figures/b_voi_dark.png">
  <img alt="Value of information: net edge vs forecast IC" src="reports/figures/b_voi.png" width="720">
</picture>

### 02 · Microstructure: does a signal survive realistic fills? · [report](reports/02_microstructure.md)

**Answer.** **No.** Chosen before 12:45 and scored after, the best post-or-cross rule under FIFO fills earns AAPL -0.192, AMZN -0.051, GOOG -0.158, INTC +0.002 and MSFT +0.003 ticks per decision: reliably negative for GOOG; indistinguishable from zero for AAPL, AMZN and MSFT; positive for INTC only because it trades on 1% of decisions for +0.25 ticks each, which a 0.3-tick taker fee wipes out. Touch fills would have said yes: INTC +0.069 ticks per decision, confidence interval above zero.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="reports/figures/d_markouts_dark.png">
  <img alt="Markouts of filled orders by fill model" src="reports/figures/d_markouts.png" width="720">
</picture>

### 03 · Arena: game theory under simulated competition · [report](reports/03_arena.md)

**Competition prices adverse selection; it does not remove it.** In a Glosten–Milgrom market with elastic noise demand, a lone Bayesian undercutter quotes a time-average spread of 124 ticks and earns 22.7 dollars per episode. One identical rival brings the spread down to 13.8 ticks, 0.8 ticks above the zero-profit spread at the same beliefs. Maker profit falls from 55.9 to 0.4 ticks per trade, and noise-trader welfare rises from 0.112 to 0.429 dollars per period. A naive fixed tight quote loses 1.31 dollars per episode alone and 3.51 in the free-for-all. There it wins 98.2% of the fills in the first 10 periods, while V is still uncertain. Each of those fills carries 18.3 ticks of adverse selection against its 5.44-tick edge: the quote that wins the flow is the one that was too cheap. The round-robin winner is the undercutter.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="reports/figures/e_arena_competition_dark.png">
  <img alt="Spread and maker profit vs number of competing makers" src="reports/figures/e_arena_competition.png" width="720">
</picture>

### 04 · Options: Black–Scholes, delta hedging, the SPY smile · [report](reports/04_options.md)

- Pricing engine: put–call parity holds to 4e-16 of the strike, Greeks match finite differences to 2e-06 relative, and IVs round-trip to 2e-10.
- Short 30-day ATM call sold at σ_imp = 20% and hedged every 5 min with σ_real = 15%: mean PnL +$0.6856 ± 0.0016 vs C(σ_imp) − C(σ_real) = +$0.6841. Short gamma earns when realized < implied.
- Hedging-error std falls like N^-0.49 (theory: 1/√N) while turnover grows like N^0.52. The best frequency for cost + 1 × std moves from 5 min (κ = 0 bp) to 1 day (κ = 10 bp).
- The gamma–theta attribution explains 99.98% of hedged-PnL variance path by path at 5 min hedging (corr 0.9999).
- SPY (yfinance, 2026-09-29 16:15 ET): ATM IV 12.0% at 7 days, 13.4% at 31 days, 14.1% at 93 days. Parity 'fails' for 367 of 370 strike pairs at mids, 64 after crossing the spread, and 6 against the American band.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="reports/figures/f_hedge_frontier_dark.png">
  <img alt="Hedging frequency vs cost frontier" src="reports/figures/f_hedge_frontier.png" width="720">
</picture>

### 05 · SIG Predictions Cup: sizing for a winner-take-most contest · [report](reports/05_predictions_cup.md)

A sizing study written and committed before the contest opens (Oct 1 2026), plus a decision journal that scores every bet afterwards (Brier and log score against the market, calibration, skill vs luck). The winning bankroll has a median of 92× the start (10th–90th percentile 19×–691×), and third place needs a median 35×. 31% of contests were won by a field player going all-in every round, and 2% by one of the sharp players. Your median at 0.75× Kelly is 1.40×, and at 1× Kelly you reach the top 3 in none of the 10,000 simulated contests. In a 1,000-player field over 40 markets, forecasting skill does not move you up the ranking; variance does.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="reports/figures/p_sizing_dark.png">
  <img alt="Contest outcomes by Kelly fraction" src="reports/figures/p_sizing.png" width="720">
</picture>

### C++ core · [section in report 02](reports/02_microstructure.md#c-core)

The order-book replay and FIFO queue simulator is sequential and stateful, the one part worth porting. `cpp/queue_sim.cpp` (C++17, pybind11) is 26–47× faster than the Python reference (median 33×) and bit-identical to it on every run (checked on the five real days and on randomized `hypothesis` event streams).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="reports/figures/g_throughput_dark.png">
  <img alt="C++ vs Python throughput" src="reports/figures/g_throughput.png" width="720">
</picture>

## What keeps it honest

- **352 tests** (`make test`), including a perturbation test that corrupts the future and requires every feature of the past to stay identical, with a negative control that proves the test catches a leak.
- **Walk-forward by day** with a calibration block, never a random split. Purging is unnecessary because labels never cross a day; the reasoning is in `src/markout/auction/cv.py`.
- **Every trial is logged** (`data/registry/trials.jsonl`, with git SHA and config hash), losers included, so the Deflated Sharpe and PBO see how much searching was done.
- **The holdout is guarded**: `HoldoutGuard` refuses a second look and audits forced ones (holdout accessed 1 time).
- **Replications are checked against closed forms**: the DSR paper's worked example, GM's spread at π = ½, Kyle's λ, Kuhn poker's −1/18, Black–Scholes parity and Greeks.
- **Every report is generated** by `make report` from fixed seeds; prose that depends on a result is chosen by code.

## Reproduce

```bash
make install     # editable install into .venv
make lobster     # LOBSTER sample day (5 stocks, 10 levels)
make optiver     # Optiver closing-auction data (needs `kaggle auth login` + accepted rules)
make cpp         # build the C++ extension (optional; pure-Python fallback)
make test        # the test suite
make report      # regenerate every report, figure and results file
make demo        # the desk's offline demo -> docs/demo/
make desk        # the desk on a paper contest: http://127.0.0.1:8765
python -m markout.readme   # regenerate this README
```

Other entry points: `python -m markout.backtest.report` (daily PnL report), `python -m markout.games.cardgame` (market-making practice game), `python -m markout.decision.journal init|add|score` (Predictions Cup journal), `make bench`.

## Layout

```
src/markout/auction/     closing-auction data, audit, causal features, walk-forward models, report 01
src/markout/backtest/    cost model, decision rule, sizing, daily PnL
src/markout/decision/    calibration, value of information, Kelly, contest study, journal
src/markout/evaluation/  trial registry, Deflated Sharpe, PBO, bootstrap, holdout guard, Diebold–Mariano
src/markout/lob/         LOBSTER parser, OFI, fill simulator, markouts, post-vs-cross, report 02
src/markout/games/       Glosten–Milgrom, Kyle, market-making arena, Kuhn CFR, card game, report 03
src/markout/options/     Black–Scholes, delta hedging, SPY smile, report 04
src/markout/cup/         Predictions Cup desk: sources, matcher, engine, sizing, paper exchange, web desk
cpp/                     C++17 queue simulator + pybind11 bindings
reports/                 generated reports, figures (light + dark) and results JSON
tests/                   pytest suite
```

## Data

- **Optiver *Trading at the Close*** (Kaggle): 200 NASDAQ stocks, 481 days, 10-second snapshots of the closing auction. Kaggle's rules forbid redistribution, so it is downloaded, never committed.
- **LOBSTER sample files**: AAPL, AMZN, GOOG, INTC, MSFT on 2012-06-21 at 10 levels. The official download links stopped working when lobsterdata.com was rebuilt, so `scripts/download_lobster.py` tries them first and otherwise uses a pinned, hash-verified mirror; a 100% message/book consistency replay is the authenticity check. Not committed.
- **SPY option chain**: one snapshot taken with yfinance on 2026-09-29 after the close (see report 04).

## Limitations

Each report ends with its own. The big ones: the auction target is relative to a synthetic index, so PnL assumes a hedge; the microstructure study is a single 2012 day with no impact from our own orders; the arena is a stylized dealer market with one shared belief; the contest study's rules, field and markets are assumptions.

