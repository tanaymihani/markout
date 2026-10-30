# Using Markout on your own data

The reports in this repository run on fixed datasets, but the tools underneath are
general: a queue-position fill simulator for any LOBSTER day, statistics that correct a
backtest for the search behind it, Kelly sizing, and Black–Scholes with a hedging
simulator. This page shows each one from the shell and from Python.

Every Python block below runs in the test suite (`tests/test_usage.py`), top to bottom in
one session, so the examples cannot drift away from the code. The full reference for each
module is at [tanaymihani.github.io/markout/api](https://tanaymihani.github.io/markout/api/).

## Install

```bash
git clone https://github.com/tanaymihani/markout && cd markout
python3.11 -m venv .venv && .venv/bin/pip install -e ".[dev,research]"
make cpp          # optional: the C++ fill engine, bit-identical to Python and much faster
.venv/bin/markout status
```

Python 3.11 or newer. Without `make cpp` everything runs on the pure-Python engine.

The toolkit on its own (the evaluation tools, the LOBSTER and ITCH readers, the fill simulator
and this command line) needs only numpy, pandas, polars, pyarrow, scipy and arch. `pip install .`
installs just that, and compiles the C++ engine if a C++17 compiler is around; the `research`
extra adds what the studies need (LightGBM, scikit-learn, statsmodels, matplotlib). Outside a
clone, data and results go to `$MARKOUT_HOME`, or the current directory.
`make install-locked` installs the exact versions CI uses (`requirements.lock`).

## From the shell

**Fills and markouts for a LOBSTER day.** Give it the two files LOBSTER delivers (any
ticker, date or depth). Every 5 seconds it places a buy at the bid and a sell at the ask,
lets each rest for up to 60 seconds, and replays the day under three fill models:

```console
$ markout fills AAPL_2012-06-21_34200000_57600000_message_10.csv AAPL_2012-06-21_34200000_57600000_orderbook_10.csv
$ markout fills --ticker INTC        # a bundled sample day, after `make lobster`
INTC: 624,040 messages, 10 levels; 9,074 orders (both sides every 5 s, 100 shares, resting up to 60 s)
markouts in ticks, quantity-weighted over filled orders; 'random' = an instant fill at the decision time
model     P(fill)    mk 1s   mk 10s   mk 60s  random 10s
touch       0.677   +0.191   +0.089   -0.055      +0.504
fifo        0.478   -0.351   -0.443   -0.669      +0.504
through     0.446   -0.490   -0.559   -0.796      +0.504
```

*touch* counts an order as filled when anything trades at its price, *fifo* when the
displayed queue ahead of it has been used up, and *through* only when the price trades
through it. On INTC, where queues are long, the touch backtest says a resting bid earns
money and the queue says it loses. `--every`, `--wait`, `--size`, `--start`, `--end` and
`--models` change the experiment; `--out rows.csv` keeps one row per order; `--json`
prints machine-readable output.

**Is a backtest's Sharpe real?** Pass a CSV or Parquet file of per-period PnL and say how
many variants you tried, and how much their (per-period) Sharpe ratios varied; together
they set how high the best one could get by luck alone. Here `pnl.csv` is the best of 50
variants that have no skill at all (the Python example below builds the same files):

```console
$ markout sharpe pnl.csv --column pnl --trials 50 --sr-var 0.0009
observations                              750
Sharpe, annualized (x sqrt 252)           1.185
95% CI (stationary bootstrap)             [0.06376, 2.308]
skew / kurtosis                           0.06858 / 3.015
P(true Sharpe > 0), PSR                   0.9795
trials searched                           50
best Sharpe expected by luck, annualized  1.084
Deflated Sharpe (P > luck)                0.5688
```

Taken alone, the winner looks like skill: the interval excludes zero and PSR is 0.98.
Against 50 tries it does not. `--trial-sharpes FILE` takes the per-period Sharpe of every
trial instead of `--trials`/`--sr-var`.

**Probability of backtest overfitting.** One row per period, one column per variant:

```console
$ markout pbo trials.csv --index date --blocks 10
strategies x periods                           50 x 750
CSCV splits                                    252
PBO: P(in-sample winner below the OOS median)  0.5675
P(in-sample winner loses out of sample)        0.5159
OOS vs IS performance slope                    -0.7105
most often selected                            v18 (24%), v23 (17%), v39 (10%), v33 (9%), v47 (6%)
```

**Kelly for a binary contract.** Your probability, the YES price, a fraction:

```console
$ markout kelly --p 0.62 --price 0.55 --fraction 0.5 --bankroll 10000
buy                              YES at 0.55
full Kelly, share of wealth      0.1556
0.5 x Kelly                      0.07778
expected profit per unit staked  0.1273
expected log growth per bet      0.007482
stake out of 10000               777.8
```

## From Python

### Replaying hypothetical orders against an order book

A LOBSTER day is a `Day`: the message file and the book after each message, as NumPy
arrays. Prices are integers in dollars × 10⁴, so one cent is 100. Real files come in with
`lobster.read_csv(message_csv, orderbook_csv)`. To keep this page self-contained, the
examples use a random but valid event stream from the synthetic exchange the tests use:

```python
import numpy as np

from markout.lob import lobster
from markout.lob.synthetic import random_stream

messages, book = random_stream(np.random.default_rng(0), n_events=5000, levels=5)
day = lobster.Day("DEMO", messages, book)  # real data: lobster.read_csv(message_csv, orderbook_csv)
assert len(day) == len(day.book) and day.book.levels == 5
```

Orders are a struct of arrays: the message after which each one is placed, its side,
limit price, size and how long it may rest. `simulate` returns one row per order:

```python
from markout.lob.fills import Orders, simulate

i = np.array([500, 1500, 2500])
side = np.array([1, 1, -1])  # buy, buy, sell
price = np.where(side > 0, book.bid_price[i, 0], book.ask_price[i, 0])  # join the best quote
orders = Orders.make(place_idx=i, side=side, price=price, size=100, max_wait=60.0)

touch = simulate(day.messages, day.book, orders, model="touch")
fifo = simulate(day.messages, day.book, orders, model="fifo")
assert (touch.filled_qty >= fifo.filled_qty).all()  # FIFO is never more generous than touch
print(fifo.queue_ahead, fifo.filled_qty, fifo.end_reason)  # shares ahead at placement, fills, why it ended
```

`engine="cpp"` forces the C++ engine and `engine="python"` the reference one; the default
uses C++ when it is built. The tests check that the two agree bit for bit on randomized
streams.

The experiment behind `markout fills` is three calls: sample decisions on a grid, replay
them under each model, and summarize the markouts (the mid-price move after each fill, in
ticks, positive when it favours you):

```python
from markout.lob.markouts import order_markouts, sample_orders, summarize_markouts

sample = sample_orders(day, every=2.0, max_wait=30.0, size=100, session=(messages.time[0], messages.time[-1]))
per_order = order_markouts(day, sample)  # one row per order and fill model
for row in summarize_markouts(per_order):
    print(row["model"], round(row["p_fill"], 3), round(row["mk_10"], 3))
```

On a random stream the markouts mean nothing; on real data they are report 02.

### Correcting a backtest for the search behind it

Fifty variants of a strategy with no skill at all, and we keep the best one, as any
search would:

```python
import pandas as pd

from markout.evaluation import deflated_sharpe, pbo, sharpe, sharpe_ci

rng = np.random.default_rng(42)
days = pd.bdate_range("2022-01-03", periods=750)
trials = pd.DataFrame(rng.normal(0, 0.01, (750, 50)), index=days, columns=[f"v{i}" for i in range(50)])
best = trials.mean().idxmax()
r = trials[best]

ci = sharpe_ci(r, seed=0)  # stationary bootstrap, block length chosen from the data
d = deflated_sharpe(r, trial_srs=[sharpe(trials[c]) for c in trials])
print(f"Sharpe {sharpe(r) * 252**0.5:.2f} a year, CI [{ci['lo'] * 252**0.5:.2f}, {ci['hi'] * 252**0.5:.2f}]")
print(f"PSR {d['psr_0']:.2f}, deflated {d['dsr']:.2f}")
assert ci["lo"] > 0 and d["psr_0"] > 0.95  # taken alone, it looks like skill
assert d["dsr"] < 0.95  # against 50 tries, it does not
```

Everything in `markout.evaluation` is per period: Sharpe ratios are never annualized
inside, and kurtosis is the plain fourth moment (3 for a normal). Multiply by √252 only
for display.

PBO asks the same question without a model: split the history into blocks, pick the
in-sample winner on every half of them, and see how often it lands below the median out
of sample:

```python
res = pbo(trials, n_blocks=10)
print(round(res["pbo"], 2), round(res["prob_oos_loss"], 2))
assert 0.3 < res["pbo"] < 0.8  # a coin flip, as it should be for pure noise
```

To make the trial count honest, log every variant, including the ones you throw away.
The registry stores each trial's config, metrics, daily PnL and git commit, and
`effective_n` merges near-duplicates (PnL correlation above 0.9) before they reach DSR:

```python
from markout.evaluation import Registry

reg = Registry(path="trials.jsonl", repo_dir=".")
for c in trials:
    reg.log({"variant": c}, {"sharpe": sharpe(trials[c])}, trials[c])
d = deflated_sharpe(r, trial_srs=reg.trial_sharpes(), n_trials=reg.effective_n())
assert reg.n_trials() == 50 and d["dsr"] < 0.95
```

A holdout is out of sample only the first time you look. `HoldoutGuard` opens it once
and refuses after that (unless forced, which is logged):

```python
from markout.evaluation import HoldoutAlreadyUsed, HoldoutGuard

guard = HoldoutGuard(lock_path="holdout.lock", audit_path="holdout_audit.jsonl", repo_dir=".")
guard.run(lambda: {"sharpe": 0.4}, config={"variant": best})
try:
    guard.run(lambda: {"sharpe": 0.9}, config={"variant": best})
except HoldoutAlreadyUsed:
    print("refused; holdout opened", guard.accessed_count(), "time")
```

### Sizing

```python
from markout.decision import kelly

side, f = kelly.best_side(p=0.62, q=0.55)  # belief 62%, YES costs 0.55
g_full = kelly.growth_rate(f, 0.62, 0.55, side)
g_double = kelly.growth_rate(2 * f, 0.62, 0.55, side)
assert side == "YES" and g_full > 0 > g_double  # twice Kelly already loses money in log terms
```

`kelly.fractional`, `kelly.kelly_continuous` (μ/σ²) and `markout.vol.strategy.kelly_exact`
(the growth-optimal size on an empirical PnL series, with ruin) cover the other cases.

### Options

```python
from markout.options import bs, hedging

S, K, T, rate, div, vol = 100.0, 105.0, 30 / 365, 0.04, 0.01, 0.22
c = bs.price(S, K, T, rate, div, vol, kind="call")
g = bs.greeks(S, K, T, rate, div, vol, kind="call")
assert abs(bs.implied_vol(float(c), S, K, T, rate, div, kind="call") - vol) < 1e-6
print({k: round(float(v), 4) for k, v in g.items()})

cfg = hedging.HedgeConfig(sigma_imp=0.20, sigma_real=0.20, n_paths=2000, days=21)
runs = hedging.simulate(cfg, intervals=[1, 6, 78])  # rebalance every 5 min, 30 min, 1 day
spread = [hedging.summarize(runs[h])["std"] for h in (1, 6, 78)]
assert spread[0] < spread[1] < spread[2]  # hedging less often costs less and risks more
```

## The desk

[tanaymihani.github.io/markout/desk](https://tanaymihani.github.io/markout/desk/) runs the
Predictions Cup desk in your browser, on fictional markets and paper money. Locally,
`python -m markout.cup serve --paper` runs it on a paper contest at http://127.0.0.1:8765, and
`python -m markout.cup public` (or `docker run -p 7860:7860 markout-desk`) serves the
server version, which gives every visitor a private sandbox.
[ARCHITECTURE.md](ARCHITECTURE.md) explains how both are put together, and
[deploy/README.md](deploy/README.md) how the hosted one is deployed.
