"""Daily PnL report of the selected policy, the backtest's daily feedback loop.

    python -m markout.backtest.report              # research period, last 20 days
    python -m markout.backtest.report --holdout    # the holdout days
    python -m markout.backtest.report --all

Reads the CSVs that `python -m markout.auction.report` writes to reports/results/.
"""

from __future__ import annotations

import argparse

import polars as pl

from markout.paths import RESULTS


def render(daily: pl.DataFrame) -> str:
    cols = [c for c in ["date_id", "n_trades", "notional", "gross_usd", "cost_usd", "pnl_usd", "cum_pnl_usd",
                        "drawdown_usd", "hit_rate"] if c in daily.columns]
    if "cum_pnl_usd" not in daily.columns:
        daily = daily.with_columns(pl.col("pnl_usd").cum_sum().alias("cum_pnl_usd"))
        cols.append("cum_pnl_usd")
    fmt = {"notional": "{:>12,.0f}", "gross_usd": "{:>10,.0f}", "cost_usd": "{:>10,.0f}", "pnl_usd": "{:>10,.0f}",
           "cum_pnl_usd": "{:>11,.0f}", "drawdown_usd": "{:>11,.0f}", "hit_rate": "{:>6.0%}"}
    head = " ".join(f"{c:>12}" for c in cols)
    lines = [head, "-" * len(head)]
    for r in daily.select(cols).iter_rows(named=True):
        lines.append(" ".join(f"{(fmt.get(c, '{:>12}').format(v) if v is not None else 'n/a'):>12}"
                              for c, v in r.items()))
    tot = daily["pnl_usd"].sum()
    lines += ["-" * len(head), f"days {daily.height}   total net PnL ${tot:,.0f}   "
              f"days with trades {int((daily['n_trades'] > 0).sum())}"]
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--last", type=int, default=20)
    ap.add_argument("--prefix", default="")
    a = ap.parse_args(argv)
    name = f"{a.prefix}auction_daily_{'holdout' if a.holdout else 'research'}.csv"
    path = RESULTS / name
    if not path.exists():
        raise SystemExit(f"{path} not found: run `python -m markout.auction.report` first")
    daily = pl.read_csv(path)
    print(render(daily if a.all else daily.tail(a.last)))


if __name__ == "__main__":
    main()
