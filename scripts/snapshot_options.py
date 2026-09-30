"""Save one SPY option-chain snapshot to data/raw/options/SPY_<YYYYMMDD>.parquet.

    .venv/bin/python scripts/snapshot_options.py            # yfinance, Cboe fallback
    .venv/bin/python scripts/snapshot_options.py --source cboe

Source rule: yfinance first. After the close its bid/ask can be zero or stale, so
the script measures the share of two-sided quotes near the money and falls back to
Cboe's delayed-quotes JSON when fewer than half are usable. When yfinance is used,
Cboe's quotes for the same contracts are merged in as a staleness cross-check
(`cboe_bid`, `cboe_ask`, `matches_cboe`); nothing from Cboe replaces a yfinance quote.

Also recorded, in the Parquet schema metadata (see markout.options.smile.load_snapshot):
* the quote time: 16:15 ET (the SPY option close) when run after the close;
* the underlying price synchronous with those quotes: the close of the 16:14 ET
  one-minute bar (yfinance pre/post-market bars), plus the 16:00 close for reference;
* r: the 13-week T-bill discount yield (^IRX), converted to continuous compounding;
* dividends: the last paid SPY dividend and projected ex-dates (third Friday of
  Mar/Jun/Sep/Dec), each assumed equal to the last amount.
Every step that fails is written into the metadata instead of being papered over.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from markout.options import smile

TICKER = "SPY"
CBOE_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/{ticker}.json"
OPTION_CLOSE = time(16, 15)   # SPY (ETF) options trade until 16:15 ET
USABLE_MIN_SHARE = 0.5        # below this share of two-sided near-money quotes, use Cboe
CHAIN_COLUMNS = ["contract", "expiry", "kind", "strike", "bid", "ask", "last", "volume",
                 "open_interest", "last_trade_utc", "vendor_iv"]


def fetch_yfinance_chain(tk) -> tuple[pd.DataFrame, dict]:
    frames = []
    underlying = {}
    for e in tk.options:
        ch = tk.option_chain(e)
        underlying = getattr(ch, "underlying", {}) or underlying
        for kind, d in (("C", ch.calls), ("P", ch.puts)):
            frames.append(pd.DataFrame({
                "contract": d["contractSymbol"], "expiry": pd.to_datetime(e).date(), "kind": kind,
                "strike": d["strike"].astype(float), "bid": d["bid"].astype(float), "ask": d["ask"].astype(float),
                "last": d["lastPrice"].astype(float), "volume": d["volume"].astype(float),
                "open_interest": d["openInterest"].astype(float),
                "last_trade_utc": pd.to_datetime(d["lastTradeDate"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "vendor_iv": d["impliedVolatility"].astype(float)}))
    return pd.concat(frames, ignore_index=True), underlying


def fetch_cboe_chain(ticker: str) -> tuple[pd.DataFrame, dict]:
    req = urllib.request.Request(CBOE_URL.format(ticker=ticker), headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=90) as resp:
        payload = json.load(resp)
    rows = payload["data"]["options"]
    pat = re.compile(rf"^{ticker}(\d{{6}})([CP])(\d{{8}})$")
    recs = []
    for o in rows:
        m = pat.match(o["option"])
        if not m:
            continue   # adjusted or non-standard roots
        recs.append({"contract": o["option"], "expiry": datetime.strptime(m.group(1), "%y%m%d").date(),
                     "kind": m.group(2), "strike": int(m.group(3)) / 1000.0,
                     "bid": float(o["bid"]), "ask": float(o["ask"]), "last": float(o["last_trade_price"]),
                     "volume": float(o["volume"]), "open_interest": float(o["open_interest"]),
                     "last_trade_utc": o["last_trade_time"] or "", "vendor_iv": float(o["iv"])})
    und = {k: v for k, v in payload["data"].items() if k != "options"}
    und["timestamp_utc"] = payload.get("timestamp")
    return pd.DataFrame(recs), und


def usability(df: pd.DataFrame, spot: float, asof: date) -> dict:
    ok = smile.two_sided(df["bid"], df["ask"])
    days = np.array([(e - asof).days for e in df["expiry"]])
    lo, hi = smile.NEAR_DAYS
    near = (np.abs(df["strike"] / spot - 1) <= smile.NEAR_MONEY) & (days >= lo) & (days <= hi)
    return {"rows": int(len(df)), "two_sided_share": float(ok.mean()),
            "near_money_rows": int(near.sum()), "near_money_two_sided_share": float(ok[near].mean())}


def underlying_prices(tk, now_et: datetime) -> dict:
    """The SPY price synchronous with the option quotes, and the 16:00 close."""
    bars = tk.history(period="1d", interval="1m", prepost=True)
    bars.index = bars.index.tz_convert(smile.ET)
    session = bars.index[-1].date()
    close_1600 = float(tk.history(period="5d", interval="1d")["Close"].iloc[-1])
    after_close = now_et >= datetime.combine(session, OPTION_CLOSE, tzinfo=smile.ET)
    if after_close:
        cut = datetime.combine(session, OPTION_CLOSE, tzinfo=smile.ET) - timedelta(minutes=1)
        bar = bars[bars.index <= cut].iloc[-1]
        return {"session_date": session.isoformat(), "market_state": "after_option_close",
                "quote_time_et": datetime.combine(session, OPTION_CLOSE, tzinfo=smile.ET).isoformat(),
                "underlying_price": float(bar["Close"]),
                "underlying_price_source": f"close of the {bar.name.strftime('%H:%M')} ET one-minute bar "
                                           "(yfinance pre/post-market bars), the last minute of option trading",
                "underlying_close_1600": close_1600,
                "underlying_last": float(bars["Close"].iloc[-1]),
                "underlying_last_time_et": bars.index[-1].isoformat()}
    last = bars.iloc[-1]
    return {"session_date": session.isoformat(), "market_state": "intraday",
            "quote_time_et": now_et.replace(microsecond=0).isoformat(),
            "underlying_price": float(last["Close"]),
            "underlying_price_source": f"close of the latest one-minute bar ({last.name.strftime('%H:%M')} ET); "
                                       "option quotes may be delayed relative to it",
            "underlying_close_1600": close_1600, "underlying_last": float(last["Close"]),
            "underlying_last_time_et": last.name.isoformat()}


def rate_info(yf) -> dict:
    try:
        irx = yf.Ticker("^IRX").history(period="10d")["Close"].dropna()
        pct = float(irx.iloc[-1])
        return {"r_cc": smile.tbill_cc_rate(pct), "irx_discount_pct": pct, "irx_date": irx.index[-1].date().isoformat(),
                "source": "^IRX 13-week T-bill discount yield (yfinance), converted: "
                          "P = 1 - d*91/360, r = -ln(P)*365/91"}
    except Exception as exc:  # noqa: BLE001 - recorded, and the analysis backs r out of parity instead
        return {"r_cc": None, "source": f"^IRX unavailable ({exc!r}); back r out of put-call parity"}


def dividend_info(tk, asof: date) -> dict:
    div = tk.dividends
    last_date, last_amt = div.index[-1].date(), float(div.iloc[-1])
    recent = [{"ex_date": d.date().isoformat(), "amount": float(a)} for d, a in div.iloc[-8:].items()]
    projected = [{"ex_date": d.isoformat(), "amount": last_amt}
                 for d in smile.projected_ex_dates(max(asof, last_date), asof + timedelta(days=3 * 366))]
    return {"last_ex_date": last_date.isoformat(), "last_amount": last_amt, "recent": recent, "projected": projected,
            "rule": "ex-dates on the third Friday of Mar/Jun/Sep/Dec; each future amount = last paid amount"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["auto", "yfinance", "cboe"], default="auto")
    ap.add_argument("--out-dir", default=str(smile.OPTIONS_DIR))
    args = ap.parse_args(argv)

    import yfinance as yf

    now_utc = datetime.now(timezone.utc).replace(microsecond=0)
    now_et = now_utc.astimezone(smile.ET)
    tk = yf.Ticker(TICKER)
    und = underlying_prices(tk, now_et)
    session = date.fromisoformat(und["session_date"])
    spot = und["underlying_price"]
    meta = {"ticker": TICKER, "snapshot_utc": now_utc.isoformat(), **und,
            "rate": rate_info(yf), "dividends": dividend_info(tk, session)}

    notes, cboe, cboe_und = [], None, None
    yf_chain, yf_und = (None, {})
    if args.source in ("auto", "yfinance"):
        try:
            yf_chain, yf_und = fetch_yfinance_chain(tk)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"yfinance chain failed: {exc!r}")
    if args.source in ("auto", "cboe") or yf_chain is not None:
        try:
            cboe, cboe_und = fetch_cboe_chain(TICKER)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"Cboe delayed quotes failed: {exc!r}")

    use = None
    if yf_chain is not None:
        u = usability(yf_chain[yf_chain["expiry"] > session], spot, session)
        meta["usability_yfinance"] = u
        if args.source == "yfinance" or u["near_money_two_sided_share"] >= USABLE_MIN_SHARE:
            use = "yfinance"
        else:
            notes.append(f"yfinance: only {u['near_money_two_sided_share']:.0%} of near-money quotes two-sided")
    if use is None and cboe is not None:
        use = "cboe"
    if use is None:
        print("No usable option quotes from yfinance or Cboe:", *notes, sep="\n  ", file=sys.stderr)
        return 1

    chain = yf_chain if use == "yfinance" else cboe
    # drop contracts that have expired (a same-day expiry is gone once the option close has passed)
    alive = chain["expiry"] > session if und["market_state"] == "after_option_close" else chain["expiry"] >= session
    chain = chain[alive].reset_index(drop=True)
    meta["usability"] = usability(chain, spot, session)
    if use == "yfinance" and cboe is not None:
        cb = cboe[["contract", "bid", "ask"]].rename(columns={"bid": "cboe_bid", "ask": "cboe_ask"})
        chain = chain.merge(cb, on="contract", how="left")
        both = chain["cboe_bid"].notna()
        chain["matches_cboe"] = both & np.isclose(chain["bid"], chain["cboe_bid"]) & np.isclose(chain["ask"], chain["cboe_ask"])
        meta["cboe_crosscheck"] = {
            "available": True, "timestamp_utc": cboe_und.get("timestamp_utc"),
            "contracts_in_cboe": int(len(cboe[cboe["expiry"] > session])), "matched": int(both.sum()),
            "identical_bid_ask_share": float(chain.loc[both, "matches_cboe"].mean()),
            "underlying_current_price": cboe_und.get("current_price"),
            "underlying_bid": cboe_und.get("bid"), "underlying_ask": cboe_und.get("ask"),
            "underlying_close": cboe_und.get("close")}
    else:
        chain["cboe_bid"], chain["cboe_ask"], chain["matches_cboe"] = np.nan, np.nan, False
        meta["cboe_crosscheck"] = {"available": False}
    und_bid, und_ask = (yf_und.get("bid"), yf_und.get("ask")) if use == "yfinance" else \
        (cboe_und.get("bid"), cboe_und.get("ask"))
    meta.update({"source": {"yfinance": "yfinance (Yahoo Finance option chains)",
                            "cboe": "Cboe delayed quotes JSON (" + CBOE_URL.format(ticker=TICKER) + ")"}[use],
                 "source_key": use, "notes": notes,
                 "underlying_quote_at_snapshot": {"bid": und_bid, "ask": und_ask}})

    path = Path(args.out_dir) / f"{TICKER}_{session.strftime('%Y%m%d')}.parquet"
    chain["usable"] = smile.two_sided(chain["bid"], chain["ask"])
    smile.save_snapshot(chain[CHAIN_COLUMNS + ["cboe_bid", "cboe_ask", "matches_cboe", "usable"]], meta, path)
    print(f"saved {path} ({path.stat().st_size / 1e6:.2f} MB): {len(chain)} contracts, "
          f"{chain['expiry'].nunique()} expiries, source={use}")
    print(f"quote time {meta['quote_time_et']}, underlying {spot:.2f} ({und['underlying_price_source']})")
    print(f"r = {meta['rate']['r_cc']} ({meta['rate']['source']})")
    if meta["cboe_crosscheck"]["available"]:
        print(f"Cboe cross-check: {meta['cboe_crosscheck']['identical_bid_ask_share']:.1%} of matched quotes identical")
    for n in notes:
        print("note:", n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
