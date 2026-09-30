"""SPY option-chain snapshot: loading, rates and dividends, implied-vol smiles and
put-call parity checks against the bid/ask.

Conventions
-----------
* `kind` is "C" or "P"; strikes and prices are in dollars per share.
* T is in years of calendar time, measured from the quote time to the 16:00 ET
  expiry close on the expiry date (ACT/365).
* r is a continuously compounded rate. The snapshot records a 13-week Treasury
  bill yield (^IRX, a *discount* yield) converted with `tbill_cc_rate`.
* Dividends are discrete (SPY pays quarterly). A dividend with ex-date before
  expiry is "escrowed": the forward is F = (S - PV(D)) e^{rT}, which BSM with a
  continuous yield reproduces exactly when q_T = -ln(1 - PV(D)/S) / T.

Put-call parity (European, Hull ch. 11):  C - P = S e^{-qT} - K e^{-rT}.
With quotes, a conversion (buy stock at S_ask, sell C_bid, buy P_ask) or a
reversal (sell stock at S_bid, buy C_ask, sell P_bid) is riskless only if
    C_bid - P_ask  >  S_ask e^{-qT} - K e^{-rT}      (conversion profits), or
    C_ask - P_bid  <  S_bid e^{-qT} - K e^{-rT}      (reversal profits).
SPY options are American, for which parity is only a band:
    S e^{-qT} - K  <=  C - P  <=  S - K e^{-rT}.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from markout.options import bs
from markout.paths import RAW

ET = ZoneInfo("America/New_York")
OPTIONS_DIR = RAW / "options"
META_KEY = b"markout"
EXPIRY_CLOSE = time(16, 0)          # SPY options stop exercising at the 16:00 ET close
TICK = 0.01                         # SPY option price increment (dollars)
NEAR_MONEY = 0.10                   # quote-quality check: |K/S - 1| <= 10% ...
NEAR_DAYS = (5, 100)                # ... at expiries 5-100 calendar days out
SKEW_Z = 1.0                        # skew slope fitted over |ln(K/F)| <= SKEW_Z ATM standard deviations


# --------------------------------------------------------------------------- rates

def tbill_cc_rate(discount_pct: float, days: int = 91) -> float:
    """Continuously compounded rate from a T-bill *discount* yield in percent.

    Bill price per $1 face: P = 1 - d * days/360, and r_cc = -ln(P) * 365/days.
    (^IRX quotes the 13-week bill, days = 91.)
    """
    d = discount_pct / 100.0
    price = 1.0 - d * days / 360.0
    return -math.log(price) * 365.0 / days


# ----------------------------------------------------------------------- dividends

def third_friday(year: int, month: int) -> date:
    """Third Friday of a month (weekday 4)."""
    first = date(year, month, 1)
    offset = (4 - first.weekday()) % 7
    return first + timedelta(days=offset + 14)


def projected_ex_dates(after: date, until: date) -> list[date]:
    """SPY ex-dividend dates: the third Friday of Mar/Jun/Sep/Dec (moved to the
    Thursday when that Friday is Juneteenth, as in June 2026)."""
    out = []
    for year in range(after.year, until.year + 1):
        for month in (3, 6, 9, 12):
            d = third_friday(year, month)
            if month == 6 and d.day == 19:
                d -= timedelta(days=1)
            if after < d <= until:
                out.append(d)
    return out


def pv_dividends(dividends: Sequence[dict], asof: date, expiry: date, r: float) -> float:
    """PV at `asof` of dividends with asof < ex_date <= expiry (paid on the ex-date)."""
    pv = 0.0
    for dv in dividends:
        ex = date.fromisoformat(dv["ex_date"])
        if asof < ex <= expiry:
            pv += dv["amount"] * math.exp(-r * (ex - asof).days / 365.0)
    return pv


def escrowed_q(S: float, pv_div: float, T: float) -> float:
    """Continuous yield equivalent to escrowing a dividend PV: S e^{-qT} = S - PV(D)."""
    return -math.log(1.0 - pv_div / S) / T if T > 0 else 0.0


# ------------------------------------------------------------------------ snapshot

def two_sided(bid: pd.Series, ask: pd.Series) -> pd.Series:
    """A usable quote has a positive bid and an ask at or above it."""
    return (bid > 0) & (ask > 0) & (ask >= bid)


def screen_stale(df: pd.DataFrame, meta: dict) -> tuple[pd.DataFrame, int]:
    """Restrict `usable` to quotes confirmed by the Cboe cross-check, when one exists.

    yfinance keeps old quotes on illiquid contracts (some deep-ITM bids sit below
    intrinsic value); those show up as quotes that differ from Cboe's closing quote.
    Returns the screened copy and the number of usable quotes removed.
    """
    out = df.copy()
    if not meta.get("cboe_crosscheck", {}).get("available") or meta.get("source_key") != "yfinance":
        return out, 0
    stale = out["usable"] & ~out["matches_cboe"]
    out["usable"] = out["usable"] & out["matches_cboe"]
    return out, int(stale.sum())


def latest_snapshot(directory: Path = OPTIONS_DIR, ticker: str = "SPY") -> Path | None:
    files = sorted(directory.glob(f"{ticker}_*.parquet"))
    return files[-1] if files else None


def load_snapshot(path: Path) -> tuple[pd.DataFrame, dict]:
    """Read a snapshot Parquet file and the JSON metadata stored in its schema."""
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    meta = json.loads(table.schema.metadata[META_KEY].decode())
    df = table.to_pandas()
    df["expiry"] = pd.to_datetime(df["expiry"]).dt.date
    return df, meta


def save_snapshot(df: pd.DataFrame, meta: dict, path: Path) -> Path:
    """Write the chain with `meta` as JSON in the Parquet schema metadata."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    md = dict(table.schema.metadata or {})
    md[META_KEY] = json.dumps(meta, default=str).encode()
    pq.write_table(table.replace_schema_metadata(md), path, compression="zstd")
    return path


def quote_datetime(meta: dict) -> datetime:
    return datetime.fromisoformat(meta["quote_time_et"])


def year_fraction(quote_time: datetime, expiry: date) -> float:
    """Calendar time (ACT/365) from the quote to 16:00 ET on the expiry date."""
    end = datetime.combine(expiry, EXPIRY_CLOSE, tzinfo=ET)
    return (end - quote_time).total_seconds() / (365.0 * 86400.0)


def pick_expiries(expiries: Sequence[date], asof: date,
                  targets: Sequence[int] = (7, 30, 91)) -> list[date]:
    """For each target (calendar days), the listed expiry closest to it."""
    future = sorted(e for e in set(expiries) if e > asof)
    out = []
    for t in targets:
        best = min(future, key=lambda e: (abs((e - asof).days - t), e))
        if best not in out:
            out.append(best)
    return out


@dataclass(frozen=True)
class Carry:
    """Inputs that turn a spot price into a forward for one expiry."""
    expiry: date
    T: float        # years
    r: float        # continuously compounded
    q: float        # continuous yield equivalent to the escrowed dividends
    pv_div: float   # PV of dividends with ex-date before expiry

    def forward(self, S: float) -> float:
        return S * math.exp((self.r - self.q) * self.T)


def carry_for(meta: dict, expiry: date, r: float | None = None) -> Carry:
    """Rate and dividend inputs for one expiry from the snapshot metadata."""
    r = meta["rate"]["r_cc"] if r is None else r
    qt = quote_datetime(meta)
    T = year_fraction(qt, expiry)
    pv = pv_dividends(meta["dividends"]["projected"], qt.date(), expiry, r)
    return Carry(expiry, T, r, escrowed_q(meta["underlying_price"], pv, T), pv)


# ---------------------------------------------------------------------- the smile

def pairs(df: pd.DataFrame, expiry: date) -> pd.DataFrame:
    """One row per strike with both a call and a put quote for `expiry`."""
    sub = df[df["expiry"] == expiry]
    cols = ["strike", "bid", "ask", "usable"]
    c = sub[sub["kind"] == "C"][cols].rename(columns={"bid": "c_bid", "ask": "c_ask", "usable": "c_ok"})
    p = sub[sub["kind"] == "P"][cols].rename(columns={"bid": "p_bid", "ask": "p_ask", "usable": "p_ok"})
    return c.merge(p, on="strike").sort_values("strike").reset_index(drop=True)


def smile(df: pd.DataFrame, meta: dict, expiry: date, max_abs_z: float = 3.5) -> pd.DataFrame:
    """Implied vols from mid, bid and ask for the out-of-the-money side of each strike.

    Out-of-the-money options are used (puts below the forward, calls above) because
    they are the liquid side and carry little early-exercise premium; the American
    ITM quotes would bias a European IV upward. Strikes are kept while
    |ln(K/F)| <= max_abs_z * sigma_atm * sqrt(T), a band a few ATM standard deviations wide.
    """
    S = meta["underlying_price"]
    cy = carry_for(meta, expiry)
    F = cy.forward(S)
    sub = df[(df["expiry"] == expiry) & df["usable"]].copy()
    otm = ((sub["kind"] == "P") & (sub["strike"] < F)) | ((sub["kind"] == "C") & (sub["strike"] >= F))
    sub = sub[otm].sort_values("strike")
    kinds = sub["kind"].map({"C": "call", "P": "put"}).to_numpy()
    K = sub["strike"].to_numpy(float)
    bid, ask = sub["bid"].to_numpy(float), sub["ask"].to_numpy(float)
    mid = 0.5 * (bid + ask)
    iv = {name: bs.implied_vol_array(px, S, K, cy.T, cy.r, cy.q, kinds)
          for name, px in (("iv_mid", mid), ("iv_bid", bid), ("iv_ask", ask))}
    out = pd.DataFrame({"strike": K, "kind": sub["kind"].to_numpy(), "bid": bid, "ask": ask, "mid": mid,
                        "log_moneyness": np.log(K / F), **iv})
    atm = atm_vol(out)
    band = max_abs_z * atm * math.sqrt(cy.T)
    out = out[np.abs(out["log_moneyness"]) <= band].reset_index(drop=True)
    out["z"] = out["log_moneyness"] / (atm * math.sqrt(cy.T))
    out["delta"] = bs.delta(S, out["strike"].to_numpy(), cy.T, cy.r, cy.q, out["iv_mid"].to_numpy(),
                            np.where(out["kind"] == "C", "call", "put"))
    return out


def atm_vol(sm: pd.DataFrame) -> float:
    """Mid IV at the forward, linearly interpolated in log-moneyness."""
    s = sm.dropna(subset=["iv_mid"]).sort_values("log_moneyness")
    return float(np.interp(0.0, s["log_moneyness"], s["iv_mid"]))


def smile_summary(sm: pd.DataFrame, T: float) -> dict:
    """ATM vol, the 25-delta risk reversal, and the skew slope near the money.

    * risk reversal RR25 = IV(25-delta call) - IV(25-delta put), negative for equity skew;
    * slope = d IV / d ln(K/F), fitted over |z| <= SKEW_Z (z = ln(K/F) / (sigma_atm sqrt T)),
      also quoted per 10% of moneyness.
    """
    s = sm.dropna(subset=["iv_mid"])
    atm = atm_vol(s)
    calls = s[s["kind"] == "C"].sort_values("delta")
    puts = s[s["kind"] == "P"].sort_values("delta")
    iv_c25 = float(np.interp(0.25, calls["delta"], calls["iv_mid"])) if len(calls) > 1 else float("nan")
    iv_p25 = float(np.interp(-0.25, puts["delta"], puts["iv_mid"])) if len(puts) > 1 else float("nan")
    near = s[np.abs(s["z"]) <= SKEW_Z]
    slope = float(np.polyfit(near["log_moneyness"], near["iv_mid"], 1)[0]) if len(near) >= 3 else float("nan")
    width = s["iv_ask"] - s["iv_bid"]
    return {"atm_iv": atm, "iv_25d_put": iv_p25, "iv_25d_call": iv_c25, "rr25": iv_c25 - iv_p25,
            "skew_slope": slope, "skew_per_10pct": slope * math.log(1.1),
            "skew_per_sigma": slope * atm * math.sqrt(T),
            "n_strikes": int(len(s)), "median_iv_spread": float(np.nanmedian(width)),
            "iv_spread_atm": float(np.nanmedian(width[np.abs(s["z"]) <= 0.5])),
            "iv_spread_wings": float(np.nanmedian(width[np.abs(s["z"]) >= 2.0])),
            "T_days": T * 365.0}


# --------------------------------------------------------------- put-call parity

def implied_forward(pr: pd.DataFrame, cy: Carry, S: float, n_strikes: int = 6) -> dict:
    """The forward implied by parity at the mids, F_K = K + e^{rT} (C_mid - P_mid),
    as the median over the `n_strikes` two-sided strikes nearest the model forward.

    It measures, in one number, how far the inputs (synchronous S, T-bill r, the
    dividend assumption) sit from the market's own forward. It is not used to fit
    anything: a regression of C - P on K cannot pin down r for short expiries
    (a slope error of 0.001 is 5% of rate at one week), and American early exercise
    tilts C - P away from the money.
    """
    F = cy.forward(S)
    ok = pr[pr["c_ok"] & pr["p_ok"]].copy()
    ok = ok.iloc[np.argsort(np.abs(ok["strike"].to_numpy() - F))[:n_strikes]]
    mids = 0.5 * (ok["c_bid"] + ok["c_ask"]) - 0.5 * (ok["p_bid"] + ok["p_ask"])
    f_k = ok["strike"] + math.exp(cy.r * cy.T) * mids
    f_impl = float(np.median(f_k))
    return {"n": int(len(ok)), "forward_model": F, "forward_implied": f_impl,
            "diff": f_impl - F, "diff_bps": 1e4 * (f_impl - F) / S}


def edge_as_rate(edge: float, K: float, cy: Carry) -> float:
    """An arbitrage edge (dollars today) expressed as an annual rate spread on the
    synthetic loan of K e^{-rT} that a conversion/reversal is: edge / (K e^{-rT}) / T."""
    return edge / (K * math.exp(-cy.r * cy.T)) / cy.T


def parity_check(pr: pd.DataFrame, S: float, S_bid: float, S_ask: float, cy: Carry,
                 tol: float = TICK) -> pd.DataFrame:
    """Parity residuals for each strike, at the mids and after crossing the spread.

    residual_mid = (C_mid - P_mid) - (S e^{-qT} - K e^{-rT}); an apparent arbitrage
    at the mids is |residual_mid| > tol. After crossing: conversion edge
    = (C_bid - P_ask) - (S_ask e^{-qT} - K e^{-rT}) and reversal edge
    = (S_bid e^{-qT} - K e^{-rT}) - (C_ask - P_bid); a violation is an edge > tol.
    The American band test asks whether a conversion/reversal is riskless even
    allowing early exercise: C_bid - P_ask > S_ask - K e^{-rT} or
    C_ask - P_bid < S_bid e^{-qT} - K.
    """
    p = pr[pr["c_ok"] & pr["p_ok"]].copy()
    K = p["strike"].to_numpy(float)
    dq, dr = math.exp(-cy.q * cy.T), math.exp(-cy.r * cy.T)
    c_mid = 0.5 * (p["c_bid"] + p["c_ask"])
    p_mid = 0.5 * (p["p_bid"] + p["p_ask"])
    theo = S * dq - K * dr
    p["residual_mid"] = (c_mid - p_mid) - theo
    p["conversion_edge"] = (p["c_bid"] - p["p_ask"]) - (S_ask * dq - K * dr)
    p["reversal_edge"] = (S_bid * dq - K * dr) - (p["c_ask"] - p["p_bid"])
    p["american_conv_edge"] = (p["c_bid"] - p["p_ask"]) - (S_ask - K * dr)
    p["american_rev_edge"] = (S_bid * dq - K) - (p["c_ask"] - p["p_bid"])
    p["half_band"] = 0.5 * ((p["c_ask"] - p["c_bid"]) + (p["p_ask"] - p["p_bid"]) + (S_ask - S_bid))
    p["mid_violation"] = np.abs(p["residual_mid"]) > tol
    p["spread_violation"] = (p["conversion_edge"] > tol) | (p["reversal_edge"] > tol)
    p["american_violation"] = (p["american_conv_edge"] > tol) | (p["american_rev_edge"] > tol)
    p["moneyness"] = K / S
    return p.reset_index(drop=True)


def parity_summary(chk: pd.DataFrame) -> dict:
    itm_put = chk["moneyness"] > 1.0
    conv = chk["conversion_edge"] > TICK
    return {
        "pairs": int(len(chk)),
        "mid_violations": int(chk["mid_violation"].sum()),
        "spread_violations": int(chk["spread_violation"].sum()),
        "spread_violations_conversion": int((chk["conversion_edge"] > TICK).sum()),
        "spread_violations_reversal": int((chk["reversal_edge"] > TICK).sum()),
        "spread_violations_itm_put_side": int((chk["spread_violation"] & itm_put).sum()),
        "reversal_violations_itm_put": int(((chk["reversal_edge"] > TICK) & itm_put).sum()),
        "conversion_violations_below_spot": int(((chk["conversion_edge"] > TICK) & ~itm_put).sum()),
        "american_violations": int(chk["american_violation"].sum()),
        "median_abs_residual_mid": float(np.median(np.abs(chk["residual_mid"]))) if len(chk) else float("nan"),
        "median_half_band": float(np.median(chk["half_band"])) if len(chk) else float("nan"),
        "max_edge_after_spread": float(np.max(np.maximum(chk["conversion_edge"], chk["reversal_edge"])))
        if len(chk) else float("nan"),
        "max_american_edge": float(np.max(np.maximum(chk["american_conv_edge"], chk["american_rev_edge"])))
        if len(chk) else float("nan"),
        "conversion_moneyness_range": [float(chk.loc[conv, "moneyness"].min()), float(chk.loc[conv, "moneyness"].max())]
        if conv.any() else None,
    }
