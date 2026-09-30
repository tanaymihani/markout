"""Snapshot helpers, the smile and the parity check, on synthetic Black-Scholes chains."""

import math
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from markout.options import bs, smile

ASOF = datetime(2026, 9, 29, 16, 15, tzinfo=smile.ET)
S, R = 500.0, 0.04


def vol_curve(k_over_s: np.ndarray) -> np.ndarray:
    """A downward-sloping equity-style smile."""
    return 0.18 - 0.25 * np.log(k_over_s) + 0.4 * np.log(k_over_s) ** 2


def synthetic(expiry: date, half_spread: float = 0.02, dividend: float = 0.0) -> tuple[pd.DataFrame, dict]:
    divs = [{"ex_date": "2026-12-18", "amount": dividend}] if dividend else []
    meta = {"underlying_price": S, "quote_time_et": ASOF.isoformat(), "rate": {"r_cc": R},
            "dividends": {"projected": divs}, "cboe_crosscheck": {"available": False}, "source_key": "yfinance"}
    cy = smile.carry_for(meta, expiry)
    K = np.arange(400.0, 601.0, 5.0)
    sig = vol_curve(K / S)
    rows = []
    for kind in ("C", "P"):
        px = bs.price(S, K, cy.T, cy.r, cy.q, sig, "call" if kind == "C" else "put")
        for k, p in zip(K, px):
            rows.append({"contract": f"X{kind}{k:g}", "expiry": expiry, "kind": kind, "strike": k,
                         "bid": max(p - half_spread, 0.0), "ask": p + half_spread, "matches_cboe": True})
    df = pd.DataFrame(rows)
    df["usable"] = smile.two_sided(df["bid"], df["ask"])
    return df, meta


def test_tbill_discount_yield_to_continuous_rate():
    d = 4.065
    price = 1 - d / 100 * 91 / 360
    assert smile.tbill_cc_rate(d) == pytest.approx(-math.log(price) * 365 / 91)
    assert smile.tbill_cc_rate(d) > d / 100      # a discount yield understates the rate


def test_third_friday_and_projected_ex_dates():
    assert smile.third_friday(2026, 9) == date(2026, 9, 18)
    assert smile.third_friday(2026, 12) == date(2026, 12, 18)
    # June 2026's third Friday is Juneteenth, so SPY went ex on the Thursday
    assert smile.projected_ex_dates(date(2026, 1, 1), date(2026, 12, 31)) == [
        date(2026, 3, 20), date(2026, 6, 18), date(2026, 9, 18), date(2026, 12, 18)]


def test_escrowed_dividend_reproduces_the_forward():
    divs = [{"ex_date": "2026-12-18", "amount": 1.9}]
    T = 0.5
    pv = smile.pv_dividends(divs, date(2026, 9, 29), date(2027, 3, 30), R)
    assert pv == pytest.approx(1.9 * math.exp(-R * 80 / 365))
    q = smile.escrowed_q(S, pv, T)
    assert S * math.exp(-q * T) == pytest.approx(S - pv)
    assert smile.pv_dividends(divs, date(2026, 9, 29), date(2026, 12, 17), R) == 0.0   # ex-date after expiry


def test_pick_expiries_takes_the_nearest_listed_dates():
    listed = [date(2026, 9, 30), date(2026, 10, 2), date(2026, 10, 6), date(2026, 10, 30), date(2026, 12, 18),
              date(2026, 12, 31)]
    assert smile.pick_expiries(listed, date(2026, 9, 29)) == [date(2026, 10, 6), date(2026, 10, 30), date(2026, 12, 31)]


def test_smile_recovers_the_input_vols_and_forward():
    e = date(2026, 10, 30)
    df, meta = synthetic(e, half_spread=0.0)
    sm = smile.smile(df, meta, e)
    assert len(sm) > 10
    F = smile.carry_for(meta, e).forward(S)
    expected = vol_curve(sm["strike"].to_numpy() / S)
    assert np.allclose(sm["iv_mid"], expected, atol=1e-6)
    summ = smile.smile_summary(sm, smile.carry_for(meta, e).T)
    assert summ["atm_iv"] == pytest.approx(float(vol_curve(np.array([F / S]))[0]), abs=2e-3)
    assert summ["rr25"] < 0 and summ["skew_slope"] < 0
    fwd = smile.implied_forward(smile.pairs(df, e), smile.carry_for(meta, e), S)
    assert fwd["diff"] == pytest.approx(0.0, abs=1e-8)


@pytest.mark.parametrize("dividend", [0.0, 2.0])
def test_parity_holds_inside_the_spread_for_consistent_prices(dividend):
    e = date(2026, 12, 31)
    df, meta = synthetic(e, half_spread=0.03, dividend=dividend)
    cy = smile.carry_for(meta, e)
    assert (cy.pv_div > 0) == (dividend > 0)
    chk = smile.parity_check(smile.pairs(df, e), S, S - 0.005, S + 0.005, cy)
    assert np.allclose(chk["residual_mid"], 0.0, atol=1e-9)
    summ = smile.parity_summary(chk)
    assert summ["mid_violations"] == 0 and summ["spread_violations"] == 0 and summ["american_violations"] == 0


def test_parity_flags_a_planted_conversion_arbitrage():
    e = date(2026, 10, 30)
    df, meta = synthetic(e, half_spread=0.02)
    cy = smile.carry_for(meta, e)
    i = df.index[(df["kind"] == "C") & (df["strike"] == 500.0)][0]
    df.loc[i, ["bid", "ask"]] += 0.50         # the call bid is now 0.48 above fair value
    chk = smile.parity_check(smile.pairs(df, e), S, S - 0.005, S + 0.005, cy)
    hit = chk[chk["spread_violation"]]
    assert list(hit["strike"]) == [500.0]
    assert hit["conversion_edge"].iloc[0] == pytest.approx(0.50 - 0.02 - 0.02 - 0.005, abs=1e-6)
    # with no dividend the American conversion bound is the European one
    assert bool(hit["american_violation"].iloc[0])
    rate = smile.edge_as_rate(float(hit["conversion_edge"].iloc[0]), 500.0, cy)
    assert rate > 0


def test_snapshot_round_trip_and_stale_screen(tmp_path):
    e = date(2026, 10, 30)
    df, meta = synthetic(e)
    df.loc[0, "matches_cboe"] = False
    meta["cboe_crosscheck"] = {"available": True}
    path = smile.save_snapshot(df, meta, tmp_path / "SPY_20260929.parquet")
    back, meta2 = smile.load_snapshot(path)
    assert meta2 == meta
    assert len(back) == len(df) and back["expiry"].iloc[0] == e
    screened, n = smile.screen_stale(back, meta2)
    assert n == 1 and not screened.loc[0, "usable"]
    assert smile.latest_snapshot(tmp_path) == path
