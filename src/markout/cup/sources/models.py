"""Model-based reference probabilities from free market data (yfinance):

- `prob_above`: risk-neutral P(S_T > K) for "will X close above K on date D?" questions,
  from a call spread straddling K on the first listed expiry on or after D:
      P(S_T > K) ~= e^{rT} (C(K1) - C(K2)) / (K2 - K1),   K1 < K < K2.
  The bid/ask band uses the worst and best spread prices. Risk-neutral and physical
  probabilities differ by a risk premium that is small over days to weeks.
- `beat_probability`: P(a company beats consensus EPS) as a Beta-binomial posterior:
  a prior Beta(6, 2) (mean 0.75: most large US companies beat consensus in a typical
  quarter) updated with the company's own record over its last 12 reported quarters.

Both return None when data is missing; the bot then simply has no estimate from them.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timezone

from markout.cup.sources.base import ExternalQuote

PRIOR_A, PRIOR_B = 6.0, 2.0


def digital_from_calls(strikes, bids, asks, strike: float, T: float, r: float = 0.04):
    """(p, lo, hi) of P(S_T > strike) from call quotes; None if strike isn't bracketed."""
    rows = sorted((k, b, a) for k, b, a in zip(strikes, bids, asks) if b > 0 and a >= b)
    below = [x for x in rows if x[0] < strike]  # strict on both sides: a listed strike gets a
    above = [x for x in rows if x[0] > strike]  # spread centred on it, not starting at it
    if not below or not above:
        return None
    (k1, b1, a1), (k2, b2, a2) = below[-1], above[0]
    disc, dk = math.exp(r * T), k2 - k1
    mid = disc * (((b1 + a1) / 2) - ((b2 + a2) / 2)) / dk
    lo, hi = disc * (b1 - a2) / dk, disc * (a1 - b2) / dk
    clip = lambda x: min(max(x, 0.005), 0.995)  # noqa: E731
    return clip(mid), clip(lo), clip(hi)


def prob_above(ticker: str, strike: float, when: datetime, r: float = 0.04) -> ExternalQuote | None:
    try:
        import yfinance as yf

        t = yf.Ticker(ticker)
        target = when.date()
        exp = next((e for e in t.options if date.fromisoformat(e) >= target), None)
        if exp is None:
            return None
        calls = t.option_chain(exp).calls
        T = max((date.fromisoformat(exp) - date.today()).days, 1) / 365.0
        res = digital_from_calls(calls["strike"].tolist(), calls["bid"].tolist(), calls["ask"].tolist(), strike, T, r)
    except Exception:  # noqa: BLE001 - any data failure means "no estimate"
        return None
    if res is None:
        return None
    p, lo, hi = res
    lag = (date.fromisoformat(exp) - when.date()).days
    return ExternalQuote("options", f"{ticker}:{exp}:{strike:g}", f"{ticker} above {strike:g} on {exp}", "Yes", p,
                         "model", lo, hi, end=datetime.fromisoformat(exp).replace(tzinfo=timezone.utc),
                         url=f"https://finance.yahoo.com/quote/{ticker}/options",
                         note=f"call spread on the {exp} expiry" + (f" ({lag} days after the question's date)" if lag else ""))


def beat_posterior(beats: int, n: int, a: float = PRIOR_A, b: float = PRIOR_B) -> float:
    return (a + beats) / (a + b + n)


def beat_probability(ticker: str, quarters: int = 12) -> ExternalQuote | None:
    try:
        import yfinance as yf

        df = yf.Ticker(ticker).get_earnings_dates(limit=quarters + 8)
        past = df.dropna(subset=["EPS Estimate", "Reported EPS"]).head(quarters)
    except Exception:  # noqa: BLE001
        return None
    n = len(past)
    if n == 0:
        return None
    beats = int((past["Reported EPS"] > past["EPS Estimate"]).sum())
    return ExternalQuote("earnings", f"{ticker}:eps-beat", f"{ticker} beats EPS consensus", "Yes",
                         beat_posterior(beats, n), "model", n=n, url=f"https://finance.yahoo.com/quote/{ticker}/analysis",
                         note=f"{beats}/{n} beats in the last {n} reported quarters, prior Beta({PRIOR_A:g}, {PRIOR_B:g})")
