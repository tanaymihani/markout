"""Match a contest contract to reference quotes from other markets.

Matching is the step most likely to go silently wrong (a "Senate" market matched to a
"House" market, "over 4.5" to "over 5.5"), so the matcher only *suggests*: every mapping
must be confirmed by the user before a proposal that uses it can be approved.

Score = token-set Jaccard on normalized words (with aliases such as GOP -> republican)
plus character-trigram cosine, then penalties when the numbers in the contest question
(thresholds, years) are missing from the reference, when the named outcome doesn't
appear, or when resolution dates are far apart. A "No" contract maps to a "Yes" quote
with the probability inverted.

Price-threshold questions ("Will AAPL close above $250 on Oct 31?") are parsed instead,
so the options-implied probability can be used.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone

from markout.cup.sources.base import ExternalQuote
from markout.cup.types import Contract, Market

STOP = set("a an the of in on at to for by will be is are was were who what which when does do did "
           "than more less there any this that with as or and vs v versus from before after during end".split())
ALIASES = {"gop": "republican", "republicans": "republican", "rep": "republican", "dems": "democrat",
           "democrats": "democrat", "democratic": "democrat", "dem": "democrat", "u.s.": "us", "usa": "us",
           "senate": "senate", "house": "house", "pres": "president", "presidential": "president"}
YES_NO = {"yes", "no"}

TICKERS = {"apple": "AAPL", "microsoft": "MSFT", "nvidia": "NVDA", "amazon": "AMZN", "alphabet": "GOOGL",
           "google": "GOOGL", "meta": "META", "facebook": "META", "tesla": "TSLA", "netflix": "NFLX",
           "broadcom": "AVGO", "intel": "INTC", "amd": "AMD", "jpmorgan": "JPM", "goldman": "GS", "nike": "NKE",
           "walmart": "WMT", "costco": "COST", "disney": "DIS", "boeing": "BA", "coca-cola": "KO", "pepsico": "PEP",
           "mcdonald's": "MCD", "starbucks": "SBUX", "oracle": "ORCL", "salesforce": "CRM", "adobe": "ADBE",
           "s&p 500": "SPY", "s&p": "SPY", "nasdaq": "QQQ", "bitcoin": "BTC-USD", "ethereum": "ETH-USD",
           "gold": "GLD"}


def norm_tokens(s: str) -> list[str]:
    s = s.lower().replace("’", "'")
    words = re.findall(r"[a-z0-9$%.']+", s)
    out = []
    for w in words:
        w = w.strip(".'")
        w = ALIASES.get(w, w)
        if not w or w in STOP:
            continue
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.append(w)
    return out


def numbers(s: str) -> set[str]:
    return {n.rstrip(".").replace(",", "") for n in re.findall(r"\d[\d,]*\.?\d*", s)}


def trigram_cosine(a: str, b: str) -> float:
    def grams(x):
        x = f"  {x.lower()} "
        return Counter(x[i:i + 3] for i in range(len(x) - 2))
    ga, gb = grams(a), grams(b)
    dot = sum(ga[g] * gb[g] for g in ga.keys() & gb.keys())
    na, nb = math.sqrt(sum(v * v for v in ga.values())), math.sqrt(sum(v * v for v in gb.values()))
    return dot / (na * nb) if na and nb else 0.0


def text_similarity(a: str, b: str) -> float:
    ta, tb = set(norm_tokens(a)), set(norm_tokens(b))
    jac = len(ta & tb) / len(ta | tb) if ta | tb else 0.0
    return 0.7 * jac + 0.3 * trigram_cosine(a, b)


@dataclass(frozen=True)
class Candidate:
    quote: ExternalQuote
    score: float
    invert: bool  # contest "No" <- reference "Yes"
    why: str

    @property
    def p(self) -> float:
        return 1 - self.quote.p if self.invert else self.quote.p

    @property
    def key(self) -> str:
        return f"{self.quote.source}:{self.quote.id}:{self.quote.outcome}:{int(self.invert)}"


def score(market: Market, contract: Contract, q: ExternalQuote) -> Candidate:
    cname = contract.name.strip().lower()
    binary = cname in YES_NO
    a = market.title if binary else f"{market.title} {contract.name}"
    b = q.question if q.outcome.lower() in YES_NO else f"{q.question} {q.outcome}"
    s = text_similarity(a, b)
    why = [f"text {s:.2f}"]
    need = numbers(a) - {str(y) for y in range(2020, 2031)}
    missing = need - numbers(b)
    if need and missing:
        s -= 0.3
        why.append(f"numbers {sorted(missing)} missing")
    if not binary and q.outcome.lower() not in YES_NO:
        if not set(norm_tokens(contract.name)) & set(norm_tokens(q.outcome)):
            s -= 0.3
            why.append("outcome differs")
    if market.resolves_at and q.end:
        gap = abs((market.resolves_at - q.end).total_seconds()) / 86400
        if gap <= 3:
            s += 0.1
            why.append("dates agree")
        elif gap > 21:
            s -= 0.2
            why.append(f"dates {gap:.0f} days apart")
    invert = binary and cname == "no" and q.outcome.lower() == "yes"
    return Candidate(q, s, invert, ", ".join(why))


def candidates(market: Market, contract: Contract, quotes: list[ExternalQuote], k: int = 5,
               min_score: float = 0.35) -> list[Candidate]:
    scored = [score(market, contract, q) for q in quotes]
    return sorted((c for c in scored if c.score >= min_score), key=lambda c: -c.score)[:k]


PRICE_Q = re.compile(r"(?P<name>\$?[A-Za-z&.\- ]{2,40}?)\s*(?:\((?P<tk>[A-Z.\-]{1,6})\))?\s+"
                     r"(?:stock\s+|shares\s+)?(?:close|end|finish|trade|be)?\s*(?P<dir>above|over|higher than|below|under)"
                     r"\s+\$?(?P<k>[\d,]+(?:\.\d+)?)", re.I)


@dataclass(frozen=True)
class PriceQuestion:
    ticker: str
    strike: float
    above: bool
    when: datetime


def parse_price_question(market: Market) -> PriceQuestion | None:
    """'Will Apple (AAPL) close above $250 on Oct 31?' -> AAPL, 250, above, resolves_at."""
    if market.resolves_at is None:
        return None
    m = PRICE_Q.search(market.title)
    if not m:
        return None
    tk = m.group("tk")
    if not tk:
        name = m.group("name").strip().lstrip("$").lower()
        name = re.sub(r"^(will|does|is)\s+", "", name)
        tk = TICKERS.get(name) or (name.upper() if re.fullmatch(r"[a-z]{1,5}", name) and m.group("name").strip().startswith("$") else None)
    if not tk:
        return None
    above = m.group("dir").lower() in ("above", "over", "higher than")
    return PriceQuestion(tk.upper(), float(m.group("k").replace(",", "")), above,
                         market.resolves_at.astimezone(timezone.utc))
