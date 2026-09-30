"""Match a contest contract to reference quotes from other markets.

Matching is the step most likely to go silently wrong (a "Senate" market matched to a
"House" market, "over 4.5" to "over 5.5"), so the matcher only *suggests*: every mapping
must be confirmed by the user before a proposal that uses it can be approved.

Hard filters first (a dry run on live markets showed soft penalties are not enough: the
top suggestions matched the other team, another stat line or a different threshold):
every named entity of the contest question must appear in the reference, the threshold
numbers must be identical, a measure qualifier present on only one side ("passing",
"1st half", "innings") rejects, and so do more than a few differing words. Survivors are
ranked by token-set Jaccard (aliases such as GOP -> republican) plus character-trigram
cosine, with a bonus or penalty for resolution dates. A "No" contract maps to a "Yes"
quote with the probability inverted.

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


# Words that change *what* is being measured: a mismatch on any of them means a different
# market even when everything else agrees ("1+ touchdowns" vs "1+ passing touchdowns").
QUALIFIERS = set("""passing rushing receiving hit run rbi rbis home homer stolen base strikeout assist rebound point
three pointer 1st 2nd first second third half quarter inning period set map game goal corner card yard touchdown
reception sack save shot ace spread total moneyline margin exact popular electoral turnout primary runoff
increase decrease hike cut raise lower change pause hold up down above below higher highest lowest over under
more fewer less reach dip drop rise fall before after low high""".split())

MONTHS = {m: i for i, names in enumerate([("jan", "january"), ("feb", "february"), ("mar", "march"), ("apr", "april"),
                                         ("may",), ("jun", "june"), ("jul", "july"), ("aug", "august"),
                                         ("sep", "sept", "september"), ("oct", "october"), ("nov", "november"),
                                         ("dec", "december")], start=1) for m in names}
DATE = re.compile(r"\b(" + "|".join(sorted(MONTHS, key=len, reverse=True)) + r")\b\.?(?:\s+(\d{1,2})(?:st|nd|rd|th)?\b)?",
                  re.I)


def years(s: str) -> set[int]:
    return {int(y) for y in re.findall(r"(?<!\d)(?:19|20)\d\d(?!\d)", s)}


def dates(s: str) -> set[tuple[int, int | None]]:
    """(month, day) mentions: 'October 31' -> (10, 31); 'in September' -> (9, None).
    Whole words only ("decrease" is not December); lowercase "may" without a day is the verb."""
    out = set()
    for m in DATE.finditer(s):
        word, day = m.group(1), m.group(2)
        if word.lower() == "may" and not day and not word[0].isupper():
            continue
        out.add((MONTHS[word.lower()], int(day) if day else None))
    return out

NUMBER = re.compile(r"(?<![\w.])\$?\d[\d,]*(?:\.\d+)?")


def thresholds(s: str) -> set[str]:
    """Every number that can change what a market asks (4.5, $250, 24°C, 1+, 25 bps), i.e.
    all numbers except years and the day in a date ("October 31")."""
    days = {m.group(2) for m in DATE.finditer(s) if m.group(2)}
    out = set()
    for m in NUMBER.finditer(s):
        core = m.group(0).lstrip("$").replace(",", "")
        if re.fullmatch(r"(19|20)\d\d", core) or core in days:
            continue
        out.add(core.rstrip("0").rstrip(".") if "." in core else core)
    return out


CALENDAR = set("monday tuesday wednesday thursday friday saturday sunday january february march april may june "
               "july august september october november december jan feb mar apr jun jul aug sep sept oct nov dec "
               "today tomorrow tonight week weekend".split())


def entities(s: str) -> set[str]:
    """Capitalized names and tickers (teams, people, parties, companies, Q3), normalized like
    tokens. Weekdays and months are capitalized but are dates, not entities."""
    ents = set()
    for sentence in re.split(r"[?.!:;]\s+|\s+[-–]\s+", s):
        ents |= _sentence_entities(sentence)
    return ents


OPENERS = STOP | set("over under above below more less new total exact first any how who what why where "
                     "if can could should would has have had".split())


def _sentence_entities(s: str) -> set[str]:
    """A capitalized first word is a name ("Atlanta wins ...") unless it is a common opener
    ("Will ...", "Over 4.5 goals ...", "New album ...")."""
    words = re.findall(r"[A-Za-z][A-Za-z0-9.'&\-]*", s)
    ents = set()
    for i, w in enumerate(words):
        opener = i == 0 and w.lower() in OPENERS
        if (w[0].isupper() and not opener) or (w.isupper() and len(w) >= 2 and not opener):
            t = norm_tokens(w)
            if t and t[0] not in STOP and t[0] not in {"yes", "no"} and t[0] not in CALENDAR:
                ents.add(t[0])
    return ents


def hard_reject(a: str, b: str) -> str | None:
    """Reason the two texts cannot be the same market, or None."""
    ea = entities(a)
    if not ea:
        return "no named entity in the contest question: too ambiguous to match"
    tb = set(norm_tokens(b))
    missing = ea - tb
    if len(missing) > (1 if len(ea) >= 4 else 0):  # long titles carry descriptive capitals ("Midterm")
        return f"entities {sorted(missing)} not in the reference"
    extra = entities(b) - set(norm_tokens(a))
    if len(extra) >= 2:
        return f"the reference also names {sorted(extra)}"
    na, nb = thresholds(a), thresholds(b)
    if na != nb:
        return f"thresholds differ ({sorted(na)} vs {sorted(nb)})"
    da, db = dates(a), dates(b)
    if (da or db) and da != db:
        return f"dates differ ({sorted(da, key=str)} vs {sorted(db, key=str)})"
    ya, yb = years(a), years(b)
    if ya and yb and not ya & yb:
        return f"years differ ({sorted(ya)} vs {sorted(yb)})"
    ta = set(norm_tokens(a)) - numbers(a)
    tb = tb - numbers(b)
    diff = ta ^ tb
    q = diff & QUALIFIERS
    if q:
        return f"different measure ({', '.join(sorted(q))})"
    if len(diff - ea) > 4:
        return "the questions differ in too many words"
    return None


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
    reject = hard_reject(a, b)
    if reject:
        return Candidate(q, -1.0, False, f"rejected: {reject}")
    s = text_similarity(a, b)
    why = [f"text {s:.2f}"]
    if not binary and q.outcome.lower() not in YES_NO:
        if not set(norm_tokens(contract.name)) & set(norm_tokens(q.outcome)):
            s -= 0.3
            why.append("outcome differs")
    if market.resolves_at and q.end:
        lead = (market.resolves_at - q.end).total_seconds() / 86400  # > 0: the reference ends first
        if lead > 10:  # it closes well before the contest resolves: an earlier event
            return Candidate(q, -1.0, False, f"rejected: the reference ends {lead:.0f} days earlier")
        if abs(lead) <= 3:
            s += 0.1
            why.append("dates agree")
        elif lead < -10:  # venues often settle weeks after the event (e.g. after certification)
            s -= 0.1
            why.append(f"the reference settles {-lead:.0f} days later")
    # a binary contract maps onto a Yes/No reference with the probability flipped when the sides differ
    invert = binary and q.outcome.lower() in YES_NO and cname != q.outcome.lower()
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
