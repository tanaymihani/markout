"""All reference sources behind one call: `References.candidates(market, contract)`.

Real-money markets (Polymarket, Kalshi) are fetched in bulk, cached, and indexed by token so
each contest contract is only compared with quotes that share its informative words.
Model sources are used when a question parses: option-implied probabilities for
price-threshold questions, the earnings-beat posterior for "beat EPS" questions.
In paper/demo mode, bundled references replace the network entirely.
"""

from __future__ import annotations

import re
import time
from collections import defaultdict

from markout.cup import match
from markout.cup.match import Candidate
from markout.cup.sources import kalshi, models, polymarket
from markout.cup.sources.base import ExternalQuote
from markout.cup.types import Contract, Market

BEAT = re.compile(r"\b(beat|top|exceed)s?\b.*\b(eps|earnings|estimate|consensus|expectations)\b", re.I)


class References:
    def __init__(self, live: bool = True, static: dict[str, list[ExternalQuote]] | None = None,
                 refresh_seconds: float = 600.0):
        self.live, self.static, self.refresh_seconds = live, static or {}, refresh_seconds
        self._quotes: list[ExternalQuote] = []
        self._index: dict[str, set[int]] = defaultdict(set)
        self._df: dict[str, int] = {}
        self._t = 0.0
        self.errors: list[str] = []

    def _load(self) -> None:
        if not self.live or time.monotonic() - self._t < self.refresh_seconds and self._quotes:
            return
        quotes = []
        for name, fetch in (("polymarket", polymarket.active_markets), ("kalshi", kalshi.open_markets)):
            try:
                quotes += fetch()
            except Exception as e:  # noqa: BLE001 - a source being down must not stop the bot
                self.errors.append(f"{name}: {e}")
        self._quotes, self._index, self._t = quotes, defaultdict(set), time.monotonic()
        for i, q in enumerate(quotes):
            for tok in set(match.norm_tokens(f"{q.question} {q.outcome}")):
                self._index[tok].add(i)
        self._df = {t: len(ix) for t, ix in self._index.items()}

    def _prefilter(self, text: str, max_df: float = 0.02, need: int = 2) -> list[ExternalQuote]:
        toks = [t for t in set(match.norm_tokens(text)) if t in self._index]
        rare = [t for t in toks if self._df[t] <= max(max_df * len(self._quotes), 50)]
        hits: dict[int, int] = defaultdict(int)
        for t in rare:
            for i in self._index[t]:
                hits[i] += 1
        k = min(need, len(rare)) or 1
        return [self._quotes[i] for i, h in hits.items() if h >= k]

    def candidates(self, market: Market, contract: Contract, k: int = 5) -> list[Candidate]:
        if contract.id in self.static:
            return [Candidate(q, 1.0, False, "bundled reference") for q in self.static[contract.id]]
        out: list[Candidate] = []
        if self.live:
            self._load()
            text = f"{market.title} {contract.name}"
            out += match.candidates(market, contract, self._prefilter(text), k=k)
            pq = match.parse_price_question(market)
            if pq is not None:
                q = models.prob_above(pq.ticker, pq.strike, pq.when)
                if q is not None:
                    yes = contract.name.strip().lower() != "no"
                    invert = (not pq.above) == yes  # "below" questions and "No" contracts flip it
                    out.append(Candidate(q, 0.9, invert, f"parsed {pq.ticker} {'>' if pq.above else '<'} "
                                                          f"{pq.strike:g} by {pq.when.date()}"))
            if BEAT.search(market.title):
                tk = next((t for n, t in match.TICKERS.items() if n in market.title.lower()), None)
                q = models.beat_probability(tk) if tk else None
                if q is not None:
                    out.append(Candidate(q, 0.9, contract.name.strip().lower() == "no", f"parsed {tk} EPS beat"))
        return out
