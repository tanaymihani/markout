"""The probability engine: pool every source in log-odds space, contest price included.

    logit(p) = sum_i w_i logit(p_i) / sum_i w_i

The contest's own mid is one of the sources. That is the optimizer's-curse correction
in one line: the bot never acts on a reference as if it were the truth, it moves from the
contest price toward the references in proportion to how much it trusts them.

Weights are explicit priors, printed with every proposal, and meant to be revised from
the decision journal as markets resolve:
- the contest mid: 1, less when the contest book is wide (a wide book is a weak signal);
- a real-money market: 1 + log10(1 + liquidity / $10k) + 0.5 log10(1 + 24h volume / $10k), capped at 4;
- options-implied (risk-neutral) probability: 3;
- the earnings-beat posterior: 0.5 + 0.1 per quarter of history (at most 1.7).
The band is the references' own uncertainty (their disagreement and quoted half-spreads,
at least 2 points) around the pooled value: proposals require the whole band to clear the
price, not just the point estimate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from markout.cup.match import Candidate
from markout.cup.types import Book


def logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def source_weight(c: Candidate) -> float:
    q = c.quote
    if q.kind == "market":
        w = 1 + math.log10(1 + (q.liquidity or 0) / 1e4) + 0.5 * math.log10(1 + (q.volume_24h or 0) / 1e4)
        return min(w, 4.0)
    if q.source == "options":
        return 3.0
    if q.source == "earnings":
        return 0.5 + 0.1 * min(q.n or 0, 12)
    return 1.0


def contest_weight(book: Book) -> float:
    s = book.spread
    return 1.0 if s is None else 1.0 / (1.0 + max(s - 0.02, 0.0) / 0.05)


@dataclass
class Estimate:
    p: float
    lo: float
    hi: float
    contest_mid: float | None
    components: list[dict] = field(default_factory=list)

    @property
    def explanation(self) -> str:
        parts = [f"{c['label']} {c['p']:.2f} (w {c['w']:.1f})" for c in self.components]
        return " + ".join(parts) + f" -> {self.p:.2f} [{self.lo:.2f}, {self.hi:.2f}]"


def estimate(book: Book, refs: list[Candidate], min_band: float = 0.02) -> Estimate | None:
    """None when there is nothing but the contest price (no view, no trade)."""
    if not refs or book.mid is None:
        return None
    comps = [{"label": "contest mid", "p": book.mid, "w": contest_weight(book), "url": "", "source": "contest"}]
    for c in refs:
        comps.append({"label": f"{c.quote.source}{' (1-p)' if c.invert else ''}", "p": c.p, "w": source_weight(c),
                      "url": c.quote.url, "source": c.quote.source, "note": c.quote.note, "match": c.why})
    W = sum(x["w"] for x in comps)
    p = sigmoid(sum(x["w"] * logit(x["p"]) for x in comps) / W)
    # Uncertainty comes from the references alone: how much they disagree with each other and
    # how wide their own quotes are. (Measuring it around the contest mid would widen the band
    # exactly when the contest is mispriced, which is when a trade exists.)
    ref = comps[1:]
    wr = sum(c["w"] for c in ref)
    mean_r = sum(c["w"] * c["p"] for c in ref) / wr
    disp = math.sqrt(sum(c["w"] * (c["p"] - mean_r) ** 2 for c in ref) / wr)
    halves = [(c.quote.ask - c.quote.bid) / 2 for c in refs if c.quote.bid is not None and c.quote.ask is not None]
    half = max(min_band, disp, max(halves) if halves else 0.0)
    return Estimate(p, max(p - half, 0.0), min(p + half, 1.0), book.mid, comps)
