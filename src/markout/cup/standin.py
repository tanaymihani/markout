"""A paper contest built from live Kalshi markets, for dry runs before the real contest.

Each stand-in market copies a live Kalshi question; its latent truth is Kalshi's mid, and
its contest ("student crowd") price is that mid plus logit noise. The stand-in's own Kalshi
quote is excluded from the references, so the bot must find a *different* venue's quote for
the same event (mostly Polymarket), with different wording: a real test of the matcher.
The prices say nothing about edge (the crowd noise is simulated); the matches are real.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np

from markout.cup.paper import PaperMarketSpec, logit, sigmoid
from markout.cup.sources import kalshi, polymarket
from markout.cup.types import Contract, Market

CATEGORIES = {"elections": ("elect", "senate", "house", "governor", "president", "mayor", "vote", "party"),
              "sports": ("win", "game", "match", "score", "points", "nfl", "nba", "mlb", "nhl", "cup", "open"),
              "earnings": ("earnings", "eps", "revenue"), "markets": ("price", "above", "below", "close", "s&p")}


def category(title: str) -> str:
    t = title.lower()
    return next((c for c, words in CATEGORIES.items() if any(w in t for w in words)), "other")


def standin_specs(n: int = 40, seed: int = 0, noise: float = 0.5, horizon_days: int = 36,
                  min_volume: float = 200.0, per_category: int = 10, source: str = "kalshi",
                  ) -> tuple[list[PaperMarketSpec], set[tuple[str, str]]]:
    """Kalshi's liquidity field reads 0 nowadays, so activity is judged by 24h volume."""
    now = datetime.now(timezone.utc)
    pool = kalshi.open_markets(max_pages=6) if source == "kalshi" else \
        [q for q in polymarket.active_markets(pages=4) if q.outcome.lower() == "yes"]
    quotes = [q for q in pool
              if q.bid is not None and q.ask is not None and 0.05 < q.p < 0.95 and q.end
              and now + timedelta(hours=6) < q.end < now + timedelta(days=horizon_days)
              and (q.volume_24h or 0) >= min_volume]
    seen, chosen, per = set(), [], {}
    for q in sorted(quotes, key=lambda x: -(x.volume_24h or 0)):
        event = q.url.rsplit("/", 1)[-1] if source == "kalshi" else q.question
        cat = category(q.question)
        if event in seen or per.get(cat, 0) >= per_category:
            continue  # one market per event, and a mix of categories
        seen.add(event)
        per[cat] = per.get(cat, 0) + 1
        chosen.append(q)
        if len(chosen) >= n:
            break
    rng = np.random.default_rng(seed)
    specs, exclude = [], set()
    for q in chosen:
        mid = f"{source[0]}:{q.id}"
        cs = (Contract(f"{mid}:yes", mid, "Yes"), Contract(f"{mid}:no", mid, "No"))
        m = Market(mid, q.question, category(q.question), cs, q.end - timedelta(minutes=30), q.end, url=q.url)
        crowd = sigmoid(logit(q.p) + noise * rng.standard_normal())
        specs.append(PaperMarketSpec(m, {cs[0].id: q.p, cs[1].id: 1 - q.p}, {cs[0].id: crowd, cs[1].id: 1 - crowd},
                                     spread=0.04, depth=200.0))
        exclude.add((source, q.id))
    return specs, exclude
