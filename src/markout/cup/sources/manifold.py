"""Manifold Markets public API (api.manifold.markets/v0): read-only, no key.

Manifold trades play money, so its prices get a small weight in the engine (scaled by the
number of distinct bettors); it mainly adds coverage for culture, tech and politics
questions that the real-money venues don't list."""

from __future__ import annotations

from datetime import datetime, timezone

from markout.cup.sources.base import ExternalQuote
from markout.cup.sources.http import DEFAULT, Http

BASE = "https://api.manifold.markets/v0"


def parse_market(m: dict) -> ExternalQuote | None:
    p = m.get("probability")
    if m.get("outcomeType") != "BINARY" or m.get("isResolved") or p is None or not 0 < p < 1:
        return None
    try:  # some markets "never close" (close times past the year 9999)
        end = datetime.fromtimestamp(m["closeTime"] / 1000, tz=timezone.utc) if m.get("closeTime") else None
    except (ValueError, OverflowError, OSError):
        end = None
    return ExternalQuote("manifold", str(m.get("id")), m.get("question") or "", "Yes", float(p), "market",
                         end=end, url=m.get("url") or "", n=int(m.get("uniqueBettorCount") or 0),
                         note="play money")


def open_binary(limit: int = 1000, http: Http = DEFAULT) -> list[ExternalQuote]:
    rows = http.get_json(f"{BASE}/search-markets", {"term": "", "sort": "liquidity", "filter": "open",
                                                     "contractType": "BINARY", "limit": limit})
    out = []
    for m in rows or []:
        q = parse_market(m)
        if q is not None:
            out.append(q)
    return out
