"""Polymarket public market data (gamma-api.polymarket.com): read-only, no key.

Prices are real-money consensus probabilities. Only reading public data; Polymarket
trading itself is not used (and is restricted for US persons)."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from markout.cup.sources.base import ExternalQuote
from markout.cup.sources.http import DEFAULT, Http

BASE = "https://gamma-api.polymarket.com"


def _f(x) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _dt(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        d = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def parse_market(m: dict) -> list[ExternalQuote]:
    """One quote per outcome. bid/ask are only known for the first outcome of a binary market."""
    try:
        outcomes = json.loads(m.get("outcomes") or "[]")
        prices = [float(x) for x in json.loads(m.get("outcomePrices") or "[]")]
    except (ValueError, TypeError):
        return []
    if not outcomes or len(outcomes) != len(prices) or m.get("closed"):
        return []
    q = m.get("question") or m.get("title") or ""
    out = []
    for i, (o, p) in enumerate(zip(outcomes, prices)):
        if not 0 < p < 1:
            continue
        bid, ask = (_f(m.get("bestBid")), _f(m.get("bestAsk"))) if i == 0 and len(outcomes) == 2 else (None, None)
        if bid is not None and ask is not None and 0 < bid <= ask < 1:
            p = (bid + ask) / 2  # the mid is a better estimate than the last trade
        out.append(ExternalQuote("polymarket", str(m.get("id")), q, str(o), float(p), "market", bid, ask,
                                 _f(m.get("liquidity")), _f(m.get("volume24hr")), _dt(m.get("endDate")),
                                 f"https://polymarket.com/market/{m.get('slug', '')}"))
    return out


def active_markets(pages: int = 4, per_page: int = 500, http: Http = DEFAULT) -> list[ExternalQuote]:
    out = []
    for page in range(pages):
        rows = http.get_json(f"{BASE}/markets", {"active": "true", "closed": "false", "limit": per_page,
                                                   "offset": page * per_page, "order": "volume24hr",
                                                   "ascending": "false"})
        if not rows:
            break
        for m in rows:
            out += parse_market(m)
    return out


def search(query: str, http: Http = DEFAULT) -> list[ExternalQuote]:
    body = http.get_json(f"{BASE}/public-search", {"q": query, "limit_per_type": 10})
    out = []
    for ev in (body or {}).get("events", []) or []:
        for m in ev.get("markets", []) or []:
            out += parse_market(m)
    return out
