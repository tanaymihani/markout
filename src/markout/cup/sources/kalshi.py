"""Kalshi public market data (api.elections.kalshi.com/trade-api/v2): read-only, no key.
Multivariate combo markets are excluded (mve_filter=exclude)."""

from __future__ import annotations

from datetime import datetime, timezone

from markout.cup.sources.base import ExternalQuote
from markout.cup.sources.http import DEFAULT, Http

BASE = "https://api.elections.kalshi.com/trade-api/v2"


def _f(x) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def parse_market(m: dict) -> ExternalQuote | None:
    bid, ask = _f(m.get("yes_bid_dollars")), _f(m.get("yes_ask_dollars"))
    last = _f(m.get("last_price_dollars"))
    if bid is not None and ask is not None and 0 < bid <= ask < 1:
        p = (bid + ask) / 2
    elif last is not None and 0 < last < 1:
        p = last
    else:
        return None
    end = m.get("close_time") or m.get("expected_expiration_time")
    end_dt = datetime.fromisoformat(end.replace("Z", "+00:00")) if end else None
    q = " ".join(x for x in [m.get("title") or "", m.get("yes_sub_title") or ""] if x)
    return ExternalQuote("kalshi", m.get("ticker", ""), q, "Yes", p, "market", bid, ask,
                         _f(m.get("liquidity_dollars")), _f(m.get("volume_24h_fp")),
                         end_dt.astimezone(timezone.utc) if end_dt else None,
                         f"https://kalshi.com/markets/{(m.get('event_ticker') or '').lower()}")


def open_markets(max_pages: int = 5, http: Http = DEFAULT) -> list[ExternalQuote]:
    out, cursor = [], None
    for _ in range(max_pages):
        params = {"status": "open", "limit": 1000, "mve_filter": "exclude"}
        if cursor:
            params["cursor"] = cursor
        body = http.get_json(f"{BASE}/markets", params)
        for m in body.get("markets", []):
            q = parse_market(m)
            if q is not None:
                out.append(q)
        cursor = body.get("cursor")
        if not cursor:
            break
    return out
