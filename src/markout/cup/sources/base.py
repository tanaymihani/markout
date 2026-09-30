"""The common shape of a reference probability, whatever its source."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from markout.cup.types import utcnow


@dataclass(frozen=True)
class ExternalQuote:
    source: str  # polymarket | kalshi | options | earnings | static
    id: str  # the source's market id / ticker
    question: str  # text used for matching
    outcome: str  # the outcome this probability refers to ("Yes", "Republicans", ...)
    p: float  # probability (mid for markets, estimate for models)
    kind: str = "market"  # "market" = a real-money price; "model" = our estimate from other data
    bid: float | None = None
    ask: float | None = None
    liquidity: float | None = None  # USD, real-money markets
    volume_24h: float | None = None  # USD
    end: datetime | None = None  # resolution / close time
    url: str = ""
    n: int | None = None  # sample size behind a model estimate
    note: str = ""
    ts: datetime = field(default_factory=utcnow)
