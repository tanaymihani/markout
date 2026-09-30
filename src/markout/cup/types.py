"""Contest-agnostic types for the Predictions Cup bot.

Everything is in probability units: a contract pays 1 if its outcome happens and 0
otherwise, so a price is an implied probability in [0, 1]. The platform adapter converts
to and from whatever the real API uses (cents, SUSQies per contract, ...).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class Contract:
    """One tradable outcome of a market ("Yes", "Republicans", "Over 4.5 runs", ...)."""

    id: str
    market_id: str
    name: str


@dataclass(frozen=True)
class Market:
    id: str
    title: str
    category: str  # elections | earnings | sports | culture | markets | other
    contracts: tuple[Contract, ...]
    closes_at: datetime | None = None  # trading stops
    resolves_at: datetime | None = None  # outcome known
    description: str = ""
    url: str = ""


@dataclass(frozen=True)
class Book:
    """Top of the order book for one contract. Levels are (price, size), best first."""

    contract_id: str
    bids: tuple[tuple[float, float], ...]
    asks: tuple[tuple[float, float], ...]
    ts: datetime = field(default_factory=utcnow)

    @property
    def best_bid(self) -> float | None:
        return self.bids[0][0] if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return self.asks[0][0] if self.asks else None

    @property
    def mid(self) -> float | None:
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2
        return self.best_bid if self.best_ask is None else self.best_ask

    @property
    def spread(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return self.best_ask - self.best_bid

    def depth_at_or_below(self, price: float) -> float:
        """Contracts purchasable at prices <= `price` (walking the asks)."""
        return sum(s for p, s in self.asks if p <= price + 1e-12)


@dataclass
class Order:
    id: str
    contract_id: str
    side: str  # "buy" | "sell"
    price: float
    qty: float
    filled_qty: float = 0.0
    status: str = "open"  # open | filled | partial | cancelled | rejected
    created_at: datetime = field(default_factory=utcnow)
    client_id: str = ""


@dataclass(frozen=True)
class Fill:
    order_id: str
    contract_id: str
    side: str
    price: float
    qty: float
    ts: datetime = field(default_factory=utcnow)


@dataclass
class Position:
    contract_id: str
    qty: float  # contracts held (long only in v1)
    avg_price: float


@dataclass(frozen=True)
class Account:
    cash: float
    equity: float  # cash + positions at mid
    rank: int | None = None  # leaderboard rank (1 = best), if the platform exposes it
    n_players: int | None = None
    start_cash: float | None = None

    @property
    def in_top3(self) -> bool:
        """Holding a top-3 place means being ranked 3rd or better *and* ahead of the start:
        at the open everyone ties, and the pre-registered policy treats that as outside."""
        ahead = self.start_cash is None or self.equity > self.start_cash + 1e-9
        return self.rank is not None and self.rank <= 3 and ahead
