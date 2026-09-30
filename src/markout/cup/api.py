"""The contest adapter interface, plus a rate limiter every adapter goes through.

The real Predictions Cup API is only documented on the platform after registering
(Oct 1 2026). Everything else in the bot talks to `ContestAPI`, so connecting the real
contest means writing one class, `LiveAPI` below, against those docs. Rules to respect
(predictionscup.com/rules): bots are allowed, one account per person, the platform's
API reference, rate limits and position limits.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Protocol, runtime_checkable

from markout.cup.types import Account, Book, Fill, Market, Order, Position


@runtime_checkable
class ContestAPI(Protocol):
    name: str

    def markets(self) -> list[Market]: ...
    def books(self, contract_ids: list[str]) -> dict[str, Book]: ...
    def account(self) -> Account: ...
    def positions(self) -> list[Position]: ...
    def place_order(self, contract_id: str, side: str, price: float, qty: float, client_id: str = "") -> Order: ...
    def cancel_order(self, order_id: str) -> None: ...
    def open_orders(self) -> list[Order]: ...
    def fills(self, since: datetime | None = None) -> list[Fill]: ...


class RateLimiter:
    """Token bucket: at most `rate` calls per second, bursts up to `burst`.

    Defaults are deliberately slow; set them from the platform's documented limits."""

    def __init__(self, rate: float = 1.0, burst: int = 3):
        self.rate, self.burst = rate, burst
        self.tokens = float(burst)
        self.t = time.monotonic()
        self.lock = threading.Lock()
        self.calls = 0

    def acquire(self) -> None:
        with self.lock:
            while True:
                now = time.monotonic()
                self.tokens = min(self.burst, self.tokens + (now - self.t) * self.rate)
                self.t = now
                if self.tokens >= 1:
                    self.tokens -= 1
                    self.calls += 1
                    return
                time.sleep((1 - self.tokens) / self.rate)


class Limited:
    """Wrap any ContestAPI so every call passes the rate limiter."""

    def __init__(self, api: ContestAPI, limiter: RateLimiter | None = None):
        self.api, self.limiter = api, limiter or RateLimiter()
        self.name = api.name

    def __getattr__(self, attr):
        target = getattr(self.api, attr)
        if not callable(target):
            return target

        def call(*a, **kw):
            self.limiter.acquire()
            return target(*a, **kw)

        return call


class LiveAPI:
    """Placeholder for the real Predictions Cup API, written once the docs are available.

    Checklist for the Oct 1 integration (PROGRESS.md task 20):
    1. authentication from data/cup/credentials.json (never logged, never committed);
    2. map the platform's markets/outcomes/prices onto Market/Contract/Book (prices -> [0, 1]);
    3. place/cancel orders, positions, fills, account (cash, rank if exposed);
    4. set RateLimiter from the documented limits and position limits in the config;
    5. contract tests: tests/test_cup_api_contract.py runs against any ContestAPI.
    """

    name = "live"

    def __init__(self, *_, **__):
        raise NotImplementedError("the Predictions Cup API docs are only available after registering (Oct 1); "
                                  "use --paper until the live adapter is written")
