"""A paper-trading contest: a small simulated exchange that implements ContestAPI.

It exists so the whole bot (sources -> estimates -> proposals -> approval -> orders ->
fills -> resolution -> journal) runs end to end before the real API is available, and
so the demo can run offline. It is deliberately simple and clearly synthetic:

- each contract has a latent "truth" probability and a noisy crowd price; the crowd
  price drifts toward the truth in logit space as time passes, with noise;
- books are the crowd mid +- half the spread, three levels deep;
- marketable limit orders fill immediately against those levels; resting orders fill
  when the crowd price later crosses them (no queue model; see module D for why real
  passive fills are worse than this);
- at `resolves_at` a market resolves: a binary contract by a draw from its truth, and a
  multi-outcome market by one draw across its contracts;
- a simulated field of players gives the account a leaderboard rank, so the
  tournament sizing rule ("am I in the top 3?") can be exercised.
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from markout.cup.types import Account, Book, Contract, Fill, Market, Order, Position, utcnow


def logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


@dataclass
class PaperMarketSpec:
    market: Market
    truth: dict[str, float]  # contract id -> latent probability
    crowd: dict[str, float]  # contract id -> initial contest mid
    spread: float = 0.06
    depth: float = 150.0  # contracts per level
    exclusive: bool = False  # contracts are mutually exclusive outcomes of one event
    outcome: dict[str, int] | None = None  # forced outcomes (demo); else drawn at resolution
    references: list[dict] = field(default_factory=list)  # bundled external quotes (demo)


class PaperContest:
    """Simulated exchange implementing ContestAPI. `clock` returns the current time."""

    name = "paper"

    def __init__(self, specs: list[PaperMarketSpec], start_cash: float = 10_000.0, seed: int = 0,
                 clock: Callable[[], datetime] | None = None, drift_per_day: float = 0.35,
                 noise_per_day: float = 0.25, n_field: int = 1000):
        self.specs = {s.market.id: s for s in specs}
        self.rng = np.random.default_rng(seed)
        self.clock = clock or utcnow
        self.t = self.clock()
        self.start_cash = self.cash = float(start_cash)
        self.crowd = {cid: p for s in specs for cid, p in s.crowd.items()}
        self.contract_market = {c.id: s.market.id for s in specs for c in s.market.contracts}
        self.pos: dict[str, Position] = {}
        self.orders: dict[str, Order] = {}
        self._fills: list[Fill] = []
        self.resolved: dict[str, dict[str, int]] = {}
        self.drift, self.noise = drift_per_day, noise_per_day
        self._ids = itertools.count(1)
        # the field: log-bankrolls random-walk with player-specific volatility per sqrt(day)
        self.field_sigma = self.rng.uniform(0.05, 1.2, n_field)
        self.field_log = np.zeros(n_field)

    # -------------------------------------------------------------- simulation
    def step(self, now: datetime | None = None) -> None:
        """Advance the simulation to `now` (default: the clock)."""
        now = now or self.clock()
        dt_days = max((now - self.t).total_seconds() / 86400, 0.0)
        self.t = now
        if dt_days > 0:
            for cid, p in self.crowd.items():
                m = self.contract_market[cid]
                if m in self.resolved:
                    continue
                truth = self.specs[m].truth[cid]
                x = logit(p) + self.drift * dt_days * (logit(truth) - logit(p)) \
                    + self.noise * math.sqrt(dt_days) * self.rng.standard_normal()
                self.crowd[cid] = sigmoid(x)
            self.field_log += self.field_sigma * math.sqrt(dt_days) * self.rng.standard_normal(len(self.field_log))
            self._cross_resting()
        for mid, spec in self.specs.items():
            if mid not in self.resolved and spec.market.resolves_at and spec.market.resolves_at <= now:
                self._resolve(spec)

    def _resolve(self, spec: PaperMarketSpec) -> None:
        cids = [c.id for c in spec.market.contracts]
        if spec.outcome is not None:
            out = {c: int(spec.outcome.get(c, 0)) for c in cids}
        elif spec.exclusive:
            p = np.array([spec.truth[c] for c in cids], float)
            win = cids[int(self.rng.choice(len(cids), p=p / p.sum()))]
            out = {c: int(c == win) for c in cids}
        else:
            out = {c: int(self.rng.random() < spec.truth[c]) for c in cids}
        self.resolved[spec.market.id] = out
        for c in cids:
            if c in self.pos:
                self.cash += self.pos[c].qty * out[c]
                del self.pos[c]
            for o in self.orders.values():
                if o.contract_id == c and o.status in ("open", "partial"):
                    o.status = "cancelled"
            self.crowd[c] = float(out[c])

    def _cross_resting(self) -> None:
        for o in list(self.orders.values()):
            if o.status not in ("open", "partial"):
                continue
            b = self._book(o.contract_id)
            if o.side == "buy" and b.best_ask is not None and b.best_ask <= o.price:
                self._fill(o, o.qty - o.filled_qty, o.price)
            elif o.side == "sell" and b.best_bid is not None and b.best_bid >= o.price:
                self._fill(o, o.qty - o.filled_qty, o.price)

    # -------------------------------------------------------------- ContestAPI
    def markets(self) -> list[Market]:
        return [s.market for m, s in self.specs.items() if m not in self.resolved]

    def _book(self, cid: str) -> Book:
        spec = self.specs[self.contract_market[cid]]
        mid, h = self.crowd[cid], spec.spread / 2
        bids = tuple((round(max(mid - h - 0.01 * i, 0.01), 4), spec.depth) for i in range(3))
        asks = tuple((round(min(mid + h + 0.01 * i, 0.99), 4), spec.depth) for i in range(3))
        return Book(cid, bids, asks, ts=self.t)

    def books(self, contract_ids: list[str]) -> dict[str, Book]:
        return {c: self._book(c) for c in contract_ids if self.contract_market.get(c) not in self.resolved}

    def equity(self) -> float:
        return self.cash + sum(p.qty * self.crowd[c] for c, p in self.pos.items())

    def account(self) -> Account:
        eq = self.equity()
        field_eq = self.start_cash * np.exp(self.field_log)
        rank = 1 + int((field_eq > eq).sum())
        return Account(cash=self.cash, equity=eq, rank=rank, n_players=len(field_eq) + 1, start_cash=self.start_cash)

    def positions(self) -> list[Position]:
        return list(self.pos.values())

    def place_order(self, contract_id: str, side: str, price: float, qty: float, client_id: str = "") -> Order:
        oid = f"P{next(self._ids)}"
        o = Order(oid, contract_id, side, float(price), float(qty), client_id=client_id, created_at=self.t)
        self.orders[oid] = o
        if self.contract_market.get(contract_id) in self.resolved or qty <= 0 or not 0 < price < 1:
            o.status = "rejected"
            return o
        if side == "buy" and qty * price > self.cash + 1e-9:
            o.status = "rejected"
            return o
        if side == "sell" and qty > self.pos.get(contract_id, Position(contract_id, 0, 0)).qty + 1e-9:
            o.status = "rejected"  # long-only: can only sell what we hold
            return o
        b = self._book(contract_id)
        levels = b.asks if side == "buy" else b.bids
        for lp, size in levels:
            if (side == "buy" and lp > price + 1e-12) or (side == "sell" and lp < price - 1e-12):
                break
            take = min(size, o.qty - o.filled_qty)
            if take <= 0:
                break
            self._fill(o, take, lp)
        return o

    def _fill(self, o: Order, qty: float, price: float) -> None:
        if qty <= 0:
            return
        if o.side == "buy":
            qty = min(qty, self.cash / price)
            self.cash -= qty * price
            p = self.pos.get(o.contract_id)
            if p is None:
                self.pos[o.contract_id] = Position(o.contract_id, qty, price)
            else:
                p.avg_price = (p.avg_price * p.qty + price * qty) / (p.qty + qty)
                p.qty += qty
        else:
            p = self.pos[o.contract_id]
            qty = min(qty, p.qty)
            self.cash += qty * price
            p.qty -= qty
            if p.qty <= 1e-9:
                del self.pos[o.contract_id]
        o.filled_qty += qty
        o.status = "filled" if o.filled_qty >= o.qty - 1e-9 else "partial"
        self._fills.append(Fill(o.id, o.contract_id, o.side, price, qty, ts=self.t))

    def cancel_order(self, order_id: str) -> None:
        o = self.orders.get(order_id)
        if o and o.status in ("open", "partial"):
            o.status = "cancelled"

    def open_orders(self) -> list[Order]:
        return [o for o in self.orders.values() if o.status in ("open", "partial")]

    def fills(self, since: datetime | None = None) -> list[Fill]:
        return [f for f in self._fills if since is None or f.ts > since]

    # -------------------------------------------------------------- helpers
    def references(self, contract_id: str) -> list[dict]:
        """Bundled external quotes for a contract (the offline demo's data source)."""
        spec = self.specs[self.contract_market[contract_id]]
        return [r for r in spec.references if r.get("contract_id") == contract_id]


def _dt(s: str | None) -> datetime | None:
    if not s:
        return None
    d = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def load_scenario(path: Path, start: datetime | None = None) -> list[PaperMarketSpec]:
    """Load a JSON scenario. Times may be absolute ISO strings or `+<hours>h` offsets from `start`."""
    raw = json.loads(Path(path).read_text())
    start = start or utcnow()

    def when(v):
        if isinstance(v, str) and v.startswith("+") and v.endswith("h"):
            return start + timedelta(hours=float(v[1:-1]))
        return _dt(v)

    specs = []
    for m in raw["markets"]:
        contracts = tuple(Contract(c["id"], m["id"], c["name"]) for c in m["contracts"])
        market = Market(m["id"], m["title"], m.get("category", "other"), contracts, when(m.get("closes_at")),
                        when(m.get("resolves_at")), m.get("description", ""), m.get("url", ""))
        specs.append(PaperMarketSpec(market, {c["id"]: c["truth"] for c in m["contracts"]},
                                     {c["id"]: c["crowd"] for c in m["contracts"]}, m.get("spread", 0.06),
                                     m.get("depth", 150.0), m.get("exclusive", False), m.get("outcome"),
                                     m.get("references", [])))
    return specs
