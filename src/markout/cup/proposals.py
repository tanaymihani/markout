"""Turn estimates into trade proposals: edge, Kelly fraction, stake, and a written reason.

Long-only v1: every proposal buys a contract at (or below) its current ask. A proposal
needs edge = p - ask >= `min_edge` AND the estimate's whole band above the ask.

Sizing modes:
- "prize" = the policy pre-registered in reports/05_predictions_cup.md before the contest:
  unless the account holds a top-3 place, stake everything available on the single largest
  Kelly fraction; while it holds one, stake 0.5x Kelly on every trade. The other candidates
  are still shown, with stake 0, so their forecasts get logged and scored.
  This policy maximizes expected prize money in the study, and busts in most simulated contests.
- "steady" = 0.5x Kelly on every trade, capped at `max_frac_per_market` of equity.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone

from markout.cup.engine import Estimate
from markout.cup.types import Account, Book, Market
from markout.decision.kelly import kelly_yes

CONTEST_END = datetime(2026, 11, 4, 16, 0, tzinfo=timezone.utc)  # 12:00 ET


@dataclass
class Config:
    mode: str = "prize"  # "prize" (pre-registered) | "steady"
    min_edge: float = 0.03
    kelly_mult: float = 0.5
    max_frac_per_market: float = 0.25  # steady mode only
    max_stake: float | None = None  # the platform's position limit in currency, once known
    min_minutes_to_close: float = 10.0
    proposal_ttl_minutes: float = 30.0
    cooldown_hours: float = 4.0  # after a decision, re-propose a contract only if p or the ask moved
    requote_move: float = 0.03
    contest_end: datetime = CONTEST_END

    def to_dict(self) -> dict:
        d = asdict(self)
        d["contest_end"] = self.contest_end.isoformat()
        return d


@dataclass
class Proposal:
    id: str
    created: str
    market_id: str
    market_title: str
    category: str
    contract_id: str
    contract_name: str
    p: float
    lo: float
    hi: float
    bid: float | None
    ask: float
    edge: float
    kelly: float
    mode: str
    stake: float
    qty: float
    limit_price: float
    mapping_confirmed: bool
    mapping_keys: list[str]
    components: list[dict]
    rationale: str
    flags: list[str] = field(default_factory=list)
    status: str = "pending"  # pending | executed | partial | logged | skipped | rejected | expired
    decided: str | None = None
    decided_stake: float | None = None
    note: str = ""
    order_id: str | None = None
    fill_price: float | None = None
    fill_qty: float | None = None
    expires: str | None = None
    policy_stake: float = 0.0  # what the sizing rule wants; `stake` is what the book can fill with edge

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Proposal":
        return cls(**d)


@dataclass
class Candidate:
    market: Market
    contract_id: str
    contract_name: str
    est: Estimate
    book: Book
    kelly: float
    edge: float
    confirmed: bool
    keys: list[str]
    flags: list[str]


def screen(markets: list[Market], books: dict[str, Book], estimates: dict[str, tuple[Estimate, bool, list[str]]],
           cfg: Config, now: datetime) -> list[Candidate]:
    out = []
    for m in markets:
        if m.closes_at and m.closes_at < now + timedelta(minutes=cfg.min_minutes_to_close):
            continue
        for c in m.contracts:
            if c.id not in estimates or c.id not in books:
                continue
            est, confirmed, keys = estimates[c.id]
            b = books[c.id]
            a = b.best_ask
            if a is None or not 0 < a < 1:
                continue
            edge = est.p - a
            if edge < cfg.min_edge or est.lo <= a:
                continue
            flags = [] if confirmed else ["mapping not confirmed"]
            if m.resolves_at and m.resolves_at > cfg.contest_end:
                flags.append("resolves after the contest ends")
            out.append(Candidate(m, c.id, c.name, est, b, float(kelly_yes(est.p, a)), edge, confirmed, keys, flags))
    return sorted(out, key=lambda x: -x.kelly)


def size(cands: list[Candidate], account: Account, cfg: Config,
         held: dict[str, float] | None = None) -> list[tuple[Candidate, float, str]]:
    """(candidate, stake, why) for every candidate; stake 0 = log the forecast only.

    Kelly is a target *exposure*, not a bet per refresh: what is already held in a contract
    (valued at the ask) is subtracted, so re-proposing a contract never piles on."""
    held = held or {}
    cap = cfg.max_stake if cfg.max_stake is not None else math.inf
    cash, eq = max(account.cash, 0.0), max(account.equity, 0.0)
    out = []
    if cfg.mode == "prize" and not account.in_top3:
        for i, c in enumerate(cands):
            if i == 0:
                out.append((c, min(cash, cap), "prize mode, outside the top 3: all-in on the largest Kelly fraction"))
            else:
                out.append((c, 0.0, "prize mode bets only the largest edge; logged as a forecast"))
        return out
    why = ("prize mode, holding a top-3 place: 0.5x Kelly" if cfg.mode == "prize"
           else f"steady mode: {cfg.kelly_mult:g}x Kelly, at most {cfg.max_frac_per_market:.0%} of equity")
    for c in cands:
        target = cfg.kelly_mult * c.kelly * eq
        if cfg.mode == "steady":
            target = min(target, cfg.max_frac_per_market * eq)
        stake = max(target - held.get(c.contract_id, 0.0) * c.book.best_ask, 0.0)
        stake = min(stake, cash, max(cap - held.get(c.contract_id, 0.0) * c.book.best_ask, 0.0))
        out.append((c, stake, why if stake > 0 else "already at the target exposure; logged as a forecast"))
        cash -= stake
    return out


def fill_plan(book: Book, max_price: float, budget: float) -> tuple[int, float, float]:
    """Walk the asks while each level still has edge (price <= max_price) and money remains.
    Returns (contracts, worst price used = the limit, cost). Never plans a resting order:
    whatever the book cannot fill now is not bought (module D: resting orders get picked off)."""
    qty, cost, limit = 0, 0.0, book.best_ask or 0.0
    for px, sz in book.asks:
        if px > max_price + 1e-12 or budget - cost < px:
            break
        take = min(int(sz), math.floor((budget - cost) / px))
        if take <= 0:
            break
        qty, cost, limit = qty + take, cost + take * px, px
    return qty, limit, cost


def build(markets, books, estimates, account: Account, cfg: Config, now: datetime,
          held: dict[str, float] | None = None) -> list[Proposal]:
    props = []
    for c, stake, why in size(screen(markets, books, estimates, cfg, now), account, cfg, held):
        a = c.book.best_ask
        policy_stake = stake
        qty, limit, cost = fill_plan(c.book, c.est.p - cfg.min_edge, stake)
        if stake > 0 and qty < math.floor(stake / a):
            c.flags.append("sized to the book: not enough contracts offered with edge")
        stake = cost
        rationale = (f"Buy '{c.contract_name}' at {a:.2f}: pooled probability {c.est.p:.2f} "
                     f"[{c.est.lo:.2f}, {c.est.hi:.2f}] vs ask {a:.2f}, edge {c.edge:+.2f}; Kelly {c.kelly:.1%} of "
                     f"equity. Sizing: {why}. Sources: {c.est.explanation}.")
        props.append(Proposal(
            id=uuid.uuid4().hex[:10], created=now.isoformat(timespec="seconds"), market_id=c.market.id,
            market_title=c.market.title, category=c.market.category, contract_id=c.contract_id,
            contract_name=c.contract_name, p=c.est.p, lo=c.est.lo, hi=c.est.hi, bid=c.book.best_bid, ask=a,
            edge=c.edge, kelly=c.kelly, mode=cfg.mode, stake=float(stake), qty=float(qty), limit_price=limit,
            mapping_confirmed=c.confirmed, mapping_keys=c.keys, components=c.est.components, rationale=rationale,
            flags=c.flags, expires=(now + timedelta(minutes=cfg.proposal_ttl_minutes)).isoformat(timespec="seconds"),
            policy_stake=float(policy_stake)))
    return props
