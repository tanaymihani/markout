"""Private paper-trading sandboxes: each one has its own bot, exchange, journal, kill switch and clock.

Both public versions of the desk are built on this module: the HTTP server in `cup.public`
(many visitors, one process) and `cup.browser` (one visitor, inside their own browser tab,
on Pyodide). Nothing here touches the network, so it runs unchanged on WebAssembly.

A sandbox starts on the bundled, fictional scenario with a random seed and a manual clock
that only the visitor moves; references are the scenario's own quotes.
"""

from __future__ import annotations

import secrets
import shutil
import tempfile
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from markout.cup import proposals as P
from markout.cup.api import RateLimiter
from markout.cup.clock import ManualClock
from markout.cup.paper import SCENARIO, PaperContest, load_scenario
from markout.cup.refs import References
from markout.cup.service import Bot
from markout.cup.sources.static import bundled
from markout.cup.store import Store

START = datetime(2026, 10, 1, 16, 0, tzinfo=UTC)  # the contest opened at 12:00 ET
START_CASH = 10_000.0
MAX_SESSIONS = 200
TTL = 30 * 60.0
MAX_ADVANCE_HOURS = 72.0


@dataclass
class Session:
    id: str
    dir: Path
    bot: Bot
    clock: ManualClock
    seed: int
    limiter: RateLimiter
    seen: float


class Sessions:
    """The live sandboxes, least recently used first. Safe to call from any thread."""

    def __init__(
        self,
        root: Path,
        max_sessions: int = MAX_SESSIONS,
        ttl: float = TTL,
        rate: float = 5.0,
        burst: int = 30,
        scenario: Path = SCENARIO,
        now: Callable[[], float] = time.monotonic,
    ):
        self.root, self.max, self.ttl, self.rate, self.burst = Path(root), max_sessions, ttl, rate, burst
        self.scenario, self.now = scenario, now
        self._live: OrderedDict[str, Session] = OrderedDict()
        self.lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._live)

    def _build(self, sid: str, seed: int | None) -> Session:
        seed = secrets.randbelow(2**31) if seed is None else seed
        d = Path(tempfile.mkdtemp(prefix="s-", dir=self.root))
        clock = ManualClock(START)
        specs = load_scenario(self.scenario, START)
        api = PaperContest(specs, start_cash=START_CASH, seed=seed, clock=clock)
        bot = Bot(
            api,
            Store(d / "desk.db"),
            References(live=False, static=bundled(specs)),
            P.Config(mode="steady"),
            journal_path=d / "journal.csv",
            clock=clock,
            stop_file=d / "STOP",
        )
        bot.refresh()
        return Session(sid, d, bot, clock, seed, RateLimiter(rate=self.rate, burst=self.burst), self.now())

    def _drop(self, s: Session) -> None:
        with s.bot.lock, s.bot.store.lock:  # wait for a request in flight; later ones see a closed store
            s.bot.store.db.close()
        shutil.rmtree(s.dir, ignore_errors=True)

    def create(self, seed: int | None = None) -> Session:
        s = self._build(secrets.token_urlsafe(24), seed)
        with self.lock:
            self._live[s.id] = s
            evicted = [self._live.popitem(last=False)[1] for _ in range(len(self._live) - self.max)]
        for old in evicted:
            self._drop(old)
        return s

    def get(self, sid: str | None) -> Session | None:
        if not sid:
            return None
        with self.lock:
            s = self._live.get(sid)
            if s is None:
                return None
            expired = self.now() - s.seen > self.ttl
            if expired:
                del self._live[sid]
            else:
                s.seen = self.now()
                self._live.move_to_end(sid)
        if expired:
            self._drop(s)
            return None
        return s

    def reset(self, s: Session) -> Session:
        """Same id, a fresh sandbox with a new seed."""
        new = self._build(s.id, None)
        with self.lock:
            if s.id in self._live:
                self._live[s.id] = new
                self._live.move_to_end(s.id)
        self._drop(s)
        return new

    def sweep(self) -> int:
        cutoff = self.now() - self.ttl
        with self.lock:
            dead = [s for s in self._live.values() if s.seen < cutoff]
            for s in dead:
                del self._live[s.id]
        for s in dead:
            self._drop(s)
        return len(dead)

    def close(self) -> None:
        with self.lock:
            live, self._live = list(self._live.values()), OrderedDict()
        for s in live:
            self._drop(s)


def view(s: Session, path: str) -> Any:
    """The two GET routes: the desk's state and the market list. None for any other path."""
    if path == "/api/state":
        return s.bot.state()
    if path == "/api/markets":
        return s.bot.markets_view()
    return None


def routes(sessions: Sessions, s: Session, body: dict) -> dict[str, Callable[[], dict]]:
    """The POST routes of one sandbox, i.e. what the page's buttons call. Bad input raises
    KeyError, ValueError or TypeError, which the callers answer with a 400."""
    b = s.bot
    return {
        "/api/approve": lambda: b.approve(str(body["id"]), stake(body.get("stake")), str(body.get("note", ""))),
        "/api/skip": lambda: b.skip(str(body["id"]), str(body.get("note", ""))),
        "/api/mapping": lambda: b.set_mapping(str(body["contract_id"]), str(body["key"]), str(body["status"])),
        "/api/mode": lambda: b.set_mode(str(body["mode"])),
        "/api/refresh": lambda: {"ok": True, "message": f"refreshed: {b.refresh()}"},
        "/api/kill": lambda: b.set_kill(bool(body.get("on"))),
        "/api/search": lambda: b.search(str(body["contract_id"]), str(body.get("query", ""))),
        "/api/mapping/add": lambda: b.add_manual_mapping(
            str(body["contract_id"]),
            str(body["source"]),
            str(body["qid"]),
            str(body["outcome"]),
            bool(body.get("invert")),
        ),
        "/api/advance": lambda: advance(s, body.get("hours")),
        "/api/reset": lambda: reset(sessions, s),
    }


def advance(s: Session, hours: Any) -> dict:
    """Move the sandbox's clock forward and refresh: prices drift and markets resolve."""
    h = float(hours)
    if not 0 < h <= MAX_ADVANCE_HOURS:
        raise ValueError(f"hours must be in (0, {MAX_ADVANCE_HOURS:g}]")
    with s.bot.lock:
        s.clock.advance(timedelta(hours=h))
        info = s.bot.refresh()
    left = info["markets"]
    return {
        "ok": True,
        "message": f"+{h:g} h: {info['proposals']} new proposal(s), {left} market(s) open"
        if left
        else f"+{h:g} h: every market has resolved",
    }


def reset(sessions: Sessions, s: Session) -> dict:
    new = sessions.reset(s)
    return {"ok": True, "message": f"fresh sandbox (seed {new.seed})", "seed": new.seed}


def stake(x: Any) -> float | None:
    if x is None:
        return None
    v = float(x)
    if not 0 <= v < 1e12:
        raise ValueError("stake must be a non-negative number")
    return v
