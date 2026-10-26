"""The hosted desk: anyone can open it, and every visitor gets a private paper contest.

    python -m markout.cup public                           # http://127.0.0.1:7860
    HOST=0.0.0.0 TRUST_PROXY=1 python -m markout.cup public  # behind a proxy (the container)

The page asks for a session (POST /api/session) and sends its id back in the
X-Markout-Session header on every call. A session owns a paper exchange with a random
seed, its own store, journal and kill switch (all in one temporary directory), and a
manual clock that the visitor moves forward. Nothing is shared between sessions and
nothing leaves the process: references are the scenario's bundled quotes, so the hosted
desk makes no outbound network calls.

The session id travels in a header rather than a cookie. Browsers never attach it on
their own, so another site cannot make requests on a visitor's behalf (no CSRF), and the
desk keeps working inside an iframe on another site (a Hugging Face Space page), where
third-party cookies are blocked.

What keeps one small public instance healthy:
- at most `max_sessions` sessions; one more evicts the least recently used;
- a session idle for `ttl` seconds is deleted (a janitor sweeps every minute);
- a token bucket per session, and a slower one per client for creating sessions;
- request bodies over MAX_BODY bytes are refused, and at most MAX_CONCURRENT requests
  are served at once (503 beyond that);
- the clock moves at most MAX_ADVANCE_HOURS per call.
The page runs under a Content-Security-Policy that allows only its own inline scripts
(by hash), and every request is logged to stdout as one JSON line, with a hash of the
session id instead of the id.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import os
import secrets
import shutil
import signal
import sqlite3
import sys
import tempfile
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from html.parser import HTMLParser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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
from markout.cup.web import page

START = datetime(2026, 10, 1, 16, 0, tzinfo=UTC)  # the contest opened at 12:00 ET
START_CASH = 10_000.0
MAX_SESSIONS = 200
TTL = 30 * 60.0
MAX_BODY = 16_384
MAX_CONCURRENT = 32
MAX_ADVANCE_HOURS = 72.0
HEADER = "X-Markout-Session"
FRAME_ANCESTORS = "'self' https://huggingface.co https://*.hf.space"

# Prepended to the desk's own script: get a session, send it with every request, and
# wire the time controls. sessionStorage keeps one sandbox per tab across reloads.
SHIM = r"""<script>
(()=>{const KEY='markout-session',raw=window.fetch.bind(window);let sid=null,inflight=null;
try{sid=sessionStorage.getItem(KEY)}catch(e){}
function session(){if(sid)return Promise.resolve(sid);
inflight=inflight||raw('/api/session',{method:'POST'}).then(r=>r.json()).then(j=>{if(!j.ok)throw new Error(j.message||'no session');
sid=j.session;try{sessionStorage.setItem(KEY,sid)}catch(e){}const b=document.querySelector('#sbx');
if(b)b.textContent=`seed ${j.seed}, deleted after ${Math.round(j.ttl_s/60)} min idle`;return sid}).finally(()=>{inflight=null});return inflight}
window.fetch=async(u,o={})=>{for(let i=0;i<2;i++){const h=new Headers(o.headers||{});h.set('X-Markout-Session',await session());
const r=await raw(u,{...o,headers:h});if(r.status!==401)return r;sid=null;try{sessionStorage.removeItem(KEY)}catch(e){}}
return new Response(JSON.stringify({ok:false,message:'session expired, reload the page'}),{status:401})};
document.addEventListener('click',e=>{const b=e.target.closest('button[data-time]');if(!b)return;
if(b.dataset.time==='reset'){if(confirm('Start over with a fresh sandbox?'))post('/api/reset').then(()=>markets())}
else post('/api/advance',{hours:Number(b.dataset.time)}).then(()=>markets())});})();
</script>"""

CONTROLS = (
    ' <span class="chip" title="Move the simulated clock">time</span>'
    '<button data-time="2">+2 h</button><button data-time="24">+1 day</button>'
    '<button data-time="reset" title="Start over with a fresh sandbox">Reset</button>'
)
NOTICE = (
    '<div style="padding:8px 20px;font-size:13px;background:var(--warn-bg);color:var(--warn-ink)">'
    "<b>Public sandbox.</b> Fictional markets and paper money, private to this tab. Move the clock with "
    "<b>+2 h</b> / <b>+1 day</b>: prices drift, markets resolve and the journal scores your calls. "
    '<span id="sbx"></span></div>'
)
KILL_HELP = "Kill switch: the button at the top. It stops orders in this sandbox only."


class _InlineScripts(HTMLParser):
    """Collects the text of every inline <script> element, exactly as the browser hashes it."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.blocks: list[str] = []
        self._inside = False

    def handle_starttag(self, tag, attrs):
        if tag == "script" and not dict(attrs).get("src"):
            self._inside = True
            self.blocks.append("")

    def handle_endtag(self, tag):
        if tag == "script":
            self._inside = False

    def handle_data(self, data):
        if self._inside:
            self.blocks[-1] += data


def _csp_hashes(html: str) -> str:
    parser = _InlineScripts()
    parser.feed(html)
    parser.close()
    return " ".join(f"'sha256-{base64.b64encode(hashlib.sha256(b.encode()).digest()).decode()}'" for b in parser.blocks)


def public_page() -> tuple[str, str]:
    """The page and its Content-Security-Policy (scripts allowed by hash only)."""
    html = page("public", controls=CONTROLS, notice=NOTICE, kill_help=KILL_HELP).replace(
        "<script>", SHIM + "<script>", 1
    )
    csp = (
        f"default-src 'none'; script-src {_csp_hashes(html)}; style-src 'unsafe-inline'; img-src data:; "
        f"connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors {FRAME_ANCESTORS}"
    )
    return html, csp


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


def _log_json(event: dict) -> None:
    print(json.dumps(event, default=str), flush=True)


class PublicDesk:
    def __init__(
        self,
        sessions: Sessions,
        trust_proxy: bool = False,
        create_rate: float = 0.2,
        create_burst: int = 10,
        log: Callable[[dict], None] = _log_json,
    ):
        self.sessions, self.trust_proxy, self.log = sessions, trust_proxy, log
        self.create_rate, self.create_burst = create_rate, create_burst
        self.html, self.csp = public_page()
        self.slots = threading.BoundedSemaphore(MAX_CONCURRENT)
        self._clients: dict[str, tuple[RateLimiter, float]] = {}
        self._clients_lock = threading.Lock()
        self.started = time.monotonic()

    def allow_create(self, client: str) -> bool:
        now = time.monotonic()
        with self._clients_lock:
            if len(self._clients) > 10_000:  # forget clients idle for 10 minutes
                self._clients = {k: v for k, v in self._clients.items() if now - v[1] < 600}
            entry = self._clients.get(client)
            lim = entry[0] if entry else RateLimiter(self.create_rate, self.create_burst)
            self._clients[client] = (lim, now)
        return lim.try_acquire()

    # ------------------------------------------------------------------ routes
    def session_routes(self, s: Session, body: dict) -> dict[str, Callable[[], dict]]:
        b = s.bot
        return {
            "/api/approve": lambda: b.approve(str(body["id"]), _stake(body.get("stake")), str(body.get("note", ""))),
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
            "/api/advance": lambda: self.advance(s, body.get("hours")),
            "/api/reset": lambda: self.reset(s),
        }

    def advance(self, s: Session, hours: Any) -> dict:
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

    def reset(self, s: Session) -> dict:
        new = self.sessions.reset(s)
        return {"ok": True, "message": f"fresh sandbox (seed {new.seed})", "seed": new.seed}

    def handler(self) -> type[BaseHTTPRequestHandler]:
        desk = self

        class H(BaseHTTPRequestHandler):
            server_version = "markout-desk"
            sys_version = ""
            _status: int = 0
            _sid: str | None = None

            def log_message(self, *a):  # replaced by one JSON line per request
                return

            def client(self) -> str:
                fwd = self.headers.get("X-Forwarded-For") if desk.trust_proxy else None
                return fwd.split(",")[0].strip() if fwd else self.client_address[0]

            def _send(
                self, code: int, body: bytes, ctype: str, csp: str = "default-src 'none'; frame-ancestors 'none'"
            ):
                self._status = int(code)
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Security-Policy", csp)
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header("Cross-Origin-Resource-Policy", "same-origin")
                self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(body)

            def _json(self, obj: Any, code: int = 200) -> None:
                self._send(code, json.dumps(obj, default=str).encode(), "application/json")

            def _serve(self, fn: Callable[[], None]) -> None:
                t0, self._status, self._sid = time.perf_counter(), 0, None
                if not desk.slots.acquire(blocking=False):
                    self._json({"ok": False, "message": "busy, try again"}, HTTPStatus.SERVICE_UNAVAILABLE)
                else:
                    try:
                        fn()
                    except Exception as e:  # noqa: BLE001 - one bad request must not take the desk down
                        desk.log({"level": "error", "path": self.path, "error": f"{type(e).__name__}: {e}"})
                        if not self._status:
                            self._json({"ok": False, "message": "internal error"}, HTTPStatus.INTERNAL_SERVER_ERROR)
                    finally:
                        desk.slots.release()
                desk.log(
                    {
                        "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
                        "client": self.client(),
                        "method": self.command,
                        "path": self.path.split("?")[0][:80],
                        "status": self._status,
                        "ms": round(1e3 * (time.perf_counter() - t0), 1),
                        "session": hashlib.sha256(self._sid.encode()).hexdigest()[:10] if self._sid else None,
                    }
                )

            def _session(self) -> Session | None:
                self._sid = self.headers.get(HEADER)
                s = desk.sessions.get(self._sid)
                if s is None:
                    self._json({"ok": False, "message": "no session: reload the page"}, HTTPStatus.UNAUTHORIZED)
                    return None
                if not s.limiter.try_acquire():
                    self._json({"ok": False, "message": "slow down"}, HTTPStatus.TOO_MANY_REQUESTS)
                    return None
                return s

            def _get(self) -> None:
                path = self.path.split("?")[0]
                if path in ("/", "/index.html"):
                    return self._send(200, desk.html.encode(), "text/html; charset=utf-8", desk.csp)
                if path == "/healthz":
                    return self._json(
                        {
                            "ok": True,
                            "sessions": len(desk.sessions),
                            "max_sessions": desk.sessions.max,
                            "uptime_s": round(time.monotonic() - desk.started),
                        }
                    )
                if path in ("/api/state", "/api/markets"):
                    s = self._session()
                    if s is not None:
                        b = s.bot
                        self._json(b.state() if path == "/api/state" else b.markets_view())
                    return None
                return self._json({"ok": False, "message": "not found"}, HTTPStatus.NOT_FOUND)

            def _post(self) -> None:
                n = int(self.headers.get("Content-Length") or 0)
                if n > MAX_BODY:
                    self.close_connection = True
                    return self._json(
                        {"ok": False, "message": "request too large"}, HTTPStatus.REQUEST_ENTITY_TOO_LARGE
                    )
                raw = self.rfile.read(n) if n > 0 else b""
                if self.path == "/api/session":
                    if not desk.allow_create(self.client()):
                        return self._json(
                            {"ok": False, "message": "too many new sessions"}, HTTPStatus.TOO_MANY_REQUESTS
                        )
                    new = desk.sessions.create()
                    self._sid = new.id
                    return self._json({"ok": True, "session": new.id, "seed": new.seed, "ttl_s": desk.sessions.ttl})
                try:
                    body = json.loads(raw or b"{}")
                except json.JSONDecodeError:
                    return self._json({"ok": False, "message": "bad json"}, HTTPStatus.BAD_REQUEST)
                if not isinstance(body, dict):
                    return self._json({"ok": False, "message": "expected a JSON object"}, HTTPStatus.BAD_REQUEST)
                s = self._session()
                if s is None:
                    return None
                routes = desk.session_routes(s, body)
                if self.path not in routes:
                    return self._json({"ok": False, "message": "not found"}, HTTPStatus.NOT_FOUND)
                try:
                    return self._json(routes[self.path]())
                except (KeyError, ValueError, TypeError, AssertionError) as e:
                    return self._json({"ok": False, "message": f"bad request: {e}"}, HTTPStatus.BAD_REQUEST)
                except sqlite3.ProgrammingError:  # the session was evicted while this request waited
                    return self._json({"ok": False, "message": "no session: reload the page"}, HTTPStatus.UNAUTHORIZED)

            def do_GET(self):
                self._serve(self._get)

            def do_HEAD(self):
                self._serve(self._get)

            def do_POST(self):
                self._serve(self._post)

        return H

    def serve(self, host: str = "127.0.0.1", port: int = 7860) -> ThreadingHTTPServer:
        return _Server((host, port), self.handler())


class _Server(ThreadingHTTPServer):
    request_queue_size = 128  # socketserver's default backlog of 5 resets bursts of browser connections


def _stake(x: Any) -> float | None:
    if x is None:
        return None
    v = float(x)
    if not 0 <= v < 1e12:
        raise ValueError("stake must be a non-negative number")
    return v


def janitor(sessions: Sessions, stop: threading.Event, every: float = 60.0) -> None:
    while not stop.wait(every):
        sessions.sweep()


def main(host: str | None = None, port: int | None = None, max_sessions: int = MAX_SESSIONS, ttl: float = TTL) -> int:
    host = host or os.environ.get("HOST", "127.0.0.1")
    port = port or int(os.environ.get("PORT", "7860"))
    root = Path(tempfile.mkdtemp(prefix="markout-desk-"))
    sessions = Sessions(root, max_sessions=max_sessions, ttl=ttl)
    desk = PublicDesk(sessions, trust_proxy=os.environ.get("TRUST_PROXY") == "1")
    srv = desk.serve(host, port)
    stop = threading.Event()
    threading.Thread(target=janitor, args=(sessions, stop), daemon=True).start()
    signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=srv.shutdown, daemon=True).start())
    _log_json({"level": "info", "message": f"public desk on http://{host}:{port}", "max_sessions": max_sessions})
    try:
        with contextlib.suppress(KeyboardInterrupt):  # Ctrl-C ends serve_forever; the cleanup still runs
            srv.serve_forever()
    finally:
        stop.set()
        srv.server_close()
        sessions.close()
        shutil.rmtree(root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
