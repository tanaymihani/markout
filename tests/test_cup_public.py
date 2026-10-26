"""The hosted desk: sessions are isolated, limited and expire; the page is locked down."""

import base64
import hashlib
import json
import re
import threading
import urllib.error
import urllib.request
from html.parser import HTMLParser

import pytest

from markout.cup import risk
from markout.cup.public import HEADER, MAX_BODY, PublicDesk, Sessions
from markout.decision import journal


class FakeTime:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setattr(risk, "STOP_FILE", tmp_path / "GLOBAL_STOP")
    events = []
    now = FakeTime()
    sessions = Sessions(tmp_path / "sessions", max_sessions=3, ttl=600, rate=50, burst=25, now=now)
    (tmp_path / "sessions").mkdir()
    desk = PublicDesk(sessions, create_rate=0.001, create_burst=5, log=events.append)
    srv = desk.serve("127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield {"port": srv.server_address[1], "desk": desk, "sessions": sessions, "events": events, "now": now}
    srv.shutdown()
    srv.server_close()
    sessions.close()


class _Scripts(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.blocks, self.inside = [], False

    def handle_starttag(self, tag, attrs):
        if tag == "script":
            self.inside = True
            self.blocks.append("")

    def handle_endtag(self, tag):
        self.inside = self.inside and tag != "script"

    def handle_data(self, data):
        if self.inside:
            self.blocks[-1] += data


def inline_scripts(html: str) -> list[str]:
    """Every <script> body, read with a real HTML parser (independent of the server's)."""
    p = _Scripts()
    p.feed(html)
    p.close()
    return p.blocks


def call(port, path, body=None, sid=None, raw=None, method=None):
    data = raw if raw is not None else (None if body is None else json.dumps(body).encode())
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=data, method=method or ("GET" if data is None else "POST")
    )
    if sid:
        req.add_header(HEADER, sid)
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            text = r.read().decode()
            return (
                r.status,
                (json.loads(text) if r.headers.get_content_type() == "application/json" else text),
                r.headers,
            )
    except urllib.error.HTTPError as e:
        text = e.read().decode()
        return e.code, (json.loads(text) if text.startswith("{") else text), e.headers


def new_session(port):
    code, j, _ = call(port, "/api/session", {})
    assert code == 200 and j["ok"]
    return j["session"]


def test_health_needs_no_session_and_the_api_does(server):
    port = server["port"]
    code, j, _ = call(port, "/healthz")
    assert code == 200 and j["ok"] and j["sessions"] == 0
    assert call(port, "/api/state")[0] == 401
    assert call(port, "/api/state", sid="made-up")[0] == 401
    assert call(port, "/api/refresh", {}, sid="made-up")[0] == 401
    assert call(port, "/nope")[0] == 404


def test_sessions_are_isolated(server):
    port = server["port"]
    a, b = new_session(port), new_session(port)
    assert a != b
    sa = call(port, "/api/state", sid=a)[1]
    sb = call(port, "/api/state", sid=b)[1]
    assert sa["account"]["cash"] == sb["account"]["cash"] == 10_000
    assert sa["pending"], "a new sandbox starts with proposals"
    p = next(x for x in sa["pending"] if x["mapping_confirmed"])
    code, res, _ = call(port, "/api/approve", {"id": p["id"]}, sid=a)
    assert code == 200 and res["ok"], res
    sa = call(port, "/api/state", sid=a)[1]
    sb = call(port, "/api/state", sid=b)[1]
    assert sa["positions"] and not sb["positions"]
    assert sa["account"]["cash"] < 10_000 == sb["account"]["cash"]
    # a proposal id from one sandbox means nothing in another
    assert not call(port, "/api/approve", {"id": p["id"]}, sid=b)[1]["ok"]


def test_the_clock_moves_per_session_and_within_bounds(server):
    port = server["port"]
    a, b = new_session(port), new_session(port)
    t0 = call(port, "/api/state", sid=a)[1]["now"]
    code, res, _ = call(port, "/api/advance", {"hours": 24}, sid=a)
    assert code == 200 and res["ok"]
    assert call(port, "/api/state", sid=a)[1]["now"] > t0 == call(port, "/api/state", sid=b)[1]["now"]
    for bad in (0, -1, 1000, "soon", None):
        assert call(port, "/api/advance", {"hours": bad}, sid=a)[0] == 400


def test_markets_resolve_and_the_journal_fills_in(server):
    port = server["port"]
    a = new_session(port)
    s = call(port, "/api/state", sid=a)[1]
    for p in s["pending"]:
        if p["mapping_confirmed"]:
            call(port, "/api/approve", {"id": p["id"], "stake": 0}, sid=a)  # log forecasts only
    for _ in range(4):
        res = call(port, "/api/advance", {"hours": 48}, sid=a)[1]
    assert res["message"].endswith("every market has resolved")
    assert call(port, "/api/markets", sid=a)[1] == []
    df = journal.load(server["sessions"].get(a).dir / "journal.csv")
    assert len(df) > 0 and df["outcome"].notna().all()  # every logged forecast is now graded


def test_kill_switch_is_per_session(server):
    port = server["port"]
    a, b = new_session(port), new_session(port)
    assert call(port, "/api/kill", {"on": True}, sid=a)[1]["ok"]
    assert call(port, "/api/state", sid=a)[1]["kill_switch"] is True
    assert call(port, "/api/state", sid=b)[1]["kill_switch"] is False
    assert not risk.STOP_FILE.exists()  # the shared file is never touched
    pa = next(x for x in call(port, "/api/state", sid=a)[1]["pending"] if x["mapping_confirmed"])
    assert "kill switch" in call(port, "/api/approve", {"id": pa["id"]}, sid=a)[1]["message"]
    pb = next(x for x in call(port, "/api/state", sid=b)[1]["pending"] if x["mapping_confirmed"])
    assert call(port, "/api/approve", {"id": pb["id"]}, sid=b)[1]["ok"]


def test_bad_bodies_are_refused(server):
    port = server["port"]
    a = new_session(port)
    assert call(port, "/api/refresh", raw=b"x" * (MAX_BODY + 1), sid=a)[0] == 413
    assert call(port, "/api/refresh", raw=b"{not json", sid=a)[0] == 400
    assert call(port, "/api/refresh", raw=b"[1, 2]", sid=a)[0] == 400
    assert call(port, "/api/approve", {}, sid=a)[0] == 400  # missing id
    assert call(port, "/api/approve", {"id": "x", "stake": -5}, sid=a)[0] == 400
    assert call(port, "/api/approve", {"id": "x", "stake": "NaN"}, sid=a)[0] == 400
    assert call(port, "/api/mode", {"mode": "yolo"}, sid=a)[1]["ok"] is False


def test_rate_limits(server):
    port = server["port"]
    a = new_session(port)
    codes = [call(port, "/api/state", sid=a)[0] for _ in range(40)]
    assert codes[0] == 200 and 429 in codes  # burst of 25
    codes = [call(port, "/api/session", {})[0] for _ in range(6)]
    assert 429 in codes  # 5 new sessions per client, then wait


def test_least_recently_used_session_is_evicted(server):
    port = server["port"]
    a = new_session(port)
    a_dir = server["sessions"].get(a).dir
    b, c = new_session(port), new_session(port)
    call(port, "/api/state", sid=a)  # a is now the most recent; b is the oldest
    d = new_session(port)
    assert call(port, "/api/state", sid=b)[0] == 401
    assert all(call(port, "/api/state", sid=x)[0] == 200 for x in (a, c, d))
    assert a_dir.exists() and len(server["sessions"]) == 3


def test_idle_sessions_expire_and_their_files_go(server):
    port, now = server["port"], server["now"]
    a, b = new_session(port), new_session(port)
    a_dir = server["sessions"].get(a).dir
    now.t += 400
    call(port, "/api/state", sid=b)
    now.t += 300  # a idle for 700 s > ttl, b for 300 s
    assert server["sessions"].sweep() == 1
    assert call(port, "/api/state", sid=a)[0] == 401 and not a_dir.exists()
    assert call(port, "/api/state", sid=b)[0] == 200


def test_reset_keeps_the_id_and_starts_over(server):
    port = server["port"]
    a = new_session(port)
    old_dir = server["sessions"].get(a).dir
    call(port, "/api/advance", {"hours": 24}, sid=a)
    code, res, _ = call(port, "/api/reset", {}, sid=a)
    assert code == 200 and res["ok"]
    s = call(port, "/api/state", sid=a)[1]
    assert s["now"].startswith("2026-10-01T16:00") and s["account"]["cash"] == 10_000
    assert not old_dir.exists()


def test_page_csp_allows_only_its_own_scripts(server):
    code, html, headers = call(server["port"], "/")
    assert code == 200
    csp = headers["Content-Security-Policy"]
    script_src = re.search(r"script-src ([^;]+)", csp).group(1)
    assert "unsafe-inline" not in script_src and "unsafe-eval" not in script_src
    blocks = inline_scripts(html)
    assert len(blocks) == 2
    for b in blocks:
        h = base64.b64encode(hashlib.sha256(b.encode()).digest()).decode()
        assert f"'sha256-{h}'" in script_src
    assert "frame-ancestors 'self' https://huggingface.co" in csp
    assert headers["X-Content-Type-Options"] == "nosniff" and headers["Referrer-Policy"] == "no-referrer"
    assert "touch data/cup/STOP" not in html and "sandbox only" in html


def test_logs_are_json_and_never_hold_a_session_id(server):
    port = server["port"]
    a = new_session(port)
    call(port, "/api/state", sid=a)
    reqs = [e for e in server["events"] if "status" in e]
    assert {e["path"] for e in reqs} >= {"/api/session", "/api/state"}
    assert all(set(e) >= {"ts", "method", "path", "status", "ms"} for e in reqs)
    assert a not in json.dumps(server["events"])
    assert any(e["session"] for e in reqs if e["path"] == "/api/state")
