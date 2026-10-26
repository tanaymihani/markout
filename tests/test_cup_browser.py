"""The in-browser desk answers like the hosted desk, and its bundle holds everything it imports."""

import io
import json
import subprocess
import sys
import zipfile

import pytest

from markout import site
from markout.cup.browser import BrowserDesk


def call(desk, method, path, body=None):
    reply = json.loads(desk.handle(method, path, "" if body is None else json.dumps(body)))
    return reply["status"], reply["body"]


@pytest.fixture
def desk():
    d = BrowserDesk(seed=3)
    yield d
    d.sessions.close()


def test_it_reads_and_trades(desk):
    code, state = call(desk, "GET", "/api/state")
    assert code == 200 and state["account"]["cash"] == 10_000 and state["pending"]
    p = next(x for x in state["pending"] if x["mapping_confirmed"])
    code, res = call(desk, "POST", "/api/approve", {"id": p["id"]})
    assert code == 200 and res["ok"], res
    assert call(desk, "GET", "/api/state")[1]["positions"]
    code, markets = call(desk, "GET", "/api/markets?x=1")
    assert code == 200 and isinstance(markets, list) and markets


def test_clock_kill_switch_and_reset(desk):
    assert call(desk, "POST", "/api/advance", {"hours": 24})[1]["ok"]
    assert call(desk, "GET", "/api/state")[1]["now"].startswith("2026-10-02")
    assert call(desk, "POST", "/api/advance", {"hours": 0})[0] == 400
    assert call(desk, "POST", "/api/kill", {"on": True})[1]["ok"]
    state = call(desk, "GET", "/api/state")[1]
    assert state["kill_switch"] is True
    p = next(x for x in state["pending"] if x["mapping_confirmed"])
    assert "kill switch" in call(desk, "POST", "/api/approve", {"id": p["id"]})[1]["message"]
    code, res = call(desk, "POST", "/api/reset")
    assert code == 200 and res["seed"] == desk.seed
    state = call(desk, "GET", "/api/state")[1]
    assert state["now"].startswith("2026-10-01T16:00") and state["kill_switch"] is False
    assert state["account"]["cash"] == 10_000


def test_bad_requests_get_the_hosted_desks_answers(desk):
    assert call(desk, "GET", "/api/nope")[0] == 404
    assert call(desk, "POST", "/api/nope", {})[0] == 404
    assert json.loads(desk.handle("POST", "/api/approve", "{not json"))["status"] == 400
    assert json.loads(desk.handle("POST", "/api/approve", "[1]"))["status"] == 400
    assert call(desk, "POST", "/api/approve", {})[0] == 400
    assert call(desk, "POST", "/api/approve", {"id": "x", "stake": -1})[0] == 400


PROBE = """
import json
import markout
from markout.cup.browser import BrowserDesk
assert markout.__file__.startswith({app!r}), markout.__file__
d = BrowserDesk(seed=1)
state = json.loads(d.handle("GET", "/api/state"))["body"]
p = next(x for x in state["pending"] if x["mapping_confirmed"])
assert json.loads(d.handle("POST", "/api/approve", json.dumps({{"id": p["id"], "stake": 0}})))["body"]["ok"]
for _ in range(4):
    msg = json.loads(d.handle("POST", "/api/advance", '{{"hours": 48}}'))["body"]["message"]
assert msg.endswith("every market has resolved"), msg
print("ok")
"""


def test_the_bundle_holds_everything_the_desk_imports(tmp_path):
    data = site.desk_bundle()
    assert len(data) < 200_000  # the desk's modules only, not the research stack
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = z.namelist()
        z.extractall(tmp_path / "app")
    assert {"configs/cup_demo_scenario.json", "configs/journal_template.csv"} <= set(names)
    assert not any(n.endswith(("report.py", "plotting.py", "public.py", "web.py")) for n in names)
    app = str(tmp_path / "app")
    r = subprocess.run(
        [sys.executable, "-c", PROBE.format(app=app)],
        cwd=tmp_path,
        env={"PYTHONPATH": app + "/src", "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr[-2000:]
