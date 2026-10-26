"""The desk in a browser tab: the hosted desk's sandbox, run by Pyodide (CPython on WebAssembly).

docs/desk/ loads Pyodide, unpacks the modules this file imports into its in-memory file
system and creates one `BrowserDesk`. The page's loader then hands every /api/ request
to `BrowserDesk.handle` instead of the network, so the visitor's sandbox lives and dies
with their tab and nothing they do is sent anywhere. The routes are the ones the hosted
desk serves (`cup.sandbox`), answered with the same JSON and status codes.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from markout.cup import sandbox


class BrowserDesk:
    def __init__(self, seed: int | None = None):
        # one visitor, for as long as the tab is open: no eviction, no expiry
        self.sessions = sandbox.Sessions(Path(tempfile.mkdtemp(prefix="markout-")), max_sessions=1, ttl=float("inf"))
        self.sid = self.sessions.create(seed).id

    @property
    def session(self) -> sandbox.Session:
        s = self.sessions.get(self.sid)
        assert s is not None, "the tab's sandbox never expires"
        return s

    @property
    def seed(self) -> int:
        return self.session.seed

    def handle(self, method: str, path: str, body: str = "") -> str:
        """One request from the page, as JSON {"status": HTTP status, "body": the response}."""
        status, payload = self._route(method.upper(), path.split("?")[0], body)
        return json.dumps({"status": status, "body": payload}, default=str)

    def _route(self, method: str, path: str, body: str) -> tuple[int, Any]:
        s = self.session
        if method == "GET":
            got = sandbox.view(s, path)
            return (200, got) if got is not None else (404, {"ok": False, "message": "not found"})
        try:
            data = json.loads(body or "{}")
        except json.JSONDecodeError:
            return 400, {"ok": False, "message": "bad json"}
        if not isinstance(data, dict):
            return 400, {"ok": False, "message": "expected a JSON object"}
        routes = sandbox.routes(self.sessions, s, data)
        if path not in routes:
            return 404, {"ok": False, "message": "not found"}
        try:
            return 200, routes[path]()
        except (KeyError, ValueError, TypeError, AssertionError) as e:
            return 400, {"ok": False, "message": f"bad request: {e}"}
