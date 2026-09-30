"""Local SQLite store for proposals, mappings, events and account snapshots (data/cup/,
gitignored). Rows hold JSON so the schema can evolve while the contest runs."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from markout.paths import DATA

DB = DATA / "cup" / "cup.db"

SCHEMA = """
create table if not exists proposals (id text primary key, contract_id text, status text, created text, body text);
create table if not exists mappings (contract_id text, key text, status text, body text,
                                     primary key (contract_id, key));
create table if not exists events (ts text, kind text, body text);
create table if not exists snapshots (ts text, body text);
create table if not exists settings (k text primary key, v text);
"""


class Store:
    def __init__(self, path: Path = DB):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = Path(path)
        self.lock = threading.Lock()
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.executescript(SCHEMA)
        self.db.commit()

    def _q(self, sql: str, args=()) -> list[tuple]:
        with self.lock:
            cur = self.db.execute(sql, args)
            rows = cur.fetchall()
            self.db.commit()
            return rows

    # proposals
    def put_proposal(self, p: dict) -> None:
        self._q("insert or replace into proposals values (?,?,?,?,?)",
                (p["id"], p["contract_id"], p["status"], p["created"], json.dumps(p)))

    def proposal(self, pid: str) -> dict | None:
        r = self._q("select body from proposals where id=?", (pid,))
        return json.loads(r[0][0]) if r else None

    def proposals(self, status: str | None = None, limit: int = 500) -> list[dict]:
        if status:
            rows = self._q("select body from proposals where status=? order by created desc limit ?", (status, limit))
        else:
            rows = self._q("select body from proposals order by created desc limit ?", (limit,))
        return [json.loads(r[0]) for r in rows]

    # mappings: status is suggested | confirmed | rejected
    def put_mapping(self, contract_id: str, key: str, status: str, body: dict) -> None:
        self._q("insert or replace into mappings values (?,?,?,?)", (contract_id, key, status, json.dumps(body)))

    def mappings(self, contract_id: str | None = None) -> list[dict]:
        if contract_id:
            rows = self._q("select contract_id, key, status, body from mappings where contract_id=?", (contract_id,))
        else:
            rows = self._q("select contract_id, key, status, body from mappings")
        return [{"contract_id": c, "key": k, "status": s, **json.loads(b)} for c, k, s, b in rows]

    def mapping_status(self, contract_id: str, key: str) -> str | None:
        r = self._q("select status from mappings where contract_id=? and key=?", (contract_id, key))
        return r[0][0] if r else None

    # events, snapshots, settings
    def log(self, ts: str, kind: str, body: dict) -> None:
        self._q("insert into events values (?,?,?)", (ts, kind, json.dumps(body, default=str)))

    def events(self, limit: int = 200) -> list[dict]:
        rows = self._q("select ts, kind, body from events order by rowid desc limit ?", (limit,))
        return [{"ts": t, "kind": k, **json.loads(b)} for t, k, b in rows]

    def snapshot(self, ts: str, body: dict) -> None:
        self._q("insert into snapshots values (?,?)", (ts, json.dumps(body)))

    def snapshots(self) -> list[dict]:
        return [{"ts": t, **json.loads(b)} for t, b in self._q("select ts, body from snapshots order by rowid")]

    def get(self, k: str, default: str | None = None) -> str | None:
        r = self._q("select v from settings where k=?", (k,))
        return r[0][0] if r else default

    def set(self, k: str, v: str) -> None:
        self._q("insert or replace into settings values (?,?)", (k, v))
