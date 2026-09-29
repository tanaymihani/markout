"""Holdout guard: the sealed test period can be opened once, and every opening is on record.

Why this exists
---------------
A holdout is only out-of-sample the first time you look at it. Run it, dislike the
number, tweak, run again, and it has become one more in-sample trial, with no trace
left behind. The guard makes that impossible to do quietly:

* The first `run` claims a lock file atomically (O_CREAT | O_EXCL) BEFORE calling fn, so
  two concurrent runs can't both reach the holdout. The lock records the UTC time,
  config hash, config, git SHA and dirty flag, and, once fn returns, a summary of the
  result.
* Any later `run` raises HoldoutAlreadyUsed with the lock's details, unless
  force=True. A forced run executes and is logged as forced.
* Every attempt goes to an append-only audit log: a "start" event before fn runs (so a
  crash or kill still counts as an access), a "finish" event with the outcome, and
  "refused" events for blocked attempts.
* `accessed_count()` counts starts, which is the number of times the holdout was
  actually opened. It's the number the README quotes ("holdout accessed: 1").

A failed first run (fn raised) still consumes the holdout. fn may have printed or
computed partial holdout results before failing, so re-running needs force=True and
shows up in the count. That is strict on purpose.
"""

from __future__ import annotations

import json
import os
import socket
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from markout.evaluation.registry import (append_line, config_hash, git_state, read_lines,
                                         to_jsonable, utc_now)
from markout.paths import REGISTRY, ROOT


class HoldoutAlreadyUsed(RuntimeError):
    """The holdout lock exists. `.lock` holds the lock record (time, config hash, git SHA)."""

    def __init__(self, lock: Mapping[str, Any], lock_path: Path, accessed: int) -> None:
        self.lock = dict(lock)
        self.lock_path = lock_path
        self.accessed_count = accessed
        super().__init__(
            f"holdout already accessed {accessed} time(s); first access {lock.get('started_at')} "
            f"(status={lock.get('status')}, config_hash={str(lock.get('config_hash'))[:12]}, "
            f"git_sha={str(lock.get('git_sha'))[:12]}). Lock file: {lock_path}. Pass force=True "
            "to run again; the run will be logged as forced and counted.")


def _summarize(result: Any) -> dict[str, Any]:
    """JSON-safe summary. Scalars and nested dicts of scalars are kept, and larger objects
    (Series, arrays, lists) are described by type and length."""
    if not isinstance(result, Mapping):
        return {"_type": type(result).__name__}
    out: dict[str, Any] = {}
    for k, v in result.items():
        if v is None or isinstance(v, (str, bool, int, float, np.bool_, np.integer, np.floating)):
            out[str(k)] = to_jsonable(v)
        elif isinstance(v, Mapping):
            out[str(k)] = _summarize(v)
        elif isinstance(v, (pd.Series, pd.DataFrame, np.ndarray, list, tuple)):
            out[str(k)] = f"<{type(v).__name__} len={len(v)}>"
        else:
            out[str(k)] = f"<{type(v).__name__}>"
    return out


class HoldoutGuard:
    """Run-once guard around the holdout evaluation. See the module docstring."""

    def __init__(self, lock_path: str | Path = REGISTRY / "holdout.lock",
                 audit_path: str | Path = REGISTRY / "holdout_audit.jsonl",
                 repo_dir: str | Path = ROOT) -> None:
        self.lock_path = Path(lock_path)
        self.audit_path = Path(audit_path)
        self.repo_dir = Path(repo_dir)

    # ---- lock file -----------------------------------------------------------------------
    def _claim(self, record: Mapping[str, Any]) -> bool:
        """Create the lock atomically. False if it already exists."""
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(to_jsonable(record), indent=2, allow_nan=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return True

    def _read_lock(self) -> dict[str, Any] | None:
        if not self.lock_path.exists():
            return None
        try:
            return json.loads(self.lock_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # Mid-claim by another process, or damaged. It still counts as locked.
            return {"status": "unreadable"}

    def _update_lock(self, updates: Mapping[str, Any]) -> None:
        lock = {**(self._read_lock() or {}), **updates}
        tmp = self.lock_path.with_name(f".{self.lock_path.name}.{uuid.uuid4().hex[:8]}.tmp")
        tmp.write_text(json.dumps(to_jsonable(lock), indent=2, allow_nan=False) + "\n",
                       encoding="utf-8")
        os.replace(tmp, self.lock_path)  # atomic

    # ---- API -------------------------------------------------------------------------------
    def run(self, fn: Callable[[], dict], config: dict, force: bool = False) -> dict:
        """Evaluate the holdout once: fn() is called only if the lock is free or force=True.

        Parameters
        ----------
        fn : zero-argument callable that retrains the frozen config and scores the
            holdout. Its return value is passed through, and a summary of its top-level
            scalars goes into the lock and the audit log.
        config : the frozen config. It is hashed and stored with the lock.
        force : run even though the holdout was already used. The run is audited as
            forced and counted.

        Raises HoldoutAlreadyUsed if the lock exists and force is False. Exceptions from
        fn propagate after being recorded as a failed access.
        """
        run_id = uuid.uuid4().hex[:12]
        prov = {"run_id": run_id, "config_hash": config_hash(config), **git_state(self.repo_dir),
                "pid": os.getpid(), "host": socket.gethostname()}
        started = utc_now()
        claimed = self._claim({"status": "running", "started_at": started, **prov,
                               "config": to_jsonable(config)})
        if not claimed and not force:
            append_line(self.audit_path, {"event": "refused", "at": started, **prov})
            raise HoldoutAlreadyUsed(self._read_lock() or {}, self.lock_path, self.accessed_count())

        forced = not claimed
        append_line(self.audit_path, {"event": "start", "at": started, "forced": forced, **prov})
        try:
            result = fn()
        except BaseException as exc:
            err = f"{type(exc).__name__}: {exc}"
            finished = utc_now()
            append_line(self.audit_path, {"event": "finish", "at": finished, "status": "error",
                                          "forced": forced, "error": err, "run_id": run_id})
            if claimed:
                self._update_lock({"status": "error", "finished_at": finished, "error": err})
            raise
        summary = _summarize(result)
        finished = utc_now()
        append_line(self.audit_path, {"event": "finish", "at": finished, "status": "complete",
                                      "forced": forced, "result_summary": summary, "run_id": run_id})
        if claimed:
            self._update_lock({"status": "complete", "finished_at": finished,
                               "result_summary": summary})
        return result

    def audit(self) -> list[dict[str, Any]]:
        """All audit events in order."""
        return read_lines(self.audit_path)

    def accessed_count(self) -> int:
        """How many times fn was actually invoked on the holdout (forced runs and failures
        included). At least 1 whenever the lock exists, even if the audit log was lost."""
        starts = sum(1 for e in self.audit() if e.get("event") == "start")
        return max(starts, 1) if self.lock_path.exists() else starts

    def status(self) -> dict[str, Any]:
        """Lock state, access counts and the latest event, for the README and reports."""
        events = self.audit()
        return {
            "locked": self.lock_path.exists(),
            "accessed_count": self.accessed_count(),
            "forced_count": sum(1 for e in events if e.get("event") == "start" and e.get("forced")),
            "refused_count": sum(1 for e in events if e.get("event") == "refused"),
            "lock": self._read_lock(),
            "last_event": events[-1] if events else None,
            "lock_path": str(self.lock_path),
            "audit_path": str(self.audit_path),
        }
