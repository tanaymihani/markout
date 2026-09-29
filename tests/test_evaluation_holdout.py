"""Tests for markout.evaluation.holdout: the holdout can be opened once, visibly."""

import json
import threading
import time

import numpy as np
import pandas as pd
import pytest

from markout.evaluation.holdout import HoldoutAlreadyUsed, HoldoutGuard
from markout.evaluation.registry import config_hash

FROZEN = {"model": "lgbm", "m": 1.5, "features": ["imb", "spread"]}


def _guard(tmp_path) -> HoldoutGuard:
    return HoldoutGuard(tmp_path / "holdout.lock", tmp_path / "holdout_audit.jsonl")


class Counter:
    def __init__(self, result=None):
        self.calls = 0
        self.result = result or {"mae": 5.8, "sharpe": 0.04}

    def __call__(self):
        self.calls += 1
        return self.result


def test_first_run_executes_and_writes_the_lock(tmp_path):
    g, fn = _guard(tmp_path), Counter()
    assert g.status()["locked"] is False and g.accessed_count() == 0
    assert g.run(fn, FROZEN) == fn.result and fn.calls == 1

    lock = json.loads((tmp_path / "holdout.lock").read_text())
    assert lock["status"] == "complete"
    assert lock["config_hash"] == config_hash(FROZEN) and lock["config"] == FROZEN
    assert pd.Timestamp(lock["started_at"]).utcoffset() == pd.Timedelta(0)
    assert "git_sha" in lock and "git_dirty" in lock
    assert lock["result_summary"] == {"mae": 5.8, "sharpe": 0.04}
    events = [e["event"] for e in g.audit()]
    assert events == ["start", "finish"] and g.accessed_count() == 1


def test_second_run_is_refused_without_touching_the_holdout(tmp_path):
    g, fn = _guard(tmp_path), Counter()
    g.run(fn, FROZEN)
    with pytest.raises(HoldoutAlreadyUsed) as exc:
        g.run(fn, {**FROZEN, "m": 2.0})                 # a "tweaked" config: still refused
    assert fn.calls == 1
    assert exc.value.lock["config_hash"] == config_hash(FROZEN)
    assert exc.value.accessed_count == 1
    assert config_hash(FROZEN)[:12] in str(exc.value) and "force=True" in str(exc.value)
    st = g.status()
    assert st["accessed_count"] == 1 and st["refused_count"] == 1 and st["forced_count"] == 0


def test_forced_run_executes_and_is_audited(tmp_path):
    g, fn = _guard(tmp_path), Counter()
    g.run(fn, FROZEN)
    first_lock = (tmp_path / "holdout.lock").read_text()
    g.run(fn, FROZEN, force=True)
    assert fn.calls == 2
    starts = [e for e in g.audit() if e["event"] == "start"]
    assert [e["forced"] for e in starts] == [False, True]
    assert g.accessed_count() == 2 and g.status()["forced_count"] == 1
    assert (tmp_path / "holdout.lock").read_text() == first_lock   # lock keeps the first use


def test_failed_run_still_consumes_the_holdout(tmp_path):
    g = _guard(tmp_path)

    def boom():
        raise RuntimeError("model file missing")

    with pytest.raises(RuntimeError, match="model file missing"):
        g.run(boom, FROZEN)
    lock = g.status()["lock"]
    assert lock["status"] == "error" and "model file missing" in lock["error"]
    assert g.accessed_count() == 1
    with pytest.raises(HoldoutAlreadyUsed):
        g.run(Counter(), FROZEN)
    g.run(Counter(), FROZEN, force=True)
    assert g.accessed_count() == 2
    finishes = [e["status"] for e in g.audit() if e["event"] == "finish"]
    assert finishes == ["error", "complete"]


def test_concurrent_first_runs_open_the_holdout_exactly_once(tmp_path):
    g = _guard(tmp_path)
    barrier = threading.Barrier(4)
    outcomes, calls = [], []

    def slow():
        calls.append(1)
        time.sleep(0.2)
        return {"ok": True}

    def worker():
        barrier.wait()
        try:
            g.run(slow, FROZEN)
            outcomes.append("ran")
        except HoldoutAlreadyUsed:
            outcomes.append("refused")

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert sorted(outcomes) == ["ran", "refused", "refused", "refused"]
    assert len(calls) == 1 and g.accessed_count() == 1


def test_result_summary_is_json_safe(tmp_path):
    g = _guard(tmp_path)
    result = {"mae": np.float64(5.7), "n_days": np.int64(60), "beats_baseline": np.bool_(True),
              "ci": {"lo": -0.1, "hi": np.float32(0.3)}, "daily_pnl": pd.Series(np.ones(60)),
              "nan_metric": float("nan"), "model": object()}
    assert g.run(lambda: result, FROZEN) is result
    summary = json.loads((tmp_path / "holdout.lock").read_text())["result_summary"]
    assert summary["mae"] == 5.7 and summary["n_days"] == 60 and summary["beats_baseline"] is True
    assert summary["ci"]["lo"] == -0.1 and summary["ci"]["hi"] == pytest.approx(0.3)
    assert summary["daily_pnl"] == "<Series len=60>" and summary["nan_metric"] is None
    assert summary["model"] == "<object>"


def test_count_survives_a_deleted_audit_log(tmp_path):
    g = _guard(tmp_path)
    g.run(Counter(), FROZEN)
    (tmp_path / "holdout_audit.jsonl").unlink()
    assert g.accessed_count() == 1                        # the lock alone proves one access
    with pytest.raises(HoldoutAlreadyUsed):
        g.run(Counter(), FROZEN)
