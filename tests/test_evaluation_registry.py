"""Tests for markout.evaluation.registry: the append-only trial log and effective N."""

import hashlib
import json
import re
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

from markout.evaluation import registry as registry_mod
from markout.evaluation.dsr import sharpe
from markout.evaluation.registry import Registry, cluster_trials, config_hash, git_state
from markout.paths import ROOT


def _pnl(values, start=181, name="date_id"):
    return pd.Series(np.asarray(values, dtype=float),
                     index=pd.RangeIndex(start, start + len(values), name=name))


def test_log_writes_one_complete_record(tmp_path):
    reg = Registry(tmp_path / "trials.jsonl")
    cfg = {"model": {"type": "lgbm", "lr": 0.05}, "m": 1.5, "features": ["imb", "spread"]}
    pnl = _pnl([1.0, -2.0, np.nan, 3.5])
    tid = reg.log(cfg, {"sharpe": 0.12, "mae": np.float32(5.7)}, pnl, tags={"stage": "research"})

    lines = (tmp_path / "trials.jsonl").read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["trial_id"] == tid and re.fullmatch(r"[0-9a-f]{12}", tid)
    ts = pd.Timestamp(rec["timestamp"])
    assert ts.tzinfo is not None and ts.utcoffset() == pd.Timedelta(0)
    assert rec["git_sha"] == "unknown" or re.fullmatch(r"[0-9a-f]{40}", rec["git_sha"])
    assert rec["git_dirty"] in (True, False, "unknown")
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    assert rec["config_hash"] == hashlib.sha256(canonical.encode()).hexdigest()
    assert rec["config"] == cfg and rec["tags"] == {"stage": "research"}
    assert rec["metrics"]["sharpe"] == 0.12
    assert rec["metrics"]["mae"] == pytest.approx(5.7, rel=1e-6)
    assert rec["pnl_keys"] == [181, 182, 183, 184]
    assert rec["pnl_values"] == [1.0, -2.0, None, 3.5]            # strict JSON: NaN -> null
    assert rec["pnl_key_type"] == "int" and rec["pnl_key_name"] == "date_id"


def test_config_hash_is_canonical():
    a = {"b": 1, "a": {"y": [1, 2], "x": 0.5}}
    b = {"a": {"x": 0.5, "y": [1, 2]}, "b": 1}                   # same content, other order
    assert config_hash(a) == config_hash(b)
    assert config_hash({"a": np.int64(3), "f": np.float64(0.5)}) == config_hash({"a": 3, "f": 0.5})
    assert config_hash({"a": 1}) != config_hash({"a": 2})
    with pytest.raises(TypeError):
        config_hash({"model": object()})                          # repr would be unstable


def test_appends_never_rewrite_earlier_records(tmp_path):
    path = tmp_path / "trials.jsonl"
    reg = Registry(path)
    reg.log({"i": 0}, {}, _pnl([1.0, 2.0]))
    first = path.read_bytes()
    reg.log({"i": 1}, {}, _pnl([3.0, 4.0]))
    after = path.read_bytes()
    assert after.startswith(first) and after.count(b"\n") == 2
    assert Registry(path).n_trials() == 2                         # a fresh reader sees both


def test_load_flattens_config_metrics_and_tags(tmp_path):
    reg = Registry(tmp_path / "t.jsonl")
    reg.log({"model": {"type": "ridge", "alpha": 1.0}}, {"sharpe": 0.1}, _pnl([1, 2, 3]), {"stage": "research"})
    reg.log({"model": {"type": "lgbm", "lr": 0.1}}, {"sharpe": 0.2}, _pnl([1, 2]))
    df = reg.load()
    assert list(df["config.model.type"]) == ["ridge", "lgbm"]
    assert df.loc[0, "config.model.alpha"] == 1.0 and pd.isna(df.loc[1, "config.model.alpha"])
    assert list(df["metrics.sharpe"]) == [0.1, 0.2]
    assert list(df["n_obs"]) == [3, 2]
    assert str(df["timestamp"].dt.tz) == "UTC"
    assert df.loc[0, "tags.stage"] == "research"


def test_pnl_matrix_union_missing_and_fill(tmp_path):
    reg = Registry(tmp_path / "t.jsonl")
    a = reg.log({"k": "a"}, {}, _pnl([1, 2, 3, 4, 5], start=1), {"stage": "research"})
    b = reg.log({"k": "b"}, {}, _pnl([10, 20, 30, 40, 50], start=3), {"stage": "research"})
    c = reg.log({"k": "c"}, {}, _pnl([7], start=100), {"stage": "smoke"})
    m = reg.pnl_matrix(where=lambda r: r["tags"].get("stage") == "research")
    assert list(m.columns) == [a, b]
    assert list(m.index) == [1, 2, 3, 4, 5, 6, 7] and m.index.name == "date_id"
    assert m.loc[1, b] != m.loc[1, b]                              # NaN: b has no day 1
    assert m.loc[6, b] == 40.0 and np.isnan(m.loc[7, a])
    filled = reg.pnl_matrix(where=lambda r: r["config"]["k"] in ("a", "b"), fill_value=0.0)
    assert filled.loc[1, b] == 0.0 and filled.loc[7, a] == 0.0
    assert reg.pnl_matrix().shape == (8, 3) and reg.pnl_matrix()[c].notna().sum() == 1


def test_datetime_keys_round_trip_and_mixed_key_types_rejected(tmp_path):
    reg = Registry(tmp_path / "t.jsonl")
    days = pd.date_range("2024-01-02", periods=4, freq="B", name="date")
    tid = reg.log({"k": 1}, {}, pd.Series([1.0, 2.0, 3.0, 4.0], index=days))
    m = reg.pnl_matrix()
    assert isinstance(m.index, pd.DatetimeIndex) and (m.index == days).all()
    assert m[tid].tolist() == [1.0, 2.0, 3.0, 4.0]
    reg.log({"k": 2}, {}, _pnl([1.0, 2.0]))
    with pytest.raises(ValueError, match="incompatible"):
        reg.pnl_matrix()
    assert reg.pnl_matrix(where=lambda r: r["pnl_key_type"] == "int").shape == (2, 1)


def test_bad_pnl_is_rejected(tmp_path):
    reg = Registry(tmp_path / "t.jsonl")
    with pytest.raises(ValueError, match="duplicate"):
        reg.log({}, {}, pd.Series([1.0, 2.0], index=[5, 5]))
    with pytest.raises(ValueError, match="inf"):
        reg.log({}, {}, _pnl([1.0, np.inf]))
    with pytest.raises(TypeError):
        reg.log({}, {}, pd.Series([1.0, 2.0], index=[0.5, 1.5]))   # float keys: ambiguous
    with pytest.raises(TypeError):
        reg.log({}, {}, [1.0, 2.0])                                 # not a Series
    assert reg.n_trials() == 0                                      # nothing half-written


def test_n_trials_get_and_trial_sharpes(tmp_path):
    reg = Registry(tmp_path / "t.jsonl")
    rng = np.random.default_rng(0)
    ids = [reg.log({"i": i}, {}, _pnl(rng.standard_normal(50) + 0.1 * i), {"fam": i % 2})
           for i in range(5)]
    assert reg.n_trials() == 5 and reg.n_trials(where=lambda r: r["tags"]["fam"] == 0) == 3
    assert reg.get(ids[2])["config"] == {"i": 2}
    with pytest.raises(KeyError):
        reg.get("nope")
    srs = reg.trial_sharpes()
    m = reg.pnl_matrix()
    assert list(srs.index) == ids
    np.testing.assert_allclose(srs.to_numpy(), [sharpe(m[c]) for c in ids])


def test_effective_n_merges_near_duplicates_only(tmp_path):
    reg = Registry(tmp_path / "t.jsonl")
    rng = np.random.default_rng(1)
    bases = [rng.standard_normal(240) for _ in range(3)]
    for b in bases:                                       # 3 bets x 3 near-copies (rho ~ 0.99)
        for _ in range(3):
            reg.log({}, {}, _pnl(b + 0.1 * rng.standard_normal(240)))
    assert reg.n_trials() == 9
    assert reg.effective_n(max_corr=0.9) == 3
    assert reg.effective_n(max_corr=0.999) == 9           # stricter threshold: all distinct
    labels = reg.clusters()
    assert labels.nunique() == 3 and (labels.value_counts() == 3).all()

    reg.log({}, {}, _pnl(-bases[0]))                      # the mirror-image bet is a new bet
    assert reg.effective_n() == 4
    reg.log({}, {}, _pnl(np.zeros(240)))                  # never traded: one extra cluster
    reg.log({}, {}, _pnl(np.zeros(240)))
    assert reg.effective_n() == 5 and reg.n_trials() == 12


def test_cluster_trials_treats_unknown_correlation_as_independent():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(100)
    pnl = pd.DataFrame({"a": x, "b": x + 0.01 * rng.standard_normal(100),
                        "c": np.r_[x[:10], [np.nan] * 90]})   # only 10 days overlap
    labels = cluster_trials(pnl, max_corr=0.9, min_overlap=20)
    assert labels["a"] == labels["b"] and labels["c"] != labels["a"]
    assert cluster_trials(pnl.iloc[:, :0]).empty


def test_empty_registry(tmp_path):
    reg = Registry(tmp_path / "missing.jsonl")
    assert reg.n_trials() == 0 and reg.effective_n() == 0
    assert reg.load().empty and "trial_id" in reg.load().columns
    assert reg.pnl_matrix().shape == (0, 0)


def test_corrupt_line_is_skipped_loudly(tmp_path):
    path = tmp_path / "t.jsonl"
    reg = Registry(path)
    reg.log({"i": 0}, {}, _pnl([1.0, 2.0]))
    with open(path, "a") as f:
        f.write('{"trial_id": "trunc')                    # e.g. a writer killed mid-line
    with pytest.warns(RuntimeWarning, match="unparseable"):
        assert reg.n_trials() == 1


def test_git_state_reports_unknown_when_git_is_unavailable(tmp_path, monkeypatch):
    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")
    monkeypatch.setattr(registry_mod.subprocess, "run", no_git)
    assert git_state(ROOT) == {"git_sha": "unknown", "git_dirty": "unknown"}
    tid = Registry(tmp_path / "t.jsonl").log({}, {}, _pnl([1.0, 2.0]))
    assert Registry(tmp_path / "t.jsonl").get(tid)["git_sha"] == "unknown"


def test_git_state_outside_a_repository(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    assert git_state(tmp_path) == {"git_sha": "unknown", "git_dirty": "unknown"}


def test_git_state_matches_head_in_this_repository():
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not running inside a git checkout")
    state = git_state(ROOT)
    assert state["git_sha"] == head and isinstance(state["git_dirty"], bool)


_WRITER = textwrap.dedent("""
    import sys
    import numpy as np, pandas as pd
    from markout.evaluation.registry import Registry
    path, proc = sys.argv[1], int(sys.argv[2])
    reg = Registry(path)
    for i in range(10):
        value = float(proc * 100 + i)
        pnl = pd.Series(np.full(5000, value), index=pd.RangeIndex(5000))  # ~100 KB per line
        reg.log({"proc": proc, "i": i}, {"value": value}, pnl)
""")


def test_concurrent_writers_never_interleave(tmp_path):
    """4 processes append 10 records of ~100 KB each at the same time. Each line is far
    larger than PIPE_BUF and Python's write buffer, so interleaving would corrupt
    records. All 40 must parse and carry their own data intact."""
    path = tmp_path / "trials.jsonl"
    procs = [subprocess.Popen([sys.executable, "-c", _WRITER, str(path), str(p)],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE) for p in range(4)]
    for p in procs:
        _, err = p.communicate(timeout=120)
        assert p.returncode == 0, err.decode()
    recs = Registry(path).records()
    assert len(recs) == 40 and len({r["trial_id"] for r in recs}) == 40
    for r in recs:
        expected = float(r["config"]["proc"] * 100 + r["config"]["i"])
        assert len(r["pnl_values"]) == 5000 and set(r["pnl_values"]) == {expected}
