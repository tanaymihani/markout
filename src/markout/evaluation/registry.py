"""Append-only experiment registry: every backtest run, logged automatically.

Why this exists
---------------
The Deflated Sharpe Ratio and PBO are only as honest as their inputs: the TRUE number
of trials, and the daily PnL of every one of them, including the failures. Rebuilding
those after the fact always undercounts, so the experiment runner calls `Registry.log`
on every run from day 0 and nothing is written by hand.

Format
------
One JSON object per line (JSONL) in data/registry/trials.jsonl::

    {"schema": 1, "trial_id": "3f9c0a1b2d4e", "timestamp": "2026-10-06T14:03:11.52+00:00",
     "git_sha": "0299a2b...", "git_dirty": true, "config_hash": "<sha256>",
     "config": {...}, "metrics": {...}, "tags": {...},
     "pnl_key_type": "int", "pnl_key_name": "date_id",
     "pnl_keys": [181, 182, ...], "pnl_values": [12.5, -3.1, ...]}

Records are never modified or deleted. Appends take an exclusive advisory lock
(fcntl.flock) and write each record as one line, so concurrent runners (parallel
sweeps) can't interleave partial lines, and readers take a shared lock so they never
see a half-written record. JSON is strict: NaN is stored as null.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import subprocess
import uuid
import warnings
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from markout.evaluation.dsr import sharpe
from markout.paths import REGISTRY, ROOT

try:  # POSIX advisory locks. Without them (Windows), O_APPEND single writes still apply.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

SCHEMA_VERSION = 1
Record = dict[str, Any]
Where = Callable[[Record], bool] | None


# ----------------------------------------------------------------------------------------
# provenance helpers (also used by the holdout guard)
# ----------------------------------------------------------------------------------------

def to_jsonable(o: Any) -> Any:
    """Recursively convert to strict-JSON types. NaN/inf become None.

    Unknown types raise TypeError instead of being str()-ed. A repr such as
    '<object at 0x10f...>' would make the config hash change between identical runs.
    """
    if o is None or isinstance(o, str):
        return o
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, Mapping):
        out = {str(k): to_jsonable(v) for k, v in o.items()}
        if len(out) != len(o):
            raise ValueError("mapping keys collide after conversion to str")
        return out
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, (set, frozenset)):
        items = [to_jsonable(v) for v in o]
        return sorted(items, key=lambda v: json.dumps(v, sort_keys=True))
    if isinstance(o, np.ndarray):
        return to_jsonable(o.tolist())
    if isinstance(o, (pd.Timestamp, dt.datetime, dt.date)):
        return o.isoformat()
    if isinstance(o, np.datetime64):
        return pd.Timestamp(o).isoformat()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"value of type {type(o).__name__} is not JSON-serialisable: {o!r}")


def config_hash(config: Mapping[str, Any]) -> str:
    """sha256 of the canonical JSON of a config: sorted keys, compact separators, UTF-8.

    Key order doesn't matter, but types do (1 and 1.0 hash differently), so write
    configs consistently.
    """
    canonical = json.dumps(to_jsonable(config), sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _git(args: list[str], cwd: Path) -> str | None:
    # GIT_OPTIONAL_LOCKS=0: `git status` must not take the index lock and collide with
    # concurrent git commands.
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
    try:
        out = subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True,
                             text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout


def git_state(cwd: str | Path = ROOT) -> dict[str, Any]:
    """{"git_sha": HEAD sha or "unknown", "git_dirty": bool or "unknown"}.

    "Dirty" includes untracked files: a new, uncommitted module can change results just
    as much as an edited one.
    """
    sha = _git(["rev-parse", "HEAD"], Path(cwd))
    status = _git(["status", "--porcelain"], Path(cwd)) if sha else None
    return {"git_sha": sha.strip() if sha else "unknown",
            "git_dirty": bool(status.strip()) if status is not None else "unknown"}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def append_line(path: Path, obj: Mapping[str, Any]) -> None:
    """Append one strict-JSON line under an exclusive lock, looping over partial writes."""
    data = (json.dumps(to_jsonable(obj), separators=(",", ":"), ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_EX)
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view):]
        os.fsync(fd)
    finally:
        os.close(fd)  # also releases the lock


def read_lines(path: Path) -> list[Record]:
    """Parse a JSONL file under a shared lock. Unparseable lines are skipped LOUDLY."""
    if not path.exists():
        return []
    with open(path, "rb") as f:
        if fcntl is not None:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
        data = f.read()
    out: list[Record] = []
    for i, line in enumerate(data.split(b"\n"), start=1):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            warnings.warn(f"{path}:{i}: unparseable record skipped", RuntimeWarning, stacklevel=3)
    return out


# ----------------------------------------------------------------------------------------
# PnL (de)serialisation
# ----------------------------------------------------------------------------------------

def _serialise_pnl(pnl: pd.Series | Mapping) -> tuple[list, list, str | None, str | None]:
    if isinstance(pnl, Mapping):
        pnl = pd.Series(dict(pnl))
    if not isinstance(pnl, pd.Series):
        raise TypeError(f"daily_pnl must be a pandas Series, got {type(pnl).__name__}")
    if pnl.index.has_duplicates:
        raise ValueError("daily_pnl has duplicate keys; aggregate to one value per day first")
    values = pd.to_numeric(pnl, errors="raise").to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(values).any():
        raise ValueError("daily_pnl contains +/-inf")
    idx = pnl.index
    name = idx.name if isinstance(idx.name, str) else None
    if len(idx) == 0:  # a failed run with no PnL is still a trial: log it
        return [], [], None, name
    if pd.api.types.is_integer_dtype(idx.dtype) or all(
            isinstance(k, (int, np.integer)) and not isinstance(k, (bool, np.bool_)) for k in idx):
        key_type, keys = "int", [int(k) for k in idx]
    elif pd.api.types.is_datetime64_any_dtype(idx.dtype) or all(
            isinstance(k, (dt.date, np.datetime64)) for k in idx):
        key_type, keys = "datetime", [pd.Timestamp(k).isoformat() for k in idx]
    elif all(isinstance(k, str) for k in idx):
        key_type, keys = "str", list(idx)
    else:
        raise TypeError("daily_pnl must be indexed by integer (e.g. date_id), date or string keys")
    return keys, [float(v) if not math.isnan(v) else None for v in values], key_type, name


def _deserialise_pnl(rec: Record) -> pd.Series:
    keys, vals, kind = rec.get("pnl_keys", []), rec.get("pnl_values", []), rec.get("pnl_key_type")
    values = np.array([np.nan if v is None else v for v in vals], dtype=float)
    if kind == "datetime":
        index = pd.DatetimeIndex(pd.to_datetime(keys))
    elif kind == "int":
        index = pd.Index(np.asarray(keys, dtype=np.int64))
    else:
        index = pd.Index(keys, dtype=object)
    return pd.Series(values, index=index, name=rec.get("trial_id"))


def _flatten(d: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}"
        if isinstance(v, Mapping) and v:
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


# ----------------------------------------------------------------------------------------
# clustering of trials into independent "bets"
# ----------------------------------------------------------------------------------------

def cluster_trials(pnl: pd.DataFrame, max_corr: float = 0.9, min_overlap: int = 20) -> pd.Series:
    """Cluster trials whose daily PnL is nearly the same bet. Returns a label per column.

    Hierarchical clustering with distance d_ij = 1 - rho_ij (Pearson correlation of daily
    PnL over the days both trials have), average linkage, cut at distance 1 - max_corr.
    Two groups merge while their AVERAGE cross-correlation is >= max_corr.

    Why: the expected maximum Sharpe behind the DSR, E[max of N], assumes N INDEPENDENT
    trials (Bailey & López de Prado 2014, Appendix A.3). Ten variants that differ only in
    a threshold's third decimal are one bet tried ten times, and treating them as ten
    independent draws overstates how many chances luck had. Their paper suggests
    shrinking N by the average correlation; clustering does the same job, but only merges
    trials that really are near-duplicates.

    It errs conservative in two ways, since a larger N can only lower the DSR:
      * pairs with fewer than `min_overlap` common days, or an undefined correlation,
        are treated as uncorrelated (rho = 0);
      * a strategy and its mirror image (rho = -1) are two clusters, because searching
        over the sign is a real degree of freedom.
    Degenerate trials (fewer than 2 observations, or constant PnL such as "never
    traded") form one extra cluster between them.
    """
    if not -1.0 < max_corr <= 1.0:
        raise ValueError(f"max_corr must be in (-1, 1], got {max_corr}")
    cols = list(pnl.columns)
    labels = pd.Series(0, index=pnl.columns, dtype=int, name="cluster")
    if not cols:
        return labels
    X = pnl.astype(float)
    counts = X.notna().sum()
    spread = X.max() - X.min()
    ok = (counts >= 2) & (spread > 0)
    good = [c for c, g in zip(cols, ok.to_numpy()) if g]
    if len(good) == 1:
        labels.loc[good] = 1
    elif len(good) > 1:
        Xg = X[good]
        if Xg.notna().all().all() and len(Xg) >= min_overlap:
            C = np.corrcoef(Xg.to_numpy(), rowvar=False)
        else:
            C = Xg.corr(min_periods=max(min_overlap, 2)).to_numpy()
        C = np.where(np.isfinite(C), C, 0.0)
        C = np.clip((C + C.T) / 2.0, -1.0, 1.0)
        D = 1.0 - C
        np.fill_diagonal(D, 0.0)
        Z = linkage(squareform(D, checks=False), method="average")
        labels.loc[good] = fcluster(Z, t=1.0 - max_corr + 1e-12, criterion="distance")
    bad = [c for c, g in zip(cols, ok.to_numpy()) if not g]
    if bad:
        labels.loc[bad] = int(labels.max()) + 1
    return labels


# ----------------------------------------------------------------------------------------
# the registry
# ----------------------------------------------------------------------------------------

class Registry:
    """Append-only JSONL log of backtest trials. See the module docstring for the format.

    `where` filters, wherever they appear, receive the full record dict (keys trial_id,
    timestamp, git_sha, git_dirty, config_hash, config, metrics, tags, ...), so you can
    filter on tags or config, e.g. ``where=lambda r: r["tags"].get("stage") == "research"``.
    """

    def __init__(self, path: str | Path = REGISTRY / "trials.jsonl", repo_dir: str | Path = ROOT) -> None:
        self.path = Path(path)
        self.repo_dir = Path(repo_dir)

    # ---- writing ------------------------------------------------------------------------
    def log(self, config: dict, metrics: dict, daily_pnl: pd.Series, tags: dict | None = None) -> str:
        """Append one trial and return its trial_id (12 hex characters of a uuid4).

        Records the UTC timestamp, git SHA and dirty flag ("unknown" if git fails), the
        config hash (sha256 of canonical sorted JSON), the config itself, the metrics,
        optional tags, and the daily PnL as parallel lists of keys and values. Log
        failed and discarded variants too: they are trials.
        """
        if not isinstance(config, Mapping) or not isinstance(metrics, Mapping):
            raise TypeError("config and metrics must be dicts")
        keys, values, key_type, key_name = _serialise_pnl(daily_pnl)
        record = {
            "schema": SCHEMA_VERSION,
            "trial_id": uuid.uuid4().hex[:12],
            "timestamp": utc_now(),
            **git_state(self.repo_dir),
            "config_hash": config_hash(config),
            "config": to_jsonable(config),
            "metrics": to_jsonable(metrics),
            "tags": to_jsonable(tags or {}),
            "pnl_key_type": key_type,
            "pnl_key_name": key_name,
            "pnl_keys": keys,
            "pnl_values": values,
        }
        append_line(self.path, record)
        return record["trial_id"]

    # ---- reading ------------------------------------------------------------------------
    def records(self, where: Where = None) -> list[Record]:
        """All records in log order, optionally filtered by `where(record)`."""
        recs = read_lines(self.path)
        return recs if where is None else [r for r in recs if where(r)]

    def get(self, trial_id: str) -> Record:
        for r in self.records():
            if r.get("trial_id") == trial_id:
                return r
        raise KeyError(trial_id)

    def load(self, where: Where = None) -> pd.DataFrame:
        """One row per trial: provenance columns, n_obs, then flattened config.*,
        metrics.* and tags.* columns (nested dicts become dotted names)."""
        base = ["trial_id", "timestamp", "git_sha", "git_dirty", "config_hash", "n_obs"]
        rows = []
        for r in self.records(where):
            row = {k: r.get(k) for k in base[:-1]}
            row["n_obs"] = len(r.get("pnl_keys", []))
            for part in ("config", "metrics", "tags"):
                row.update(_flatten(r.get(part) or {}, part))
            rows.append(row)
        df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=base)
        if rows:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
            if df["trial_id"].duplicated().any():
                warnings.warn("duplicate trial_ids in registry", RuntimeWarning, stacklevel=2)
        return df

    def pnl_matrix(self, where: Where = None, fill_value: float | None = None) -> pd.DataFrame:
        """T x N matrix of daily PnL. Columns are trial_ids in log order, and the index is
        the sorted union of all trials' keys.

        A day missing from a trial is NaN. Pass fill_value=0.0 when missing means "no
        trade that day" (the right input for PBO and for Sharpe ratios over a common
        calendar). Leave it NaN when it means "not covered", e.g. a subsample run.
        """
        recs = self.records(where)
        ids = [r["trial_id"] for r in recs]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate trial_ids in registry")
        kinds = {r.get("pnl_key_type") for r in recs if r.get("pnl_keys")}
        if len(kinds) > 1:
            raise ValueError(f"trials use incompatible PnL key types {sorted(kinds)}; filter with `where`")
        series = [_deserialise_pnl(r) for r in recs]
        nonempty = [s.index for s in series if len(s)]
        if nonempty:
            index = nonempty[0]
            for other in nonempty[1:]:
                index = index.union(other)
            index = index.sort_values()
        else:
            index = pd.Index([], dtype=np.int64)
        names = {r.get("pnl_key_name") for r in recs if r.get("pnl_keys")}
        index = index.rename(names.pop() if len(names) == 1 else None)
        data = {s.name: s.reindex(index).to_numpy() for s in series}
        df = pd.DataFrame(data, index=index, columns=ids, dtype=float)
        return df if fill_value is None else df.fillna(fill_value)

    def n_trials(self, where: Where = None) -> int:
        """Raw number of logged trials, failures included."""
        return len(self.records(where))

    def trial_sharpes(self, where: Where = None, fill_value: float | None = None) -> pd.Series:
        """Per-period Sharpe of each trial's daily PnL: the `trial_srs` input of DSR."""
        pnl = self.pnl_matrix(where, fill_value)
        return pd.Series({c: sharpe(pnl[c].to_numpy()) for c in pnl.columns},
                         dtype=float, name="sharpe")

    def clusters(self, where: Where = None, max_corr: float = 0.9,
                 fill_value: float | None = None, min_overlap: int = 20) -> pd.Series:
        """Cluster label per trial_id (see `cluster_trials`)."""
        return cluster_trials(self.pnl_matrix(where, fill_value), max_corr, min_overlap)

    def effective_n(self, where: Where = None, max_corr: float = 0.9,
                    fill_value: float | None = None, min_overlap: int = 20) -> int:
        """Estimated number of INDEPENDENT trials, i.e. the number of PnL clusters.

        This is the N to pass to `deflated_sharpe`. DSR's E[max SR] assumes independent
        trials, and near-duplicate variants (rho >= max_corr) shouldn't count twice.
        See `cluster_trials` for the method and why it errs conservative.
        """
        labels = self.clusters(where, max_corr, fill_value, min_overlap)
        return int(labels.nunique())
