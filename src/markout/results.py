"""Machine-readable results: every number quoted in a report is loaded from here."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from markout.paths import RESULTS


def _clean(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        # 12 significant digits: multithreaded sums differ in the last bits from run to run,
        # and results files are committed, so keep them stable
        return float(f"{f:.12g}") if math.isfinite(f) else None
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    return o


def save(name: str, obj: dict) -> Path:
    """Write reports/results/<name>.json (NaN/inf become null)."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(_clean(obj), indent=2) + "\n")
    return path


def load(name: str) -> dict | None:
    path = RESULTS / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None
