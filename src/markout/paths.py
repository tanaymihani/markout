"""Filesystem locations. Everything is relative to the project root.

In a clone of the repository (and in the in-browser desk's bundle, which keeps the same src/ layout) the root
is the repository itself. An installed package has no repository, so its root is $MARKOUT_HOME if that is
set, and the current directory otherwise: `markout status` and the ITCH converter then use ./data and
./reports there.
"""

import os
from pathlib import Path

_HERE = Path(__file__).resolve()
if os.environ.get("MARKOUT_HOME"):
    ROOT = Path(os.environ["MARKOUT_HOME"]).expanduser().resolve()
elif _HERE.parent.parent.name == "src":  # <root>/src/markout/paths.py
    ROOT = _HERE.parents[2]
else:
    ROOT = Path.cwd()
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
REGISTRY = DATA / "registry"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
RESULTS = REPORTS / "results"


def ensure_dirs() -> None:
    for p in (RAW, PROCESSED, REGISTRY, FIGURES, RESULTS):
        p.mkdir(parents=True, exist_ok=True)
