"""Filesystem locations. Everything is relative to the repository root."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
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
