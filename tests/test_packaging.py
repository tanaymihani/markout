"""The slim package: the toolkit imports without the studies' dependencies, and finds its root outside a clone."""

import os
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLKIT = (
    "markout.cli",
    "markout.evaluation",
    "markout.lob.lobster",
    "markout.lob.itch",
    "markout.lob.fills",
    "markout.lob.markouts",
    "markout.results",
)
RESEARCH = ("sklearn", "lightgbm", "matplotlib", "yaml", "tqdm")


def _run(code: str, **env: str) -> str:
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, env={**os.environ, **env}
    )
    return out.stdout.strip()


def test_toolkit_imports_without_the_research_dependencies():
    # as if they were not installed: any import of them fails (arch uses matplotlib only when it is there)
    code = (
        "import importlib, importlib.abc, sys\n"
        "class Missing(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        f"        if name.split('.')[0] in {RESEARCH!r}: raise ImportError(name)\n"
        "sys.meta_path.insert(0, Missing())\n"
        f"for m in {TOOLKIT!r}: importlib.import_module(m)\n"
        "print('ok')"
    )
    assert _run(code) == "ok"


def test_core_dependencies_cover_the_toolkit_and_research_holds_the_rest():
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    core = {d.split(">")[0].split("=")[0].strip() for d in meta["dependencies"]}
    research = {d.split(">")[0].split("=")[0].strip() for d in meta["optional-dependencies"]["research"]}
    assert core == {"numpy", "pandas", "polars", "pyarrow", "scipy", "arch"}
    assert {"scikit-learn", "lightgbm", "matplotlib"} <= research
    assert not core & research


def test_markout_home_moves_the_data_and_results(tmp_path):
    code = "from markout import paths; print(paths.ROOT); print(paths.RESULTS)"
    root, res = _run(code, MARKOUT_HOME=str(tmp_path)).splitlines()
    assert Path(root) == tmp_path.resolve()
    assert Path(res) == tmp_path.resolve() / "reports" / "results"


def test_a_clone_is_its_own_root():
    env = {k: v for k, v in os.environ.items() if k != "MARKOUT_HOME"}
    out = subprocess.run(
        [sys.executable, "-c", "from markout import paths; print(paths.ROOT)"],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    assert Path(out.stdout.strip()) == ROOT
