"""The desk works with exactly the files the Dockerfile copies (no Docker needed).

A file the code reads but the image lacks (the journal template was one) fails here,
before the container job in CI or a deploy would.
"""

import re
import shutil
import subprocess
import sys

from markout.paths import ROOT

PROBE = """
import tempfile
from pathlib import Path
import markout
from markout.cup.public import Sessions
assert markout.__file__.startswith({app!r}), markout.__file__
s = Sessions(Path(tempfile.mkdtemp())).create()
assert s.bot.store.proposals("pending"), "a new sandbox should start with proposals"
s.clock.advance(__import__("datetime").timedelta(hours=200))
s.bot.refresh()
print("ok")
"""


def copies() -> list[tuple[str, str]]:
    lines = (ROOT / "Dockerfile").read_text().splitlines()
    out = []
    for ln in lines:
        m = re.match(r"COPY\s+(?:--chown=\S+\s+)?(\S+)\s+(\S+)$", ln.strip())
        if m and not m.group(1).startswith("deploy/"):  # requirements are installed, not imported
            out.append((m.group(1), m.group(2)))
    return out


def test_the_image_layout_is_enough_to_run_the_desk(tmp_path):
    app = tmp_path / "app"
    pairs = copies()
    assert ("src/markout", "./src/markout") in pairs
    for src, dest in pairs:
        target = app / dest.removeprefix("./")
        if (ROOT / src).is_dir():
            shutil.copytree(ROOT / src, target, ignore=shutil.ignore_patterns("__pycache__", "*.so"))
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / src, target)
    r = subprocess.run(
        [sys.executable, "-c", PROBE.format(app=str(app))],
        cwd=tmp_path,
        env={"PYTHONPATH": str(app / "src"), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr[-2000:]
