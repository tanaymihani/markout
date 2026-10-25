"""USAGE.md stays true: its Python blocks run top to bottom in one namespace."""

import re

from markout.paths import ROOT


def test_usage_examples_run(tmp_path, monkeypatch, capsys):
    blocks = re.findall(r"```python\n(.*?)```", (ROOT / "USAGE.md").read_text(), flags=re.S)
    assert len(blocks) >= 8
    monkeypatch.chdir(tmp_path)  # the registry and holdout examples write files
    ns: dict = {}
    for i, code in enumerate(blocks):
        exec(compile(code, f"USAGE.md block {i + 1}", "exec"), ns)
    assert "refused" in capsys.readouterr().out
