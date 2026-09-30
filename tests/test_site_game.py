"""The browser game (docs/game.js) must behave exactly like the Python engine."""

import json
import shutil
import subprocess

import numpy as np
import pytest

from markout.games import cardgame as C
from markout.paths import ROOT

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


def test_js_port_matches_python(tmp_path):
    cfg, rng, cases = C.Config(), np.random.default_rng(11), []
    for k in range(4):
        d = C.deal(cfg, rng)
        g = C.Game(cfg, d)
        quotes = []
        for _ in range(3):
            m = C.prior_mean(cfg, g.revealed) + (1.5 if k % 2 else -1.5)
            q = [round(m - 2, 1), round(m + 2, 1)]
            g.play_round(*q)
            quotes.append(q)
        post = C.posterior_mean(cfg, g.revealed, g.events, n_samples=300_000, rng=np.random.default_rng(1))
        cases.append({"cards": list(d.cards), "quotes": quotes, "actions": [e.action for e in g.events],
                      "bots": [[{"informed": b.informed, "slotU": b.slot_u, "side": b.side} for b in row] for row in d.bots],
                      "post": post["mean"]})
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(cases))
    r = subprocess.run(["node", str(ROOT / "tests/js/check_game.js"), str(ROOT / "docs/game.js"), str(path)],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
