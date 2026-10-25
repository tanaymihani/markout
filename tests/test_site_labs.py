"""The browser labs (docs/js/) must agree with the Python code and behave like the model says."""

import json
import re
import shutil
import subprocess

import numpy as np
import pytest

from markout import results
from markout.paths import ROOT
from markout.vol import strategy as S

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


def embedded_data() -> dict:
    page = (ROOT / "docs" / "index.html").read_text()
    m = re.search(r'<script id="lab-data" type="application/json">(.*?)</script>', page, re.S)
    assert m, "docs/index.html has no embedded lab data; run python -m markout.site"
    return json.loads(m.group(1))


@pytest.fixture(scope="module")
def labs(tmp_path_factory):
    d = embedded_data()
    payload = {"pnl": d["vol"]["pnl"], "v": d["vol"]["kelly"]["full"] / 100,
               "gross": d["auction"]["gross"], "cost": d["auction"]["cost"]}
    path = tmp_path_factory.mktemp("labs") / "input.json"
    path.write_text(json.dumps(payload))
    r = subprocess.run(["node", str(ROOT / "tests/js/check_labs.js"), str(ROOT / "docs"), str(path)],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout), d, payload


def test_simulator_is_deterministic_for_a_seed(labs):
    out, _, _ = labs
    assert out["defaults"] == out["again"]


def test_simulator_shows_adverse_selection_only_with_long_queues_and_informed_flow(labs):
    out, _, _ = labs
    touch_p, touch_mk = out["defaults"]["touch"]
    fifo_p, fifo_mk = out["defaults"]["fifo"]
    assert touch_p > fifo_p and touch_mk > 0 > fifo_mk                 # the default, INTC-like case
    assert out["long_queue"]["fifo"][1] < fifo_mk                      # waiting longer makes it worse
    assert out["short_queue"]["fifo"][0] > fifo_p                      # short queues fill more often
    assert out["short_queue"]["fifo"][1] > fifo_mk
    assert min(out["no_informed"]["touch"][1], out["no_informed"]["fifo"][1]) > 0.4  # nobody informed: no adverse selection


def test_simulator_defaults_roughly_reproduce_intc():
    m = results.load("microstructure")
    if not m:
        pytest.skip("no microstructure results")
    intc = {r["model"]: r for r in m["fills"]["INTC"]["summary"]}
    r = subprocess.run(["node", "-e", f"""
        eval(require('fs').readFileSync('{ROOT / "docs/js/queue-sim.js"}', 'utf8'));
        const r = globalThis.MarkoutQueue.runMany({{}}, 4000, 3);
        console.log(JSON.stringify([r.touch.pFill, r.touch.meanMarkout, r.fifo.pFill, r.fifo.meanMarkout]));"""],
        capture_output=True, text=True, timeout=120)
    tp, tm, fp, fm = json.loads(r.stdout)
    assert abs(tp - intc["touch"]["p_fill"]) < 0.06 and abs(fp - intc["fifo"]["p_fill"]) < 0.06
    assert abs(tm - intc["touch"]["mk_10"]) < 0.08 and abs(fm - intc["fifo"]["mk_10"]) < 0.08


def test_kelly_and_wealth_match_python(labs):
    out, d, payload = labs
    py = S.kelly_exact(np.array(payload["pnl"]))
    assert out["kelly"]["vStar"] == pytest.approx(py["v_star"], rel=1e-9)
    assert out["kelly"]["vMax"] == pytest.approx(py["v_max"], rel=1e-9)
    assert out["kelly"]["vContinuous"] == pytest.approx(py["v_continuous"], rel=1e-9)
    assert out["wealth_last"] == pytest.approx(S.wealth_path(np.array(payload["pnl"]), payload["v"])[-1], rel=1e-9)
    assert 100 * py["v_star"] == pytest.approx(d["vol"]["kelly"]["full"], abs=0.01)  # same series as the report


def test_cost_slider_reproduces_the_report(labs):
    out, _, _ = labs
    a = results.load("auction")
    if not a:
        pytest.skip("no auction results")
    rows = {r["cost_mult"]: r for r in a["robustness"]["cost_rows"]}
    assert out["cost"][0] == pytest.approx(a["selected"]["summary"]["sharpe_annualized"], abs=2e-3)
    assert out["cost"][1] == pytest.approx(rows[2.0]["unadapted"]["sharpe_annualized"], abs=2e-3)
