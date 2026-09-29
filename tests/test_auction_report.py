"""End-to-end: the full A-C pipeline on synthetic data (dev_* outputs, never committed)."""

import json

import pytest

from markout.auction import report
from markout.paths import REPORTS, RESULTS


@pytest.mark.slow
def test_synthetic_report_end_to_end():
    report.main(["--synthetic"])
    text = (REPORTS / "dev_01_auction.md").read_text()
    for heading in ("## 1. Data audit", "## 2. Forecasts", "## 3. From forecasts to trades",
                    "## 4. Honest evaluation", "## 5. Holdout", "## 6. What didn't work", "## 7. Limitations"):
        assert heading in text
    assert "Synthetic data" in text
    res = json.loads((RESULTS / "dev_auction.json").read_text())
    assert res["synthetic"] is True
    assert res["holdout"]["accessed"] == 1
    assert res["honest"]["n_trials"] == len(res["trials"]) > 0
    # the planted signal is found and survives costs in this easy synthetic world
    assert res["forecasts"]["ridge"]["ic_spearman_mean"] > 0.1
