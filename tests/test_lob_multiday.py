"""The many-days study runs end to end on converted ITCH days (two synthetic ones here)."""

import gzip

import numpy as np
import pytest
from test_lob_itch import random_feed

from markout.lob import itch, lobster, multiday


@pytest.fixture
def two_days(tmp_path, monkeypatch):
    monkeypatch.setattr(lobster, "ITCH_DIR", tmp_path)
    for name, seed in (("01302019.NASDAQ_ITCH50.gz", 1), ("03272019.NASDAQ_ITCH50.gz", 2)):
        src = tmp_path / name
        src.write_bytes(gzip.compress(bytes(random_feed(seed, n=6000).out)))
        itch.convert(src, ["TEST"], out_root=tmp_path)
    return lobster.itch_days()


def test_the_study_runs_and_its_numbers_add_up(two_days):
    assert two_days == ["2019-01-30", "2019-03-27"]
    res = multiday.run(two_days, ("TEST",))
    assert len(res["cells"]) == 2 and res["days"]["2019-01-30"]["unmatched"] == 0
    for c in res["cells"]:
        assert c["gap_bps"] == pytest.approx(c["mk10_bps"]["touch"] - c["mk10_bps"]["fifo"], nan_ok=True)
        assert c["p_fill"]["touch"] >= c["p_fill"]["fifo"] >= c["p_fill"]["through"]  # the models nest
        assert 0 <= c["l2_agrees"] <= 1
        assert c["l2_agrees"] + c["l2_fills_fifo_doesnt"] + c["fifo_fills_l2_doesnt"] == pytest.approx(1)
        assert [r["latency_s"] for r in c["latency"]] == list(multiday.LATENCIES)
        assert c["engine"]["messages_replayed"] > 0
    oos = res["oos"]["TEST"]["fifo"]
    assert [p["day"] for p in oos["per_day"]] == two_days  # each day scored once, chosen on the other
    assert np.isfinite(oos["oos"]["mean"])
    assert "07 · Many days" in multiday.write(res) and "08 · How wrong" in multiday.write_l2(res)
