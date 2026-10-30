"""The replications table: the Deflated Sharpe row is recomputed exactly, and the table renders from saved results."""

import numpy as np
import pytest

from markout import replications as rp


def test_dsr_row_matches_the_printed_example():
    here, paper = rp.dsr_row(), rp.PUBLISHED["dsr"]
    for k in ("sr0", "dsr_100", "dsr_46", "dsr_88_normal"):
        assert round(here[k], 4) == paper[k]


def test_newey_west_t_reduces_to_the_plain_t_without_lags():
    x = np.random.default_rng(3).normal(0.1, 1.0, size=400)
    plain = x.mean() / (x.std(ddof=0) / np.sqrt(len(x)))
    assert rp.newey_west_t(x, lag=0) == pytest.approx(plain, rel=1e-12)


def test_newey_west_t_shrinks_for_persistent_series():
    rng = np.random.default_rng(4)
    e = rng.normal(size=2000)
    x = np.empty_like(e)
    x[0] = e[0]
    for i in range(1, len(e)):  # AR(1) with rho = 0.9
        x[i] = 0.9 * x[i - 1] + e[i]
    x += 0.2
    assert abs(rp.newey_west_t(x, lag=30)) < abs(rp.newey_west_t(x, lag=0))


def test_table_renders_one_row_per_available_result():
    res = rp.compute()
    text = rp.table(res)
    assert text.startswith("| result | published | here | setup |")
    n_rows = text.count("\n") - 2
    expected = 1 + 2 * ("cks" in res["here"]) + ("gb" in res["here"]) + ("optiver" in res["here"])
    expected += "carr_wu" in res["here"]
    assert n_rows == expected


def test_table_is_empty_without_results():
    assert rp.table({"published": rp.PUBLISHED, "here": {}}) == ""
