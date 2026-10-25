"""The post-or-cross rule: chosen on the morning, scored on the afternoon, never on both."""

import numpy as np
import pandas as pd
import pytest

from markout.lob import postcross as pc
from markout.lob.lobster import Day
from markout.lob.synthetic import random_stream


def cells(rows):
    """Decisions from (t, queue_bucket, signal_bucket, ev_post, ev_cross) tuples."""
    df = pd.DataFrame(rows, columns=["t", "queue_bucket", "signal_bucket", "ev_post", "ev_cross"])
    return df.assign(model="fifo", tick_bps=1.0, fill_frac=1.0, diff=df["ev_post"] - df["ev_cross"])


def test_choose_takes_the_better_action_per_cell_and_skips_thin_or_losing_cells():
    rows = [(0, 0, 0, 0.5, 0.1)] * 30  # post wins
    rows += [(0, 1, 0, -0.2, 0.3)] * 30  # cross wins
    rows += [(0, 2, 0, -0.4, -0.1)] * 30  # both lose: stay out
    rows += [(0, 3, 4, 9.0, 0.0)] * 5  # too few decisions to trust
    rule = pc.choose(cells(rows))
    assert rule == {(0, 0): "post", (1, 0): "cross", (2, 0): "none", (3, 4): "none"}


def test_apply_pays_the_chosen_action_and_zero_otherwise():
    df = cells([(0, 0, 0, 0.5, 0.1), (0, 1, 0, -0.2, 0.3), (0, 3, 3, 1.0, 1.0)])
    got = pc.apply({(0, 0): "post", (1, 0): "cross"}, df)
    assert got.tolist() == pytest.approx([0.5, 0.3, 0.0])


def test_the_rule_is_scored_out_of_sample_only():
    # morning: posting looks great in cell (0, 0); afternoon: it loses there
    morning = [(36_000 + i, 0, 0, 1.0, 0.0) for i in range(40)]
    afternoon = [(50_000 + i, 0, 0, -1.0, 0.2) for i in range(40)]
    r = pc.evaluate_rule(cells(morning + afternoon), "fifo", split=45_900, horizon=30, fee_ticks=0.3)
    assert r.in_sample == pytest.approx(1.0)
    assert r.out_of_sample == pytest.approx(-1.0)  # the morning's choice, paid at afternoon prices
    assert r.share_post == 1.0 and r.share_cross == 0.0 and r.n_test == 40
    assert r.always_cross == pytest.approx(0.2)  # what hindsight would have picked
    assert r.net_of_fee == pytest.approx(r.out_of_sample)  # posts pay no taker fee


def test_crosses_pay_the_fee():
    rows = [(36_000 + i, 1, 2, -0.5, 0.6) for i in range(30)] + [(50_000 + i, 1, 2, -0.5, 0.6) for i in range(30)]
    r = pc.evaluate_rule(cells(rows), "fifo", split=45_900, horizon=30, fee_ticks=0.3)
    assert r.share_cross == 1.0 and r.out_of_sample == pytest.approx(0.6)
    assert r.net_of_fee == pytest.approx(0.3) and r.per_trade == pytest.approx(0.6)


def test_block_bootstrap_ci_covers_the_mean_and_widens_with_dependence():
    rng = np.random.default_rng(0)
    t = np.arange(6000) * 5.0
    iid = rng.normal(0.1, 1, t.size)
    lo, hi = pc.block_bootstrap_ci(t, iid)
    assert lo < iid.mean() < hi
    sticky = np.repeat(rng.normal(0.1, 1, t.size // 60), 60)  # one value per 300 s block
    lo2, hi2 = pc.block_bootstrap_ci(t, sticky)
    assert hi2 - lo2 > 3 * (hi - lo)
    assert np.isnan(pc.block_bootstrap_ci(np.array([]), np.array([]))[0])


def test_decisions_on_a_valid_stream():
    m, b = random_stream(np.random.default_rng(3), n_events=4000, levels=5)
    day = Day("TEST", m, b)
    df = pc.decisions(day, horizon=10.0, every=2.0, session=(float(m.time[0]), float(m.time[-1])))
    assert set(df["model"]) == {"touch", "fifo", "through"}
    n = df.groupby("model").size()
    assert n.nunique() == 1 and n.iloc[0] > 50  # the same decisions under every model
    assert (df["imbalance"] != 0).all() and df["side"].isin([-1, 1]).all()
    unfilled = df["fill_frac"] == 0
    assert (df.loc[unfilled, "ev_post"] == 0).all()  # an unfilled post earns nothing
    assert np.allclose(df["diff"], df["ev_post"] - df["ev_cross"])
    by = df.pivot_table(index="t", columns="model", values="fill_frac")
    assert (by["touch"] >= by["fifo"] - 1e-12).all() and (by["fifo"] >= by["through"] - 1e-12).all()
    g = pc.grid(df[df["model"] == "fifo"], min_n=10**9)
    assert np.isnan(g).all()  # nothing passes an impossible minimum
