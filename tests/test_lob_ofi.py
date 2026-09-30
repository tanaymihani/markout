"""Order-flow imbalance (Cont, Kukanov & Stoikov 2014) on hand-built sequences."""

from __future__ import annotations

import numpy as np

from markout.lob.imbalance import next_mid_change, queue_imbalance
from markout.lob.lobster import Book, Day, Messages
from markout.lob.ofi import beta_depth_slope, make_buckets, ofi_events, ols, oos_r2


def test_ofi_events_on_a_hand_built_quote_sequence():
    #            PB      qB   PA      qA
    states = [(10000, 10, 10100, 10),
              (10000, 15, 10100, 10),   # bid queue +5              -> +5
              (10000, 15, 10100, 7),    # ask queue -3              -> +3
              (10050, 4, 10100, 7),     # bid improves, new size 4  -> +4
              (10050, 4, 10080, 6),     # ask improves, new size 6  -> -6
              (10000, 15, 10080, 6),    # bid retreats, old size 4  -> -4
              (10000, 15, 10100, 7),    # ask retreats, old size 6  -> +6
              (10000, 15, 10100, 7)]    # nothing changes           ->  0
    pb, qb, pa, qa = (np.array(c) for c in zip(*states))
    assert ofi_events(pb, qb, pa, qa).tolist() == [0, 5, 3, 4, -6, -4, 6, 0]


def _day(time, pb, qb, pa, qa) -> Day:
    n = len(time)
    col = lambda v: np.asarray(v, dtype=np.int64)[:, None]  # noqa: E731
    msgs = Messages(np.asarray(time, float), np.ones(n, np.int8), np.arange(n, dtype=np.int64),
                    np.ones(n, np.int64), col(pb)[:, 0], np.ones(n, np.int8))
    return Day("TEST", msgs, Book(col(pa), col(qa), col(pb), col(qb)))


def test_buckets_sum_events_between_edges_and_take_the_mid_change():
    # states at t = 0, 5, 12, 15, 25 ; buckets (0,10], (10,20], (20,30]
    day = _day([0, 5, 12, 15, 25],
               pb=[10000, 10000, 10100, 10100, 10000],
               qb=[10, 12, 3, 3, 9],
               pa=[10200, 10200, 10200, 10300, 10300],
               qa=[5, 5, 5, 8, 8])
    bk = make_buckets(day, width=10.0, session=(0.0, 30.0))
    assert bk.ofi.tolist() == [2.0, 3.0 + 5.0, -3.0]     # +2 | +3 (bid up), +5 (ask up) | -3
    # mids (ticks): 101 -> 101 | -> 101.5 -> 102 | -> 101.5
    assert np.allclose(bk.dmid, [0.0, 1.0, -0.5])
    # depth (qB+qA)/2 is 7.5 on [0,5) then 8.5 on [5,10)
    assert np.isclose(bk.depth[0], 8.0)


def test_ols_and_oos_r2():
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    y = 2.0 + 0.5 * x
    a, beta, r2 = ols(x, y)
    assert np.isclose(a, 2.0) and np.isclose(beta, 0.5) and np.isclose(r2, 1.0)
    noise = rng.normal(size=500)
    train = np.arange(500) < 250
    res = oos_r2(x, noise, train, ~train)       # unrelated series: no out-of-sample skill
    assert res.r2_oos < 0.02


def test_beta_depth_slope_recovers_minus_one():
    rng = np.random.default_rng(1)
    depth = np.exp(rng.uniform(4, 10, 60))
    beta = 3.0 / depth * np.exp(rng.normal(0, 0.05, 60))
    s = beta_depth_slope(beta, depth)
    assert abs(s.slope + 1) < 0.05 and s.ci[0] < -1 < s.ci[1]


def test_queue_imbalance_and_next_mid_change():
    assert queue_imbalance(np.array([300, 0, 0]), np.array([100, 0, 50])).tolist() == [0.5, 0.0, -1.0]
    mid = np.array([10.0, 10.0, 10.5, 10.5, 10.5, 10.0])
    assert next_mid_change(mid, np.array([0, 1, 2, 4, 5])).tolist() == [2, 2, 5, 5, 6]
