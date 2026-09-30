"""LOBSTER helpers and the message/book consistency check on synthetic data."""

from __future__ import annotations

import numpy as np

from markout.lob.lobster import (
    DUMMY_BID,
    Book,
    check_consistency,
    in_view,
    state_durations,
    state_index,
    time_average,
)
from markout.lob.synthetic import SyntheticExchange, random_stream

P, TICK = 1_000_000, 100


def test_consistency_passes_on_a_valid_stream():
    m, b = random_stream(np.random.default_rng(0), n_events=3000)
    c = check_consistency(m, b)
    assert c.n_checked > 1000
    assert c.pass_rate == 1.0 and len(c.failures) == 0
    assert set(c.by_type) == {1, 2, 3, 4}


def test_consistency_flags_a_corrupted_book_row():
    ex = SyntheticExchange()
    ex.submit(0.0, -1, P + TICK, 300)
    ex.submit(1.0, 1, P, 200)
    a = ex.submit(2.0, 1, P, 100)
    ex.cancel(3.0, a, 40)
    m, b = ex.build()
    bad = b.bid_size.copy()
    bad[3, 0] += 1                     # after the cancel the level should hold 260
    c = check_consistency(m, Book(b.ask_price, b.ask_size, b.bid_price, bad))
    assert c.n_pass == c.n_checked - 1
    assert list(c.failures) == [3]


def test_in_view_uses_the_deepest_visible_level():
    prices = np.array([[P, P - TICK, P - 2 * TICK],        # three occupied bid levels
                       [P, P - TICK, DUMMY_BID]])          # only two: everything below is empty
    side = np.array([1, 1])
    assert list(in_view(prices, side, np.array([P - 3 * TICK] * 2))) == [False, True]
    assert list(in_view(prices, side, np.array([P + TICK] * 2))) == [True, True]


def test_state_index_and_durations():
    time = np.array([10.0, 11.0, 11.0, 13.5])
    assert list(state_index(time, np.array([9.0, 10.0, 11.0, 12.0, 20.0]))) == [-1, 0, 2, 2, 3]
    w = state_durations(time, start=10.5, end=14.0)
    assert np.allclose(w, [0.5, 0.0, 2.5, 0.5])


def test_time_average_matches_a_hand_computation():
    time = np.array([0.0, 1.0, 3.0])
    v = np.array([10.0, 20.0, 40.0])   # 10 on [0,1), 20 on [1,3), 40 from 3 on
    got = time_average(time, v, np.array([0.0, 0.5, 2.0]), np.array([3.0, 1.5, 4.0]))
    assert np.allclose(got, [(10 + 40) / 3, 15.0, 30.0])
