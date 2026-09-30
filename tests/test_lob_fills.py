"""Fill models on hand-built LOBSTER-format scenarios with known answers.

Prices are in LOBSTER units (dollars x 10^4): P = $100.00, one tick = 100.
Each scenario seeds an ask at P + 1 tick and a bid at P - 1 tick, then builds the
queue at P that our hypothetical order joins.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from markout.lob.fills import MODELS, End, Fills, Orders, simulate
from markout.lob.lobster import check_consistency
from markout.lob.synthetic import SyntheticExchange, random_stream

P, TICK = 1_000_000, 100
BUY, SELL = 1, -1


def base(ask_ticks: int = 1) -> SyntheticExchange:
    ex = SyntheticExchange()
    ex.submit(0.0, SELL, P + ask_ticks * TICK, 500)
    ex.submit(0.0, BUY, P - TICK, 500)
    return ex


def here(ex: SyntheticExchange) -> int:
    """Placement index: right after the last message written so far."""
    return len(ex.rows) - 1


def run(ex: SyntheticExchange, place: int, side: int, price: int, max_wait: float = 60.0,
        size: int = 100) -> dict[str, Fills]:
    m, b = ex.build()
    assert check_consistency(m, b).pass_rate == 1.0          # the scenario itself is valid
    orders = Orders.make([place], side, price, size, max_wait)
    return {model: simulate(m, b, orders, model, engine="python") for model in MODELS}


def one(f: Fills) -> dict:
    return {k: getattr(f, k)[0].item() for k in Fills.FIELDS}


def far_message(ex: SyntheticExchange, t: float) -> None:
    """A message that touches nothing near P (a new ask level far away)."""
    ex.submit(t, SELL, P + 50 * TICK, 100)


def test_queue_ahead_cancelled_then_execution_fills_fifo():
    ex = base()
    a = ex.submit(1.0, BUY, P, 200)
    place = here(ex)                   # we join behind A's 200 shares
    ex.cancel(2.0, a, 150)             # ahead: 200 -> 50
    ex.delete(3.0, a)                  # ahead: 50 -> 0
    ex.submit(4.0, BUY, P, 300)        # B joins behind us
    ex.market(5.0, SELL, 100)          # executes B: in FIFO it would have hit us first
    far_message(ex, 100.0)
    r = run(ex, place, BUY, P)
    fifo = one(r["fifo"])
    assert fifo["queue_ahead"] == 200
    assert (fifo["filled_qty"], fifo["fill_time"], fifo["end_reason"]) == (100, 5.0, End.FILLED)
    assert fifo["fill_price"] == P
    assert one(r["touch"])["fill_time"] == 5.0                 # first print at P
    through = one(r["through"])                               # nothing traded beyond P
    assert (through["filled_qty"], through["end_reason"]) == (0, End.TIMEOUT)
    assert through["end_time"] == 61.0


def test_executions_consume_queue_ahead_before_filling_us():
    ex = base()
    ex.submit(1.0, BUY, P, 300)
    place = here(ex)                   # 300 ahead
    ex.submit(2.0, BUY, P, 100)        # B, behind us
    ex.market(3.0, SELL, 250)          # A 250: ahead 300 -> 50
    ex.market(4.0, SELL, 100)          # A 50 (ahead -> 0), then B 50: we get 50
    ex.market(5.0, SELL, 20)           # B 20: we get 20 more
    far_message(ex, 100.0)
    r = run(ex, place, BUY, P)
    fifo = one(r["fifo"])
    assert fifo["filled_qty"] == 70
    assert fifo["fill_time"] == 5.0
    assert fifo["end_reason"] == End.TIMEOUT                  # partial fill, then expired
    assert one(r["touch"])["fill_time"] == 3.0
    assert one(r["through"])["filled_qty"] == 0


def test_cancels_behind_us_do_not_shrink_queue_ahead():
    ex = base()
    ex.submit(1.0, BUY, P, 100)
    place = here(ex)
    b = ex.submit(2.0, BUY, P, 500)    # behind us
    ex.cancel(3.0, b, 300)
    ex.delete(4.0, b)
    ex.market(5.0, SELL, 100)          # executes A, exactly our queue ahead
    far_message(ex, 100.0)
    fifo = one(run(ex, place, BUY, P)["fifo"])
    assert fifo["filled_qty"] == 0     # counting B's cancels would have filled us here
    assert fifo["end_reason"] == End.TIMEOUT


def test_hidden_executions_are_ignored_by_fifo_and_through():
    ex = base()
    ex.submit(1.0, BUY, P, 100)
    place = here(ex)
    ex.hidden(2.0, P, 500, BUY)        # a hidden buy order at P was hit
    far_message(ex, 100.0)
    r = run(ex, place, BUY, P)
    assert one(r["fifo"])["filled_qty"] == 0
    assert one(r["through"])["filled_qty"] == 0
    touch = one(r["touch"])            # the optimist counts any print at our price
    assert (touch["filled_qty"], touch["fill_time"]) == (100, 2.0)


def test_timeout_is_strictly_after_the_deadline():
    ex = base()
    ex.submit(1.0, BUY, P, 100)
    place = here(ex)                   # deadline = 1.0 + 10.0
    ex.submit(11.0, SELL, P + 3 * TICK, 100)   # exactly at the deadline: still alive
    at_deadline = here(ex)
    ex.market(12.0, SELL, 600)         # would fill everyone, but we are gone
    r = run(ex, place, BUY, P, max_wait=10.0)
    for model in MODELS:
        f = one(r[model])
        assert f["filled_qty"] == 0
        assert f["end_reason"] == End.TIMEOUT
        assert f["end_time"] == 11.0
        assert f["end_idx"] == at_deadline
        assert math.isnan(f["fill_time"]) and f["fill_idx"] == -1


def test_trade_through_our_price_fills_every_model():
    ex = base()
    a = ex.submit(1.0, BUY, P, 100)
    place = here(ex)
    ex.delete(2.0, a)                  # the real level at P is gone; we would be alone there
    ex.market(3.0, SELL, 100)          # prints at P - 1 tick: the seller would have hit us
    r = run(ex, place, BUY, P)
    for model in MODELS:
        f = one(r[model])
        assert (f["filled_qty"], f["fill_time"], f["fill_price"]) == (100, 3.0, P)


def test_opposite_quote_reaching_our_price_fills_every_model():
    ex = base()
    a = ex.submit(1.0, BUY, P, 100)
    place = here(ex)
    ex.delete(2.0, a)
    ex.submit(3.0, SELL, P, 100)       # best ask now equals our bid
    r = run(ex, place, BUY, P)
    for model in MODELS:
        assert one(r[model])["fill_time"] == 3.0


def test_level_exit_when_price_leaves_ten_visible_levels():
    ex = base(ask_ticks=20)
    place = here(ex)                   # we join the bid at P - 1 tick
    for k in range(10):                # ten better bids push P - 1 tick to level 11
        ex.submit(1.0 + k, BUY, P + k * TICK, 100)
    ex.market(20.0, SELL, 10_000)      # a sweep that would reach us, too late
    r = run(ex, place, BUY, P - TICK)
    for model in MODELS:
        f = one(r[model])
        assert f["end_reason"] == End.LEVEL_EXIT
        assert f["end_time"] == 10.0 and f["filled_qty"] == 0


def test_sell_side_mirror():
    ex = base()
    a = ex.submit(1.0, SELL, P, 200)   # the ask at P is now the best
    place = here(ex)
    ex.delete(2.0, a)
    ex.submit(3.0, SELL, P, 300)       # behind us
    ex.market(4.0, BUY, 100)           # executes it: FIFO fills us
    far_message(ex, 100.0)
    r = run(ex, place, SELL, P)
    fifo = one(r["fifo"])
    assert (fifo["queue_ahead"], fifo["filled_qty"], fifo["fill_time"]) == (200, 100, 4.0)
    assert one(r["through"])["filled_qty"] == 0


def test_orders_that_would_cross_or_sit_outside_the_book_are_rejected():
    ex = base()
    place = here(ex)
    ex.submit(1.0, BUY, P - 5 * TICK, 100)
    m, b = ex.build()
    orders = Orders.make([place, place], [BUY, SELL], [P + TICK, P - TICK])
    for model in MODELS:
        assert np.all(simulate(m, b, orders, model, engine="python").end_reason == End.REJECTED)


def test_many_concurrent_orders_match_one_at_a_time():
    m, b = random_stream(np.random.default_rng(7), n_events=500)
    rng = np.random.default_rng(8)
    idx = np.sort(rng.integers(0, len(m) - 1, 40))
    side = rng.choice([BUY, SELL], 40).astype(np.int8)
    price = np.where(side > 0, b.bid_price[idx, 0], b.ask_price[idx, 0])
    orders = Orders.make(idx, side, price, 100, rng.uniform(1, 30, 40))
    for model in MODELS:
        together = simulate(m, b, orders, model, engine="python")
        for k in range(len(orders)):
            alone = simulate(m, b, Orders.make(idx[k:k + 1], side[k:k + 1], price[k:k + 1], 100,
                                               orders.max_wait[k:k + 1]), model, engine="python")
            for field in Fills.FIELDS:
                a, c = getattr(together, field)[k], getattr(alone, field)[0]
                assert a == c or (np.isnan(a) and np.isnan(c)), (model, k, field)


def test_fill_models_nest_and_are_deterministic():
    m, b = random_stream(np.random.default_rng(3), n_events=2000)
    rng = np.random.default_rng(4)
    idx = np.sort(rng.integers(0, len(m) - 1, 300))
    side = rng.choice([BUY, SELL], 300).astype(np.int8)
    price = np.where(side > 0, b.bid_price[idx, 0], b.ask_price[idx, 0])
    orders = Orders.make(idx, side, price, 100, 20.0)
    touch, fifo, through = (simulate(m, b, orders, k, engine="python") for k in MODELS)
    assert np.all(fifo.filled_qty <= touch.filled_qty)
    assert np.all(through.filled_qty <= fifo.filled_qty)
    both = (fifo.filled_qty > 0) & (touch.filled_qty > 0)
    # touch fills in one go at the first print; FIFO can only be later
    assert np.all(fifo.fill_idx[both] >= touch.fill_idx[both])
    assert simulate(m, b, orders, "fifo", engine="python").equals(fifo)


def test_bad_inputs_raise():
    ex = base()
    m, b = ex.build()
    with pytest.raises(ValueError):
        simulate(m, b, Orders.make([len(m)], BUY, P - TICK), "fifo", engine="python")
    with pytest.raises(ValueError):
        simulate(m, b, Orders.make([0], BUY, P - TICK), "queue", engine="python")
