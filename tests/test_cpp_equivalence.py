"""The C++ fill simulator must reproduce the Python reference bit for bit.

Skipped when the extension is not built (`make cpp`). Three layers:
* random but valid LOBSTER-like streams from `hypothesis` (shrinkable action lists),
  with hypothetical orders anywhere: at, inside and behind the touch, crossing the book,
  outside the visible levels, with waits from zero to longer than the stream;
* seeded streams with many concurrent orders;
* the five real LOBSTER days (skipped when the Parquet files are missing).
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from markout.lob.fills import MODELS, Fills, Orders, cpp_available, simulate
from markout.lob.lobster import TICKERS, available, load_day
from markout.lob.markouts import sample_orders
from markout.lob.synthetic import KINDS, SIZES, apply_action, random_stream, seeded_exchange

pytestmark = pytest.mark.skipif(not cpp_available(), reason="C++ extension not built (make cpp)")

TICK = 100

# weights: mostly joining the touch and marketable orders, which is what moves queues;
# halts (which end every live order) and auction crosses are rare
KIND_WEIGHTS = {"submit": 12, "cancel": 3, "delete": 4, "market": 10, "hidden": 2, "cross": 1, "halt": 1}
assert set(KIND_WEIGHTS) == set(KINDS)

action = st.tuples(
    st.sampled_from([k for k, w in KIND_WEIGHTS.items() for _ in range(w)]),
    st.sampled_from([1, -1]),
    st.sampled_from([0, 0, 0, 0, -1, 1, 2, 4]),       # submit offset: mostly at the touch
    st.sampled_from(SIZES),
    st.integers(0, 10**6),
    st.sampled_from([0.0, 0.0, 0.001, 0.3, 1.0, 4.0]),  # timestamp ties are common in LOBSTER
)


@st.composite
def streams(draw, max_events: int = 250):
    levels = draw(st.integers(1, 10))
    ex = seeded_exchange(levels, depth=draw(st.integers(1, 4)), size=draw(st.sampled_from(SIZES)))
    t = 34_200.0
    for kind, side, offset, size, pick, dt in draw(st.lists(action, min_size=1, max_size=max_events)):
        t += dt
        apply_action(ex, t, kind, side, offset, size, pick)
    return ex.build()


def orders_for(data, messages, book, k_max: int = 25) -> Orders:
    n = len(messages)
    k = data.draw(st.integers(1, k_max))
    idx = np.array(data.draw(st.lists(st.integers(0, n - 1), min_size=k, max_size=k)))
    side = np.array(data.draw(st.lists(st.sampled_from([1, -1]), min_size=k, max_size=k)), np.int8)
    # ticks behind our own best (-1 = inside the spread, large = deep or out of view)
    off = np.array(data.draw(st.lists(st.sampled_from([0, 0, 0, 1, 2, -1, -2, 12]),
                                      min_size=k, max_size=k)))
    best = np.where(side > 0, book.bid_price[idx, 0], book.ask_price[idx, 0])
    price = best - side.astype(np.int64) * off * TICK
    size = data.draw(st.lists(st.sampled_from([1, 100, 100, 250]), min_size=k, max_size=k))
    wait = data.draw(st.lists(st.sampled_from([0.0, 0.25, 1.0, 5.0, 30.0, 1e6]), min_size=k, max_size=k))
    return Orders.make(idx, side, price, np.array(size), np.array(wait))


def assert_identical(messages, book, orders) -> None:
    for model in MODELS:
        ref = simulate(messages, book, orders, model, engine="python")
        got = simulate(messages, book, orders, model, engine="cpp")
        for field in Fills.FIELDS:
            a, b = getattr(ref, field), getattr(got, field)
            assert a.dtype == b.dtype, (model, field)
            if a.dtype.kind == "f":      # compare bit patterns: NaN == NaN, -0.0 != 0.0
                a, b = a.view(np.int64), b.view(np.int64)
            assert np.array_equal(a, b), (model, field, np.flatnonzero(a != b)[:5])


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(stream=streams(), data=st.data())
def test_random_streams_match_python(stream, data):
    messages, book = stream
    assert_identical(messages, book, orders_for(data, messages, book))


@pytest.mark.parametrize("seed", range(6))
def test_seeded_streams_with_many_concurrent_orders(seed):
    rng = np.random.default_rng(100 + seed)
    messages, book = random_stream(rng, n_events=3000, levels=int(rng.integers(2, 11)))
    k = 400
    idx = np.sort(rng.integers(0, len(messages), k))
    side = rng.choice([1, -1], k).astype(np.int8)
    price = np.where(side > 0, book.bid_price[idx, 0], book.ask_price[idx, 0]) \
        - side * rng.choice([0, 0, 1, 3], k) * TICK
    orders = Orders.make(idx, side, price, rng.choice([1, 100, 300], k), rng.uniform(0, 60, k))
    assert_identical(messages, book, orders)


@pytest.mark.data
@pytest.mark.skipif(not available(), reason="LOBSTER Parquet files not downloaded")
@pytest.mark.parametrize("ticker", TICKERS)
def test_real_days_match_python(ticker):
    day = load_day(ticker)
    orders = sample_orders(day).orders       # the exact order set behind report 02
    assert_identical(day.messages, day.book, orders)


def test_empty_orders_and_bad_input():
    messages, book = random_stream(np.random.default_rng(0), n_events=50)
    none = Orders.make(np.array([], dtype=np.int64), 1, 0)
    for model in MODELS:
        assert len(simulate(messages, book, none, model, engine="cpp")) == 0
    from markout.lob import _queue_sim
    good = dict(time=messages.time, type=messages.type, order_id=messages.order_id,
                size=messages.size, price=messages.price, direction=messages.direction,
                bid_price=book.bid_price, bid_size=book.bid_size, ask_price=book.ask_price,
                ask_size=book.ask_size, place_idx=np.array([0]), side=np.array([1], np.int8),
                order_price=np.array([book.bid_price[0, 0]]), order_size=np.array([100]),
                max_wait=np.array([1.0]), model=1)
    assert _queue_sim.simulate(**good)["filled_qty"].shape == (1,)
    with pytest.raises(ValueError):
        _queue_sim.simulate(**{**good, "place_idx": np.array([len(messages)])})
    with pytest.raises(ValueError):
        _queue_sim.simulate(**{**good, "size": messages.size[:-1]})
    with pytest.raises(ValueError):
        _queue_sim.simulate(**{**good, "model": 3})


def test_auto_engine_uses_cpp():
    messages, book = random_stream(np.random.default_rng(1), n_events=200)
    orders = Orders.make([10, 20], [1, -1], [book.bid_price[10, 0], book.ask_price[20, 0]])
    assert simulate(messages, book, orders, "fifo", engine="auto").equals(
        simulate(messages, book, orders, "fifo", engine="cpp"))
