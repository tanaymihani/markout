"""Checks on the real LOBSTER files (skipped when the Parquet files are missing).

These test integrity and identities, not results: the consistency replay, the nesting of
the fill models, and accounting identities that must hold whatever the data say.
"""

from __future__ import annotations

import numpy as np
import pytest

from markout.lob.facts import compute_facts
from markout.lob.fills import MODELS, simulate
from markout.lob.lobster import TICK, TICKERS, available, check_consistency, load_day
from markout.lob.markouts import mid_at, order_markouts, sample_orders, spread_decomposition

pytestmark = [pytest.mark.data,
              pytest.mark.skipif(not available(), reason="LOBSTER Parquet files not downloaded")]


@pytest.fixture(scope="module")
def goog():
    return load_day("GOOG")


@pytest.mark.parametrize("ticker", TICKERS)
def test_messages_replay_exactly_against_the_book(ticker):
    day = load_day(ticker)
    assert len(day.messages) == len(day.book)
    assert np.all(np.diff(day.messages.time) >= 0)
    c = check_consistency(day.messages, day.book)
    assert c.coverage > 0.99
    assert c.pass_rate >= 0.999, f"{ticker}: {len(c.failures)} failures"
    assert np.all(day.book.best_bid < day.book.best_ask)          # never crossed or locked


def test_tick_size_regimes_match_the_plan():
    cls = {t: compute_facts(load_day(t)).tick_class for t in ("INTC", "MSFT", "AAPL", "GOOG")}
    assert cls == {"INTC": "large-tick", "MSFT": "large-tick",
                   "AAPL": "small-tick", "GOOG": "small-tick"}


def test_fill_models_nest_on_real_data(goog):
    o = sample_orders(goog, every=20.0).orders
    touch, fifo, through = (simulate(goog.messages, goog.book, o, m, engine="python") for m in MODELS)
    assert np.all(through.filled_qty <= fifo.filled_qty)
    assert np.all(fifo.filled_qty <= touch.filled_qty)
    assert np.all(fifo.queue_ahead == touch.queue_ahead)          # same book row at placement
    filled = fifo.filled_qty > 0
    assert np.all(fifo.fill_time[filled] >= goog.messages.time[o.place_idx[filled]])


def test_unconditional_markout_is_the_half_spread(goog):
    # both sides at every decision time: the mid move cancels, the half-spread remains
    s = sample_orders(goog, every=30.0)
    df = order_markouts(goog, s, models=("fifo",), engine="python")
    i = np.searchsorted(goog.messages.time, s.decision_time, side="right") - 1
    half = (goog.book.best_ask[i] - goog.book.best_bid[i]) / 2 / TICK
    assert np.isclose(df["uncond_10"].mean(), half.mean())


def test_spread_decomposition_identity(goog):
    for row in spread_decomposition(goog):
        assert np.isclose(row["effective_ticks"], row["realized_ticks"] + row["impact_ticks"])
        assert np.isclose(row["effective_bps"], row["realized_bps"] + row["impact_bps"])


def test_mid_at_reads_the_state_in_effect(goog):
    m = goog.messages
    t = m.time[1000]
    assert mid_at(goog, np.array([t]))[0] == goog.book.mid()[np.searchsorted(m.time, t, "right") - 1]
