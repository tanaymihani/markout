"""The ITCH 5.0 converter: random feeds whose LOBSTER output is known, written as real ITCH bytes.

A small writer emits ITCH 5.0 messages for two stocks (one converted, one noise), and drives
the same events through `SyntheticExchange`, the test exchange the fill simulator is tested
against, to get the rows and books LOBSTER would show. The converter must reproduce them
exactly, whatever the chunk boundaries of the stream.
"""

import gzip
import json
import struct

import numpy as np
import pytest

from markout.lob import itch, lobster
from markout.lob.lobster import Book, Messages, MsgType, check_consistency
from markout.lob.synthetic import SyntheticExchange

NS = 10**9
TICK = 100
MID = 500_000  # $50.00


def s8(sym: str) -> bytes:
    return sym.encode().ljust(8)


class Feed:
    """Writes ITCH 5.0 (with the 2-byte length prefix) and mirrors the tracked stock in a SyntheticExchange."""

    def __init__(self, rng: np.random.Generator, levels: int = 10):
        self.rng, self.out = rng, bytearray()
        self.ex = SyntheticExchange(levels=levels)  # what LOBSTER should show for TEST
        self.noise = SyntheticExchange(levels=levels)  # a second stock, never converted
        self.next_ref, self.match = 1, 0
        self.phase = "before"  # before 09:30, open, closed (after 16:00)

    def _put(self, body: bytes) -> None:
        self.out += struct.pack(">H", len(body)) + body

    def _head(self, t: bytes, locate: int, ns: int) -> bytes:
        return struct.pack(">cHHHI", t, locate, 0, ns >> 32, ns & 0xFFFFFFFF)

    def _clock(self, ns: int) -> None:
        """Rows count from 09:30: LOBSTER shows the book as it stands then, not the events before."""
        if self.phase == "before" and ns >= itch.OPEN_NS:
            self.ex.rows.clear()
            self.ex.books.clear()
            self.phase = "open"
        if ns >= itch.CLOSE_NS:
            self.phase = "closed"  # nothing after 16:00 is written

    def mirror(self, ns: int, fn, *args) -> None:
        """Apply the event to the reference exchange; after the close keep its state but drop the row."""
        self._clock(ns)
        n = len(self.ex.rows)
        fn(ns / NS, *args)
        if self.phase == "closed":
            del self.ex.rows[n:], self.ex.books[n:]

    # ------------------------------------------------------------------ messages
    def directory(self, ns: int, locate: int, sym: str) -> None:
        body = self._head(b"R", locate, ns) + s8(sym) + b"Q" + b"N" + struct.pack(">I", 100) + b"N"
        body += b"C" + b"  " + b"P" + b"N" + b"N" + b"1" + b"N" + struct.pack(">I", 0) + b"N"
        assert len(body) == 39
        self._put(body)

    def system(self, ns: int, code: bytes) -> None:
        self._put(self._head(b"S", 0, ns) + code)

    def add(self, ns, locate, ref, side, shares, price, mpid=False):
        body = self._head(b"F" if mpid else b"A", locate, ns)
        body += struct.pack(
            ">QcI8sI", ref, b"B" if side > 0 else b"S", shares, s8("TEST" if locate == 1 else "NOIS"), price
        )
        self._put(body + (b"ABCD" if mpid else b""))

    def execute(self, ns, locate, ref, shares, with_price=False):
        self.match += 1
        if with_price:
            body = self._head(b"C", locate, ns) + struct.pack(">QIQcI", ref, shares, self.match, b"Y", 123_400)
        else:
            body = self._head(b"E", locate, ns) + struct.pack(">QIQ", ref, shares, self.match)
        self._put(body)

    def cancel(self, ns, locate, ref, shares):
        self._put(self._head(b"X", locate, ns) + struct.pack(">QI", ref, shares))

    def delete(self, ns, locate, ref):
        self._put(self._head(b"D", locate, ns) + struct.pack(">Q", ref))

    def replace(self, ns, locate, old, new, shares, price):
        self._put(self._head(b"U", locate, ns) + struct.pack(">QQII", old, new, shares, price))

    def trade(self, ns, locate, shares, price):
        self.match += 1
        sym = s8("TEST" if locate == 1 else "NOIS")
        self._put(self._head(b"P", locate, ns) + struct.pack(">QcI8sIQ", 0, b"B", shares, sym, price, self.match))

    def cross(self, ns, locate, shares, price):
        self.match += 1
        sym = s8("TEST" if locate == 1 else "NOIS")
        self._put(self._head(b"Q", locate, ns) + struct.pack(">Q8sIQc", shares, sym, price, self.match, b"O"))

    def action(self, ns, locate, state: bytes):
        self._put(self._head(b"H", locate, ns) + s8("TEST") + state + b" " + b"    ")

    def other(self, ns, locate):
        """A message type the converter must skip (Reg SHO restriction)."""
        self._put(self._head(b"Y", locate, ns) + s8("TEST") + b"0")

    # ------------------------------------------------------------------ one random event
    def step(self, ns: int, locate: int) -> None:
        ex = self.ex if locate == 1 else self.noise
        mirror = (lambda fn, *a: self.mirror(ns, fn, *a)) if locate == 1 else (lambda fn, *a: fn(ns / NS, *a))
        rng = self.rng
        kinds = ["add"] * 5 + ["exec", "cancel", "delete", "replace", "hidden", "other"]
        kind = kinds[rng.integers(len(kinds))] if ex.orders else "add"
        if kind == "add" or ex.best(1) is None or ex.best(-1) is None:
            side = 1 if rng.random() < 0.5 else -1
            own, opp = ex.best(side), ex.best(-side)
            ref_px = own if own is not None else (opp - side * TICK if opp is not None else MID - side * TICK)
            price = ref_px - side * TICK * int(rng.integers(-1, 14))  # some land beyond the visible levels
            if opp is not None and (price >= opp if side > 0 else price <= opp):
                price = opp - side * TICK
            shares, ref = int(rng.choice([1, 100, 100, 200, 300, 500])), self.next_ref
            self.next_ref += 1
            self.add(ns, locate, ref, side, shares, price, mpid=bool(rng.random() < 0.2))
            mirror(ex.submit, side, price, shares, ref)
        elif kind == "exec":
            side = 1 if rng.random() < 0.5 else -1
            ref = ex.queues[side][ex.best(side)][0]  # price-time priority: the front of the best queue
            shares = int(rng.integers(1, ex.orders[ref][2] + 1))
            self.execute(ns, locate, ref, shares, with_price=bool(rng.random() < 0.1))
            mirror(ex.execute, ref, shares)
        elif kind == "cancel":
            ref = int(rng.choice(sorted(ex.orders)))
            left = ex.orders[ref][2]
            if left > 1 and rng.random() < 0.8:
                shares = int(rng.integers(1, left))
                self.cancel(ns, locate, ref, shares)
                mirror(ex.cancel, ref, shares)
            else:  # a cancel of everything that is left is a delete in LOBSTER
                self.cancel(ns, locate, ref, left)
                mirror(ex.delete, ref)
        elif kind == "delete":
            ref = int(rng.choice(sorted(ex.orders)))
            self.delete(ns, locate, ref)
            mirror(ex.delete, ref)
        elif kind == "replace":
            ref = int(rng.choice(sorted(ex.orders)))
            side = ex.orders[ref][0]
            opp = ex.best(-side)
            price = ex.orders[ref][1] - side * TICK * int(rng.integers(-1, 3))
            if opp is not None and (price >= opp if side > 0 else price <= opp):
                price = opp - side * TICK
            shares, new = int(rng.choice([100, 200, 400])), self.next_ref
            self.next_ref += 1
            self.replace(ns, locate, ref, new, shares, price)
            mirror(ex.delete, ref)
            mirror(ex.submit, side, price, shares, new)
        elif kind == "hidden":
            price = (ex.best(1) + ex.best(-1)) // 2
            self.trade(ns, locate, 100, price)
            mirror(ex.hidden, price, 100, 1)
        else:
            self.other(ns, locate)


def random_feed(seed: int, n: int = 3000, levels: int = 10) -> Feed:
    f = Feed(np.random.default_rng(seed), levels)
    f.system(3 * 3600 * NS, b"O")
    f.directory(3 * 3600 * NS, 1, "TEST")
    f.directory(3 * 3600 * NS, 2, "NOIS")
    times = np.sort(f.rng.integers(int(9.0 * 3600 * NS), int(16.5 * 3600 * NS), n))  # 09:00-16:30
    for i, ns in enumerate(times):
        ns = int(ns)
        f.step(ns, 1 if i % 3 else 2)
        if i == n // 2:  # a halt and a resume, and an opening-style cross, mid-session
            f.action(ns, 1, b"H")
            f.mirror(ns, f.ex.halt, -1)
            f.action(ns, 1, b"T")
            f.mirror(ns, f.ex.halt, 1)
            f.cross(ns, 1, 5000, MID)
            f.mirror(ns, f.ex.cross_trade, MID, 5000)
    f.system(int(20 * 3600 * NS), b"C")
    return f


def chunks(data: bytes, rng: np.random.Generator, lo: int = 1, hi: int = 97):
    """The feed in random pieces, so messages straddle chunk boundaries."""
    i = 0
    while i < len(data):
        j = i + int(rng.integers(lo, hi))
        yield data[i:j]
        i = j


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_every_event_matches_the_reference_exchange(seed):
    f = random_feed(seed)
    books, info = itch.parse(chunks(bytes(f.out), np.random.default_rng(seed)), ["TEST"], emit_all=True)
    tb = books["TEST"]
    cols, book = tb.arrays()
    m, b = f.ex.build()
    assert len(cols["time"]) == len(m) > 500
    assert np.array_equal(cols["time"] / NS, m.time)
    for c in ("type", "order_id", "size", "price", "direction"):
        assert np.array_equal(cols[c], getattr(m, c).astype(np.int64)), c
    for k, name in enumerate(("ask_price", "ask_size", "bid_price", "bid_size")):
        assert np.array_equal(book[:, k::4], getattr(b, name)), name
    assert not any(tb.stats.unmatched.values())
    assert info["itch_messages"] > len(m)  # the noise stock and the skipped message types were read too


def test_default_output_keeps_only_visible_changes_and_trades():
    f = random_feed(7)
    every = itch.parse([bytes(f.out)], ["TEST"], emit_all=True)[0]["TEST"]
    seen = itch.parse([bytes(f.out)], ["TEST"])[0]["TEST"]
    (ca, ba), (cs, bs) = every.arrays(), seen.arrays()
    keep, last = [], None
    for i in range(len(ca["time"])):
        trade = ca["type"][i] in (MsgType.EXECUTE, MsgType.HIDDEN, MsgType.CROSS, MsgType.HALT)
        if trade or last is None or not np.array_equal(ba[i], last):
            keep.append(i)
            last = ba[i]
    assert 0 < len(keep) < len(ca["time"])  # some events were deeper than the 10 visible levels
    assert np.array_equal(bs, ba[keep]) and np.array_equal(cs["order_id"], ca["order_id"][keep])
    messages = Messages(
        cs["time"] / NS,
        cs["type"].astype(np.int8),
        cs["order_id"],
        cs["size"],
        cs["price"],
        cs["direction"].astype(np.int8),
    )
    book = Book(*(np.ascontiguousarray(bs[:, k::4]) for k in range(4)))
    c = check_consistency(messages, book)
    assert c.n_checked > 0.9 * c.n_eligible and c.pass_rate == 1.0  # every replayed depth change adds up


def test_unknown_order_references_are_counted():
    f = Feed(np.random.default_rng(0))
    f.directory(NS, 1, "TEST")
    f.add(itch.OPEN_NS + 1, 1, 10, 1, 100, MID)
    f.execute(itch.OPEN_NS + 2, 1, 99, 50)
    f.execute(itch.OPEN_NS + 3, 1, 98, 50, with_price=True)
    f.cancel(itch.OPEN_NS + 4, 1, 97, 10)
    f.delete(itch.OPEN_NS + 5, 1, 96)
    f.replace(itch.OPEN_NS + 6, 1, 95, 94, 100, MID)
    tb = itch.parse([bytes(f.out)], ["TEST"])[0]["TEST"]
    assert tb.stats.unmatched == {"E": 1, "C": 1, "X": 1, "D": 1, "U": 1}
    assert tb.stats.rows == 1  # only the add made it


def test_a_truncated_feed_is_an_error():
    f = random_feed(3, n=50)
    with pytest.raises(ValueError, match="inside a message"):
        itch.parse([bytes(f.out)[:-5]], ["TEST"])


def test_file_dates():
    assert str(itch.file_date("01302019.NASDAQ_ITCH50.gz")) == "2019-01-30"
    assert str(itch.file_date("S101819-v50.txt.gz")) == "2019-10-18"
    with pytest.raises(ValueError):
        itch.file_date("tvagg.gz")


def test_convert_writes_a_lobster_day_the_loader_reads(tmp_path, monkeypatch):
    f = random_feed(11)
    src = tmp_path / "01302019.NASDAQ_ITCH50.gz"
    src.write_bytes(gzip.compress(bytes(f.out)))
    monkeypatch.setattr(itch, "FLUSH_ROWS", 97)  # many small row groups
    monkeypatch.setattr(lobster, "ITCH_DIR", tmp_path / "itch")
    report = itch.convert(src, ["test", "MISSING"], out_root=tmp_path / "itch")
    assert report["date"] == "2019-01-30" and report["missing"] == ["MISSING"]
    stats = json.loads((tmp_path / "itch" / "2019-01-30" / "stats.json").read_text())
    assert stats["tickers"]["TEST"]["rows"] == report["tickers"]["TEST"]["rows"] > 0
    assert lobster.itch_days() == ["2019-01-30"] and lobster.available(("TEST",), "2019-01-30")
    day = lobster.load_day("TEST", "2019-01-30")
    direct = itch.parse([bytes(f.out)], ["TEST"])[0]["TEST"]
    cols, book = direct.arrays()
    assert np.array_equal(day.messages.time, cols["time"] / NS) and np.array_equal(day.messages.price, cols["price"])
    assert np.array_equal(day.book.bid_size, book[:, 3::4]) and day.book.levels == 10
    assert not (tmp_path / "itch" / ".2019-01-30.partial").exists()
