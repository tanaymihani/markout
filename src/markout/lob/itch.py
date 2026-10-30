"""NASDAQ TotalView-ITCH 5.0 -> LOBSTER-format messages and books, for any day and ticker.

NASDAQ publishes full sample days of its ITCH feed (emi.nasdaq.com/ITCH/Nasdaq ITCH/): every
message the exchange sent that day, for every stock, keyed by order reference number. `convert`
streams one gzipped day once and, for the tickers asked for, rebuilds each book from the first
add of the day. It writes what LOBSTER would: one row for every event in the regular session
(09:30-16:00) that changes the `levels` visible levels or prints a trade, with the visible book
after it. The fill simulator, the markouts and `lobster.check_consistency` run on it unchanged.

ITCH message -> LOBSTER type:
    A, F  add order                   1 submit
    X     partial cancel              2 cancel
    D     delete                      3 delete
    E     execution                   4 execute (visible)
    C     execution with price        4, at the resting order's limit price (where the depth changes)
    U     replace                     3 delete of the old order, then 1 submit of the new one
                                        (a replaced order loses its place in the queue)
    P     execution of a hidden order 5 (NASDAQ reports the side as 'B' for these since 2014)
    Q     opening/closing cross       6
    H     trading action              7, price -1 halt, 0 quotation only or paused, 1 resume
Events before 09:30 and after 16:00 only update the book.

Correctness check: every execution, cancel, delete and replace must name an order added earlier
the same day (books start empty every morning). `Stats.unmatched` counts those that don't; a
correct parse of a full day has none.

    python -m markout.lob.itch data/raw/itch/01302019.NASDAQ_ITCH50.gz --tickers AAPL,MSFT
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import json
import re
import shutil
import struct
import subprocess
import sys
import time
from array import array
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from markout.lob.lobster import DUMMY_ASK, DUMMY_BID, ITCH_DIR, LEVELS, MESSAGE_COLUMNS, MsgType
from markout.paths import RAW

ITCH_RAW = RAW / "itch"
OPEN_NS, CLOSE_NS = 34_200 * 10**9, 57_600 * 10**9  # 09:30 and 16:00, nanoseconds after midnight
CHUNK = 1 << 24  # bytes of decompressed feed per read
FLUSH_ROWS = 100_000  # rows buffered per ticker before they go to disk

# Message layouts (big-endian). The header is type, stock locate, tracking number and a 6-byte
# timestamp in nanoseconds after midnight, read here as a 16-bit and a 32-bit half.
HEAD = ">cHHHI"
ADD = struct.Struct(HEAD + "QcI8sI")  # ref, side, shares, stock, price (A; F adds a 4-byte MPID)
EXEC = struct.Struct(HEAD + "QIQ")  # ref, shares, match number (E)
EXEC_PX = struct.Struct(HEAD + "QIQcI")  # ref, shares, match, printable, price (C)
CANCEL = struct.Struct(HEAD + "QI")  # ref, shares (X)
DELETE = struct.Struct(HEAD + "Q")  # ref (D)
REPLACE = struct.Struct(HEAD + "QQII")  # old ref, new ref, shares, price (U)
TRADE = struct.Struct(HEAD + "QcI8sIQ")  # ref, side, shares, stock, price, match (P)
CROSS = struct.Struct(HEAD + "Q8sIQc")  # shares, stock, price, match, cross type (Q)
DIRECTORY = struct.Struct(HEAD + "8s")  # stock (R; the rest is not needed)
ACTION = struct.Struct(HEAD + "8sc")  # stock, trading state (H)
SIZES = {b"A": 36, b"F": 40, b"E": 31, b"C": 36, b"X": 23, b"D": 19, b"U": 35, b"P": 44, b"Q": 40, b"R": 39, b"H": 25}
HALT_CODE = {b"H": -1, b"P": 0, b"Q": 0, b"T": 1}


@dataclass
class Stats:
    ticker: str
    locate: int = -1
    itch_messages: int = 0  # every message for this stock, all day
    rows: int = 0  # LOBSTER rows written (regular session, visible changes and trades)
    adds: int = 0
    executions: int = 0
    cancels: int = 0
    deletes: int = 0
    replaces: int = 0
    hidden: int = 0
    crosses: int = 0
    unmatched: dict[str, int] = field(default_factory=lambda: {"E": 0, "C": 0, "X": 0, "D": 0, "U": 0})


class TickerBook:
    """One stock's book, rebuilt from the feed, and the LOBSTER rows it writes."""

    def __init__(self, ticker: str, levels: int, out_dir: Path | None, emit_all: bool = False):
        self.stats = Stats(ticker)
        self.levels, self.emit_all = levels, emit_all
        self.orders: dict[int, list[int]] = {}  # ref -> [side, price, shares]
        self.depth: dict[int, dict[int, int]] = {1: {}, -1: {}}  # side -> price -> displayed shares
        self.prices: dict[int, list[int]] = {1: [], -1: []}  # occupied prices per side, ascending
        self.last: tuple[int, ...] | None = None  # last written snapshot
        self.msg = {c: array("q") for c in MESSAGE_COLUMNS}
        self.book = array("q")
        self.out_dir = out_dir
        self._writers: dict[str, pq.ParquetWriter] = {}

    # ---------------------------------------------------------------- book updates
    def _change(self, side: int, price: int, delta: int) -> None:
        level = self.depth[side]
        q = level.get(price, 0) + delta
        if q > 0:
            if price not in level:
                bisect.insort(self.prices[side], price)
            level[price] = q
        else:
            if price in level:
                del level[price]
                ps = self.prices[side]
                del ps[bisect.bisect_left(ps, price)]

    def snapshot(self) -> tuple[int, ...]:
        k = self.levels
        asks = self.prices[-1][:k]
        bids = self.prices[1][: -k - 1 : -1]  # the k highest, best first
        da, db = self.depth[-1], self.depth[1]
        row: list[int] = []
        for i in range(k):
            if i < len(asks):
                row += (asks[i], da[asks[i]])
            else:
                row += (DUMMY_ASK, 0)
            if i < len(bids):
                row += (bids[i], db[bids[i]])
            else:
                row += (DUMMY_BID, 0)
        return tuple(row)

    def _row(self, ts: int, typ: int, ref: int, size: int, price: int, direction: int, trade: bool) -> None:
        """Write one LOBSTER row if the event is in the session and visible (or a trade)."""
        if not OPEN_NS <= ts < CLOSE_NS:
            return
        snap = self.snapshot()
        if not (trade or self.emit_all or snap != self.last):
            return  # deeper in the book than the visible levels: LOBSTER leaves these out too
        self.last = snap
        m = self.msg
        m["time"].append(ts)
        m["type"].append(typ)
        m["order_id"].append(ref)
        m["size"].append(size)
        m["price"].append(price)
        m["direction"].append(direction)
        self.book.extend(snap)
        self.stats.rows += 1
        if len(m["time"]) >= FLUSH_ROWS:
            self.flush()

    # ---------------------------------------------------------------- ITCH events
    def add(self, ts: int, ref: int, side: int, shares: int, price: int) -> None:
        self.orders[ref] = [side, price, shares]
        self._change(side, price, shares)
        self.stats.adds += 1
        self._row(ts, MsgType.SUBMIT, ref, shares, price, side, False)

    def execute(self, ts: int, ref: int, shares: int, kind: str) -> None:
        o = self.orders.get(ref)
        if o is None:
            self.stats.unmatched[kind] += 1
            return
        side, price, left = o
        self._take(ref, o, shares)
        self.stats.executions += 1
        self._row(ts, MsgType.EXECUTE, ref, min(shares, left), price, side, True)

    def cancel(self, ts: int, ref: int, shares: int) -> None:
        o = self.orders.get(ref)
        if o is None:
            self.stats.unmatched["X"] += 1
            return
        side, price, left = o
        self._take(ref, o, shares)
        self.stats.cancels += 1
        self._row(ts, MsgType.CANCEL if shares < left else MsgType.DELETE, ref, min(shares, left), price, side, False)

    def delete(self, ts: int, ref: int, kind: str = "D") -> None:
        o = self.orders.pop(ref, None)
        if o is None:
            self.stats.unmatched[kind] += 1
            return
        side, price, left = o
        self._change(side, price, -left)
        self.stats.deletes += kind == "D"
        self._row(ts, MsgType.DELETE, ref, left, price, side, False)

    def replace(self, ts: int, old: int, new: int, shares: int, price: int) -> None:
        o = self.orders.get(old)
        if o is None:
            self.stats.unmatched["U"] += 1
            return
        side = o[0]
        self.delete(ts, old, kind="U")
        self.stats.replaces += 1
        self.orders[new] = [side, price, shares]
        self._change(side, price, shares)
        self._row(ts, MsgType.SUBMIT, new, shares, price, side, False)

    def hidden(self, ts: int, side: int, shares: int, price: int) -> None:
        self.stats.hidden += 1
        self._row(ts, MsgType.HIDDEN, 0, shares, price, side, True)

    def cross(self, ts: int, shares: int, price: int) -> None:
        self.stats.crosses += 1
        if shares:
            self._row(ts, MsgType.CROSS, 0, shares, price, -1, True)

    def action(self, ts: int, state: bytes) -> None:
        self._row(ts, MsgType.HALT, 0, 0, HALT_CODE.get(state, 0), -1, True)

    def _take(self, ref: int, o: list[int], shares: int) -> None:
        side, price, left = o
        n = min(shares, left)
        self._change(side, price, -n)
        if n >= left:
            del self.orders[ref]
        else:
            o[2] = left - n

    # ---------------------------------------------------------------- output
    def arrays(self) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """What is still buffered, as NumPy: message columns and an (n, 4 * levels) book."""
        cols = {c: np.frombuffer(self.msg[c], dtype=np.int64) for c in MESSAGE_COLUMNS}
        book = np.frombuffer(self.book, dtype=np.int64).reshape(-1, 4 * self.levels)
        return cols, book

    def flush(self) -> None:
        if self.out_dir is None or not len(self.msg["time"]):
            return
        cols, book = self.arrays()
        msgs = pa.table(
            {
                "time": pa.array(cols["time"] / 1e9, pa.float64()),
                "type": pa.array(cols["type"].astype(np.int8)),
                "order_id": pa.array(cols["order_id"]),
                "size": pa.array(cols["size"]),
                "price": pa.array(cols["price"]),
                "direction": pa.array(cols["direction"].astype(np.int8)),
            }
        )
        names = [
            f"{f}_{i}" for i in range(1, self.levels + 1) for f in ("ask_price", "ask_size", "bid_price", "bid_size")
        ]
        books = pa.table({n: pa.array(book[:, j]) for j, n in enumerate(names)})
        for kind, table in (("messages", msgs), ("book", books)):
            if kind not in self._writers:
                path = self.out_dir / f"{self.stats.ticker}_{kind}.parquet"
                self._writers[kind] = pq.ParquetWriter(path, table.schema, compression="zstd")
            self._writers[kind].write_table(table)
        self.msg = {c: array("q") for c in MESSAGE_COLUMNS}
        self.book = array("q")

    def close(self) -> None:
        self.flush()
        for w in self._writers.values():
            w.close()


# -------------------------------------------------------------------------------- the stream
def feed(path: Path):
    """Decompressed chunks of a gzipped ITCH file (gzip runs in its own process, in parallel)."""
    if path.suffix != ".gz":
        with open(path, "rb") as f:
            while chunk := f.read(CHUNK):
                yield chunk
        return
    if shutil.which("gzip"):
        proc = subprocess.Popen(["gzip", "-dc", str(path)], stdout=subprocess.PIPE)
        assert proc.stdout is not None
        while chunk := proc.stdout.read(CHUNK):
            yield chunk
        if proc.wait():
            raise OSError(f"gzip failed on {path} (truncated or corrupt download?)")
        return
    with gzip.open(path, "rb") as f:  # pragma: no cover - every supported OS ships gzip
        while chunk := f.read(CHUNK):
            yield chunk


def parse(chunks, tickers: list[str], levels: int = LEVELS, out_dir: Path | None = None, emit_all: bool = False):
    """Run the feed through one book per ticker. Returns ({ticker: TickerBook}, day stats)."""
    wanted = {t.upper().encode().ljust(8) for t in tickers}
    books: dict[int, TickerBook] = {}
    watch = bytearray(65536)  # stock locate -> 1 if one of ours (the directory assigns locates)
    total, t0, tail = 0, time.perf_counter(), b""
    by_name: dict[str, TickerBook] = {}
    for chunk in chunks:
        buf = tail + chunk if tail else chunk
        n, pos = len(buf), 0
        while pos + 2 <= n:
            end = pos + 2 + ((buf[pos] << 8) | buf[pos + 1])
            if end > n:
                break
            if watch[(buf[pos + 3] << 8) | buf[pos + 4]] or buf[pos + 2] == 82:  # 82 = b"R"
                _event(buf, pos + 2, books, watch, wanted, by_name, levels, out_dir, emit_all)
            total += 1
            pos = end
        tail = buf[pos:]
    if tail:
        raise ValueError(f"feed ends inside a message ({len(tail)} bytes left over)")
    for b in by_name.values():
        b.close()
    seconds = time.perf_counter() - t0
    return by_name, {"itch_messages": total, "seconds": seconds, "messages_per_s": total / max(seconds, 1e-9)}


def _event(buf, at, books, watch, wanted, by_name, levels, out_dir, emit_all) -> None:
    t = buf[at : at + 1]
    if t == b"R":
        _, locate, _, hi, lo, stock = DIRECTORY.unpack_from(buf, at)
        if stock in wanted and locate not in books:
            name = stock.decode().strip()
            books[locate] = by_name[name] = TickerBook(name, levels, out_dir, emit_all)
            books[locate].stats.locate = locate
            watch[locate] = 1
        return
    _, locate, _, hi, lo = struct.unpack_from(HEAD, buf, at)
    b = books[locate]
    b.stats.itch_messages += 1
    ts = (hi << 32) | lo
    if t == b"A" or t == b"F":
        _, _, _, _, _, ref, side, shares, _, price = ADD.unpack_from(buf, at)
        b.add(ts, ref, 1 if side == b"B" else -1, shares, price)
    elif t == b"E":
        ref, shares = EXEC.unpack_from(buf, at)[5:7]
        b.execute(ts, ref, shares, "E")
    elif t == b"C":
        ref, shares = EXEC_PX.unpack_from(buf, at)[5:7]
        b.execute(ts, ref, shares, "C")
    elif t == b"X":
        ref, shares = CANCEL.unpack_from(buf, at)[5:7]
        b.cancel(ts, ref, shares)
    elif t == b"D":
        b.delete(ts, DELETE.unpack_from(buf, at)[5])
    elif t == b"U":
        old, new, shares, price = REPLACE.unpack_from(buf, at)[5:9]
        b.replace(ts, old, new, shares, price)
    elif t == b"P":
        _, _, _, _, _, _, side, shares, _, price, _ = TRADE.unpack_from(buf, at)
        b.hidden(ts, 1 if side == b"B" else -1, shares, price)
    elif t == b"Q":
        shares, _, price = CROSS.unpack_from(buf, at)[5:8]
        b.cross(ts, shares, price)
    elif t == b"H":
        b.action(ts, ACTION.unpack_from(buf, at)[6])


# -------------------------------------------------------------------------------- files
def file_date(path: Path) -> date:
    """The trading day in NASDAQ's file names: 01302019.NASDAQ_ITCH50.gz or S101819-v50.txt.gz."""
    name = Path(path).name
    if m := re.match(r"(\d{2})(\d{2})(\d{4})\.NASDAQ_ITCH50", name):
        return date(int(m[3]), int(m[1]), int(m[2]))
    if m := re.match(r"S(\d{2})(\d{2})(\d{2})-v50", name):
        return date(2000 + int(m[3]), int(m[1]), int(m[2]))
    raise ValueError(f"no date in {name}")


def day_dir(day: date | str) -> Path:
    return ITCH_DIR / str(day)


def convert(path: Path, tickers: list[str], levels: int = LEVELS, out_root: Path = ITCH_DIR) -> dict:
    """Convert one day for `tickers` into out_root/<YYYY-MM-DD>/<T>_{messages,book}.parquet plus stats.json."""
    day = file_date(path)
    out = out_root / str(day)
    tmp = out_root / f".{day}.partial"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    books, info = parse(feed(path), tickers, levels, tmp)
    report = {
        "date": str(day),
        "source": Path(path).name,
        "levels": levels,
        **info,
        "tickers": {t: asdict(b.stats) for t, b in sorted(books.items())},
        "missing": sorted(set(t.upper() for t in tickers) - set(books)),
    }
    (tmp / "stats.json").write_text(json.dumps(report, indent=2) + "\n")
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)  # only a finished day ever appears under its own name
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m markout.lob.itch", description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="+", type=Path, help="ITCH 5.0 day files (.gz)")
    ap.add_argument("--tickers", required=True, help="comma-separated symbols, e.g. AAPL,MSFT,INTC")
    ap.add_argument("--levels", type=int, default=LEVELS)
    a = ap.parse_args(argv)
    for f in a.files:
        r = convert(f, [t.strip() for t in a.tickers.split(",") if t.strip()], a.levels)
        bad = {t: s["unmatched"] for t, s in r["tickers"].items() if any(s["unmatched"].values())}
        print(
            f"{r['date']}: {r['itch_messages']:,} messages in {r['seconds']:.0f} s "
            f"({r['messages_per_s'] / 1e6:.1f}M/s); rows "
            + ", ".join(f"{t} {s['rows']:,}" for t, s in r["tickers"].items())
            + (f"; UNMATCHED {bad}" if bad else "; zero unmatched executions, cancels, deletes and replaces")
            + (f"; not in the feed: {', '.join(r['missing'])}" if r["missing"] else ""),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
