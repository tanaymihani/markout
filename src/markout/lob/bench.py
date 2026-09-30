"""Benchmark of the fill simulator: Python reference vs the C++17 port.

`python -m markout.lob.bench` (or `make bench`, which builds the extension first):

1. profiles the Python reference with cProfile on the largest file (MSFT, FIFO model), the
   evidence that the event loop is worth porting: the time is interpreter overhead spread
   over millions of tiny operations, not a hot spot NumPy could vectorise;
2. times both engines on all five stocks and all three fill models, with the order set
   behind report 02 (100 shares at both best quotes every 5 s, 60 s max wait): one warm-up
   run, then the median of 5 timed runs through the same `fills.simulate` call. The
   Python time includes its list conversions, and the C++ time includes the binding
   overhead and building the output arrays;
3. checks on every run that the C++ output is bit-identical to Python's;
4. saves reports/results/cpp_bench.json and renders g_throughput.
"""

from __future__ import annotations

import cProfile
import io
import platform
import pstats
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from markout import plotting as mp
from markout import results
from markout.lob.fills import MODELS, cpp_available, simulate
from markout.lob.lobster import TICKERS, available, load_day
from markout.lob.markouts import sample_orders
from markout.paths import ROOT

NAME = "cpp_bench"
RUNS = 5
PROFILE_TICKER = "MSFT"
PROFILE_ROWS = 12


def profile_python(ticker: str = PROFILE_TICKER, model: str = "fifo", rows: int = PROFILE_ROWS) -> dict:
    day = load_day(ticker)
    orders = sample_orders(day).orders
    pr = cProfile.Profile()
    pr.enable()
    simulate(day.messages, day.book, orders, model, engine="python")
    pr.disable()
    st = pstats.Stats(pr)
    total = st.total_tt
    top = sorted(st.stats.items(), key=lambda kv: kv[1][2], reverse=True)[:rows]
    table = []
    for (file, line, func), (cc, nc, tt, ct, _) in top:
        where = f"{Path(file).name}:{line}" if line else file
        table.append({"function": f"{where}({func})", "ncalls": int(nc), "tottime_s": tt,
                      "cumtime_s": ct, "share_of_total": tt / total if total else None})
    text = io.StringIO()
    pstats.Stats(pr, stream=text).sort_stats("tottime").print_stats(rows)
    # repo-relative paths only: the report is published, local paths are not
    clean = text.getvalue().strip().replace(str(ROOT) + "/", "")
    return {"ticker": ticker, "model": model, "total_s": total,
            "function_calls": int(sum(v[1] for v in st.stats.values())),
            "rows": table, "text": clean}


def timed(fn, runs: int = RUNS) -> tuple[float, list[float], object]:
    out = fn()                                   # warm-up (also the result we check)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), times, out


def compiler_info() -> dict:
    cache = ROOT / "build" / "cpp" / "CMakeCache.txt"
    info = {"flags": "-O3 -ffp-contract=off (no -ffast-math), C++17, CMAKE_BUILD_TYPE=Release"}
    if cache.exists():
        for line in cache.read_text().splitlines():
            for key in ("CMAKE_CXX_COMPILER:", "CMAKE_BUILD_TYPE:", "CMAKE_OSX_SYSROOT:"):
                if line.startswith(key):
                    info[key.rstrip(":").lower()] = line.split("=", 1)[1]
    return info


def run(runs: int = RUNS) -> dict:
    rows = []
    for t in TICKERS:
        day = load_day(t)
        orders = sample_orders(day).orders
        for model in MODELS:
            py_med, py_times, py_out = timed(
                lambda: simulate(day.messages, day.book, orders, model, engine="python"), runs)
            cc_med, cc_times, cc_out = timed(
                lambda: simulate(day.messages, day.book, orders, model, engine="cpp"), runs)
            first = int(orders.place_idx.min())
            replayed = int(py_out.end_idx.max()) - first + 1
            rows.append({"ticker": t, "model": model, "orders": len(orders),
                         "messages_in_file": len(day.messages), "messages_replayed": replayed,
                         "python_median_s": py_med, "cpp_median_s": cc_med,
                         "python_runs_s": py_times, "cpp_runs_s": cc_times,
                         "python_msgs_per_s": replayed / py_med, "cpp_msgs_per_s": replayed / cc_med,
                         "cpp_orders_per_s": len(orders) / cc_med,
                         "speedup": py_med / cc_med, "identical": bool(py_out.equals(cc_out))})
            print(f"  {t} {model:8s} python {py_med * 1e3:7.1f} ms  cpp {cc_med * 1e3:6.2f} ms  "
                  f"x{py_med / cc_med:5.1f}  identical={rows[-1]['identical']}")
        del day
    speed = [r["speedup"] for r in rows]
    return {
        "method": (f"median of {runs} timed runs after one warm-up, time.perf_counter around "
                   "markout.lob.fills.simulate(engine=...); messages replayed = messages from the "
                   "first placement to the last order's end; the order set of report 02"),
        "machine": {"platform": platform.platform(), "machine": platform.machine(),
                    "processor": platform.processor(), "python": sys.version.split()[0]},
        "compiler": compiler_info(),
        "rows": rows,
        "all_identical": all(r["identical"] for r in rows),
        "speedup_min": min(speed), "speedup_max": max(speed),
        "speedup_median": statistics.median(speed),
        "fifo_speedup_median": statistics.median(r["speedup"] for r in rows if r["model"] == "fifo"),
    }


def draw_throughput(bench: dict):
    rows = bench["rows"]
    tickers = list(dict.fromkeys(r["ticker"] for r in rows))

    def draw():
        fig, axes = mp.subplots(1, 2, w=7.2, h=3.4)
        x = np.arange(len(tickers))
        fifo = {r["ticker"]: r for r in rows if r["model"] == "fifo"}
        py = [fifo[t]["python_msgs_per_s"] for t in tickers]
        cc = [fifo[t]["cpp_msgs_per_s"] for t in tickers]
        # a log axis gives bar lengths no meaning, so draw dots joined per ticker
        for xi, a, b in zip(x, py, cc):
            axes[0].plot([xi, xi], [a, b], color=mp.baseline_color(), lw=1.5, zorder=1)
        axes[0].plot(x, py, "o", color=mp.series(0), label="Python reference", zorder=3)
        axes[0].plot(x, cc, "o", color=mp.series(1), label="C++17 (-O3)", zorder=3)
        axes[0].set_yscale("log")
        axes[0].set_ylim(min(py) / 3, max(cc) * 12)
        axes[0].set_xticks(x, tickers)
        axes[0].set_xlim(-0.6, len(tickers) - 0.4)
        axes[0].set_ylabel("messages per second")
        mp.legend(axes[0], loc="upper center", ncols=2)
        mp.title(axes[0], "FIFO replay throughput", f"median of {len(rows[0]['cpp_runs_s'])} runs, log scale")
        width = 0.26
        top = 0.0
        for k, model in enumerate(MODELS):
            sp = [next(r for r in rows if r["ticker"] == t and r["model"] == model)["speedup"]
                  for t in tickers]
            top = max(top, max(sp))
            mp.bars(axes[1], x + (k - 1) * width, sp, width=width, color=mp.series(k), label=model)
        axes[1].set_ylim(0, top * 1.3)
        axes[1].set_xticks(x, tickers)
        axes[1].set_ylabel("speed-up (\u00d7)")
        mp.legend(axes[1], loc="upper left", ncols=3)
        mp.title(axes[1], "C++ speed-up by fill model", "Python time / C++ time, identical output")
        return fig

    return draw


def report_section(bench: dict) -> str:
    """Markdown for the "C++ core" section of report 02 (numbers from the JSON)."""
    from markout import reporting as rp

    prof = bench["profile"]
    loop = prof["rows"][0]
    fifo = [r for r in bench["rows"] if r["model"] == "fifo"]
    same = "bit-identical to the Python reference on every run" if bench["all_identical"] else \
        "NOT identical to the Python reference on every run (see the JSON)"
    lines = [
        "## C++ core", "",
        "The replay is sequential and keeps state (queues, ids behind us, timeouts), so it cannot "
        "be vectorised, and it is the part worth porting; the NumPy analysis around it is not. "
        f"cProfile on the Python reference ({prof['ticker']}, {prof['model']}) spends "
        f"{rp.num(prof['total_s'])} s in {rp.num(prof['function_calls'])} function calls. The loop body "
        f"itself is {100 * loop['share_of_total']:.0f}% of that; the rest is spread across millions "
        "of tiny list, dict and set operations. `cpp/queue_sim.cpp` ports the loop step for step "
        "(ordered maps of live orders by price, a min-heap of deadlines, a sorted id set for the "
        "orders behind us), and `cpp/bindings.cpp` exposes it to NumPy through pybind11. "
        f"`fills.simulate(engine=\"auto\")` uses it when it is built. Its output is {same}. That "
        "is checked here and in `tests/test_cpp_equivalence.py`: the five real days, seeded "
        "streams with hundreds of concurrent orders, and random `hypothesis` event streams "
        "(halts, crosses, timestamp ties, orders that cross the book or sit outside the visible "
        "levels).", "",
        f"Throughput on this machine ({bench['machine']['machine']}, {bench['machine']['platform']}), "
        f"{bench['method']}. The C++ speed-up is {bench['speedup_min']:.0f}–{bench['speedup_max']:.0f}× "
        f"(median {bench['fifo_speedup_median']:.0f}× for FIFO). Even the Python reference replays a "
        f"whole day in at most {max(r['python_median_s'] for r in bench['rows']):.2f} s, so the port "
        "matters when the same replay runs over many days, parameter grids or much larger order "
        "sets.", "",
        mp.picture("g_throughput", "Messages per second, Python versus C++, and C++ speed-up by fill model"), "",
        rp.table([{"ticker": r["ticker"], "model": r["model"], "orders": rp.num(r["orders"]),
                   "messages replayed": rp.num(r["messages_replayed"]),
                   "Python (ms)": "{:.1f}".format(1e3 * r["python_median_s"]),
                   "C++ (ms)": "{:.2f}".format(1e3 * r["cpp_median_s"]),
                   "Python (M msg/s)": "{:.2f}".format(r["python_msgs_per_s"] / 1e6),
                   "C++ (M msg/s)": "{:.1f}".format(r["cpp_msgs_per_s"] / 1e6),
                   "speed-up": "{:.0f}×".format(r["speedup"]),
                   "identical": "yes" if r["identical"] else "NO"} for r in bench["rows"]]), "",
        "<details><summary>cProfile of the Python reference (top rows by own time)</summary>", "",
        "```", prof["text"], "```", "", "</details>", "",
        "Build: `make cpp` (CMake + pybind11; `-O3 -ffp-contract=off`, no fast-math, so float "
        "results cannot drift). Rerun: `make bench`.", "",
    ]
    return "\n".join(lines)


def main() -> int:
    if not available():
        print("LOBSTER Parquet files missing: run `make lobster` first")
        return 1
    if not cpp_available():
        print("C++ extension not built: run `make cpp` first")
        return 1
    print("profiling the Python reference ...")
    prof = profile_python()
    print("timing both engines ...")
    bench = run()
    bench["profile"] = prof
    results.save(NAME, bench)
    mp.render("g_throughput", draw_throughput(bench))
    print(f"wrote reports/results/{NAME}.json and g_throughput; speed-up "
          f"{bench['speedup_min']:.0f}-{bench['speedup_max']:.0f}x, identical={bench['all_identical']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
