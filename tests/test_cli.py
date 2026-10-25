"""The `markout` command gives the same numbers as the library, on files a user brings."""

import json

import numpy as np
import pandas as pd
import pytest

from markout import cli
from markout.evaluation import deflated_sharpe, pbo
from markout.lob import lobster
from markout.lob.markouts import order_markouts, sample_orders, summarize_markouts
from markout.lob.synthetic import random_stream


def run(capsys, *argv):
    assert cli.main(list(argv)) == 0
    return json.loads(capsys.readouterr().out)


@pytest.fixture(scope="module")
def lobster_csv(tmp_path_factory):
    """A random but valid LOBSTER day, written the way LOBSTER writes it."""
    m, b = random_stream(np.random.default_rng(5), n_events=3000, levels=5)
    d = tmp_path_factory.mktemp("lobster")
    msg, book = d / "TEST_2026-01-02_34200000_57600000_message_5.csv", d / "TEST_orderbook_5.csv"
    with open(msg, "w") as f:
        for r in zip(m.time.tolist(), m.type, m.order_id, m.size, m.price, m.direction):
            f.write(f"{r[0]!r},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]}\n")
    np.savetxt(
        book, np.stack([b.ask_price, b.ask_size, b.bid_price, b.bid_size], axis=2).reshape(len(b), -1), "%d", ","
    )
    return msg, book, m, b


def test_reading_lobster_csv_is_exact(lobster_csv):
    msg, book, m, b = lobster_csv
    day = lobster.read_csv(msg, book)
    assert day.ticker == "TEST" and day.book.levels == 5
    for f in ("time", "type", "order_id", "size", "price", "direction"):
        assert (
            np.array_equal(getattr(day.messages, f), getattr(m, f))
            and getattr(day.messages, f).dtype == getattr(m, f).dtype
        )
    for f in ("ask_price", "ask_size", "bid_price", "bid_size"):
        assert np.array_equal(getattr(day.book, f), getattr(b, f))


def test_reading_misaligned_files_fails(lobster_csv, tmp_path):
    msg, book, _, _ = lobster_csv
    short = tmp_path / "short.csv"
    short.write_text("\n".join(book.read_text().splitlines()[:-3]) + "\n")
    with pytest.raises(ValueError, match="not aligned"):
        lobster.read_csv(msg, short)


def test_fills_on_user_files_match_the_library(capsys, lobster_csv):
    msg, book, _, _ = lobster_csv
    out = run(
        capsys, "fills", str(msg), str(book), "--every", "2", "--wait", "10", "--start", "0", "--end", "99999", "--json"
    )
    day = lobster.read_csv(msg, book)
    t0, t1 = float(day.messages.time[0]), float(day.messages.time[-1])
    expect = summarize_markouts(order_markouts(day, sample_orders(day, every=2, max_wait=10, session=(t0, t1))))
    assert out["orders"] > 100 and [r["model"] for r in out["summary"]] == ["touch", "fifo", "through"]
    for got, want in zip(out["summary"], expect):
        assert got["p_fill"] == pytest.approx(want["p_fill"]) and got["n_orders"] == want["n_orders"]
    touch, fifo, through = out["summary"]
    assert touch["p_fill"] >= fifo["p_fill"] >= through["p_fill"]  # each model is stricter than the last


def test_fills_explains_bad_input(lobster_csv):
    msg, book, _, _ = lobster_csv
    with pytest.raises(SystemExit, match="unknown fill model"):
        cli.main(["fills", str(msg), str(book), "--models", "touch,magic"])
    with pytest.raises(SystemExit, match="no decision times fit"):
        cli.main(["fills", str(msg), str(book), "--start", "0", "--end", "34230"])
    with pytest.raises(SystemExit, match="MESSAGES.csv BOOK.csv"):
        cli.main(["fills"])


def test_sharpe_matches_the_library(capsys, tmp_path):
    r = np.random.default_rng(1).normal(0.001, 0.01, 500)
    pd.DataFrame({"date": range(500), "pnl": r}).to_csv(tmp_path / "pnl.csv", index=False)
    out = run(capsys, "sharpe", str(tmp_path / "pnl.csv"), "--column", "pnl", "--boot", "500", "--json")
    d = deflated_sharpe(r, n_trials=1)
    assert out["sharpe"] == pytest.approx(d["sr"]) and out["dsr"] == pytest.approx(d["psr_0"])
    assert out["sharpe_annualized"] == pytest.approx(d["sr"] * np.sqrt(252))
    lo, hi = out["ci95_annualized"]
    assert lo < out["sharpe_annualized"] < hi
    deflated = run(
        capsys,
        "sharpe",
        str(tmp_path / "pnl.csv"),
        "--column",
        "pnl",
        "--trials",
        "100",
        "--sr-var",
        "0.002",
        "--boot",
        "200",
        "--json",
    )
    assert deflated["dsr"] < deflated["psr_0"]  # searching 100 variants raises the bar
    with pytest.raises(SystemExit, match="--sr-var"):
        cli.main(["sharpe", str(tmp_path / "pnl.csv"), "--column", "pnl", "--trials", "10"])
    with pytest.raises(SystemExit, match="pick one with --column"):
        cli.main(["sharpe", str(tmp_path / "pnl.csv")])


def test_pbo_of_pure_noise_is_a_coin_flip(capsys, tmp_path):
    X = np.random.default_rng(2).normal(0, 1, (320, 12))
    pd.DataFrame(X, columns=[f"s{i}" for i in range(12)]).assign(day=range(320)).to_csv(tmp_path / "t.csv", index=False)
    out = run(capsys, "pbo", str(tmp_path / "t.csv"), "--index", "day", "--blocks", "8", "--json")
    lib = pbo(pd.DataFrame(X, columns=[f"s{i}" for i in range(12)]), n_blocks=8)
    assert out["pbo"] == pytest.approx(lib["pbo"]) and 0.2 < out["pbo"] < 0.8
    assert out["n_strategies"] == 12 and out["T"] == 320


def test_kelly(capsys):
    out = run(capsys, "kelly", "--p", "0.62", "--price", "0.55", "--json")
    assert out["side"] == "YES" and out["kelly"] == pytest.approx(0.07 / 0.45)
    out = run(capsys, "kelly", "--p", "0.40", "--price", "0.55", "--fraction", "0.5", "--bankroll", "1000", "--json")
    assert (
        out["side"] == "NO"
        and out["kelly"] == pytest.approx(0.15 / 0.55)
        and out["stake"] == pytest.approx(500 * 0.15 / 0.55)
    )
    assert run(capsys, "kelly", "--p", "0.55", "--price", "0.55", "--json")["side"] is None
    with pytest.raises(SystemExit):
        cli.main(["kelly", "--p", "0.5", "--price", "1.5"])


def test_status_and_clock(capsys):
    out = run(capsys, "status", "--json")
    assert set(out) >= {"version", "lobster_sample", "cpp_extension", "results"}
    assert cli._clock("09:35") == 34_500 and cli._clock("15:55:30") == 57_330 and cli._clock("34500") == 34_500


def test_version_is_defined_once(capsys):
    import tomllib

    from markout import __version__
    from markout.paths import ROOT

    assert tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"] == __version__
    assert cli.main(["--version"]) == 0 and capsys.readouterr().out.strip() == __version__
