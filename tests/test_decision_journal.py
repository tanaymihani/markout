"""Decision journal: writing, PnL arithmetic, scoring and the CLI, on synthetic journals."""

import numpy as np
import pandas as pd
import pytest

from markout import plotting, reporting, results
from markout.decision import journal, kelly


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Send every report, result and figure to a temporary directory."""
    for mod, attr in ((reporting, "REPORTS"), (results, "RESULTS"), (plotting, "FIGURES")):
        d = tmp_path / attr.lower()
        d.mkdir()
        monkeypatch.setattr(mod, attr, d)
    return tmp_path


def synthetic_journal(path, n=400, seed=3, stake_mult=0.5):
    """Beliefs closer to the truth than the market's prices; stakes at a fixed Kelly multiple."""
    rng = np.random.default_rng(seed)
    logit = rng.normal(0, 1, n)
    truth = 1 / (1 + np.exp(-logit))
    q = np.clip(1 / (1 + np.exp(-(logit + rng.normal(0, 0.6, n)))), 0.03, 0.97)
    p = np.clip(1 / (1 + np.exp(-(logit + rng.normal(0, 0.2, n)))), 0.03, 0.97)
    y = (rng.random(n) < truth).astype(int)
    journal.init(path)
    bankroll = 1000.0
    for i in range(n):
        side, f = kelly.best_side(p[i], q[i])
        journal.add(path, f"m{i}", side, p[i], q[i], stake=round(stake_mult * f * bankroll, 6), bankroll=bankroll,
                    timestamp=f"2026-10-{1 + i % 28:02d}T12:00:{i % 60:02d}+00:00")
        journal.resolve(path, f"m{i}", int(y[i]))
    return path


def test_init_refuses_to_overwrite(tmp_path):
    path = journal.init(tmp_path / "journal.csv")
    assert path.read_text().strip() == ",".join(journal.COLUMNS)
    with pytest.raises(FileExistsError):
        journal.init(path)
    journal.init(path, force=True)


def test_add_computes_kelly_and_resolve_sets_outcome(tmp_path):
    path = journal.init(tmp_path / "j.csv")
    row = journal.add(path, "fed-cut", "YES", 60, 50, stake=40, bankroll=1000, question="Fed cuts?", reason="dots")
    assert float(row["p_belief"]) == 0.6 and float(row["price_q"]) == 0.5
    assert float(row["kelly_fraction"]) == pytest.approx(0.2)          # (0.6 - 0.5) / (1 - 0.5)
    journal.add(path, "fed-cut", "NO", 0.6, 0.5, stake=0, bankroll=960)  # against the belief: Kelly 0
    assert journal.resolve(path, "fed-cut", 1) == 2
    assert journal.resolve(path, "fed-cut", 0) == 0                      # already resolved
    df = journal.load(path)
    assert list(df["kelly_fraction"]) == [0.2, 0.0]
    assert list(df["outcome"]) == [1, 1] and (df["resolved_at"] != "").all()
    with pytest.raises(ValueError):
        journal.add(path, "x", "YES", 0.5, 0.4, stake=2000, bankroll=1000)
    with pytest.raises(ValueError):
        journal.resolve(path, "fed-cut", 2)


def test_as_prob_reads_percent_and_rejects_certainty():
    assert journal.as_prob("63") == pytest.approx(0.63)
    assert journal.as_prob(0.63) == pytest.approx(0.63)
    for bad in (0, 100, -0.1, 150):
        with pytest.raises(ValueError):
            journal.as_prob(bad)


def test_pnl_arithmetic_by_hand():
    df = pd.DataFrame({"side": ["YES", "YES", "NO", "NO"], "p_belief": [0.5] * 4, "price_q": [0.4] * 4,
                       "outcome": [1, 0, 1, 0], "stake": [10.0, 10.0, 12.0, 12.0], "bankroll_before": [100.0] * 4})
    r = journal.pnl_columns(df)
    assert list(r["pnl"]) == pytest.approx([15.0, -10.0, -12.0, 8.0])
    assert list(r["pnl_predicted"]) == pytest.approx([2.5, 2.5, -2.0, -2.0])   # 10(0.5/0.4-1), 12(0.5/0.6-1)


def test_scores_on_a_synthetic_journal(tmp_path):
    df = journal.load(synthetic_journal(tmp_path / "j.csv"))
    s = journal.score(df)
    assert s["n_resolved"] == len(df) == 400
    # the synthetic beliefs are closer to the truth than the prices, so they should score better
    assert s["brier_you"] < s["brier_market"] and s["brier_diff_t"] < -2
    assert s["logloss_you"] < s["logloss_market"]
    assert "Yes" in journal.verdict(s)
    # manual Brier
    y, p = df["outcome"].to_numpy(float), df["p_belief"].to_numpy(float)
    assert s["brier_you"] == pytest.approx(np.mean((p - y) ** 2))
    assert sum(b["n"] for b in s["calibration_you"]) == 400
    # luck decomposition is an identity, and stakes were exactly half Kelly
    assert s["pnl_realized"] == pytest.approx(s["pnl_predicted"] + s["luck"])
    assert s["median_kelly_multiple"] == pytest.approx(0.5, abs=1e-3)
    assert s["share_over_kelly"] == 0.0


def test_verdict_needs_enough_rows(tmp_path):
    path = journal.init(tmp_path / "j.csv")
    journal.add(path, "a", "YES", 0.7, 0.5, 10, 100)
    journal.resolve(path, "a", 1)
    s = journal.score(journal.load(path))
    assert s["n_resolved"] == 1 and journal.verdict(s).startswith("Too few")
    assert journal.score(journal.load(journal.init(tmp_path / "empty.csv"))) is None


def test_part2_text_pending_and_scored(tmp_path):
    pending = journal.part2_markdown(None, 3, "pending")
    assert "Pending" in pending and "3 trades logged" in pending
    s = journal.score(journal.load(synthetic_journal(tmp_path / "j.csv", n=60)))
    text = journal.part2_markdown(s, 60, "interim", assumed_skill=0.012)
    assert "Did you beat the market?" in text and "Interim" in text and "1.2%" in text


def test_cli_end_to_end(sandbox):
    path = sandbox / "journal.csv"
    assert journal.main(["--journal", str(path), "init"]) == 0
    assert journal.main(["--journal", str(path), "add", "--market", "nfl-1", "--side", "NO", "--p", "0.3",
                         "--q", "45", "--stake", "25", "--bankroll", "1000", "--reason", "injury news"]) == 0
    assert journal.main(["--journal", str(path), "score"]) == 0
    report = (reporting.REPORTS / "05_predictions_cup.md").read_text()
    assert "Pending" in report and "1 trade logged" in report
    assert journal.main(["--journal", str(path), "resolve", "--market", "nfl-1", "--outcome", "0"]) == 0
    assert journal.main(["--journal", str(path), "score"]) == 0
    report = (reporting.REPORTS / "05_predictions_cup.md").read_text()
    assert "Did you beat the market?" in report and "Too few resolved trades" in report
    saved = results.load("predictions_cup")
    assert saved["journal"]["scores"]["n_resolved"] == 1
    assert saved["journal"]["scores"]["pnl_realized"] == pytest.approx(25 * 0.45 / 0.55)
    assert (plotting.FIGURES / "p_journal_calibration.png").exists()
