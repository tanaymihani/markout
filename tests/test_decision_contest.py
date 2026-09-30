"""Contest simulation: prize splitting, stake caps, fairness, Kelly's median, the report pipeline."""

import numpy as np
import pytest

from markout import plotting, reporting, results
from markout.decision import contest as ct


def test_prize_share_splits_ties():
    prizes = np.array([30000.0, 5000.0, 2500.0])
    above = np.array([0, 0, 2, 5, 1])
    ties = np.array([0, 1, 2, 0, 0])
    first, top3, prize = ct._prize_share(above, ties, prizes)
    assert list(first) == [1.0, 0.5, 0.0, 0.0, 0.0]
    assert list(top3) == pytest.approx([1.0, 1.0, 1 / 3, 0.0, 1.0])
    assert list(prize) == pytest.approx([30000.0, 17500.0, 2500 / 3, 0.0, 5000.0])


def test_round_gain_caps_stakes_and_all_in_takes_the_best_edge():
    f = np.array([[0.4, 0.4, 0.1]])
    ret = np.array([[1.0, -1.0, 3.0]])
    capped = ct._round_gain(f, ret, np.array([2.0]), np.array([False]))
    stakes = np.array([0.8, 0.8, 0.2]) / 1.8                  # 2x Kelly sums to 1.8, scaled back to the bankroll
    assert capped[0] == pytest.approx((stakes * ret[0]).sum())
    assert ct._round_gain(f, ret, np.array([0.5]), np.array([False]))[0] == pytest.approx(0.2 - 0.2 + 0.15)
    assert ct._round_gain(f, ret, np.array([1.0]), np.array([True]))[0] == pytest.approx(1.0)   # argmax f = market 0
    assert ct._round_gain(np.zeros((1, 3)), ret, np.array([1.0]), np.array([True]))[0] == 0.0   # no edge, no bet


SMALL = ct.ContestConfig(n_players=20, n_rounds=6, markets_per_round=3, n_sims=3000, sharp_share=0.0,
                         noisy_ratio_range=(0.7, 0.7), user_noise_ratio=0.7, field_allin_share=0.0,
                         field_multipliers=((1.0, 1.0),), seed=5)


def test_a_player_identical_to_the_field_has_fair_odds():
    sim = ct.simulate(SMALL, [ct.Policy("1x Kelly", "kelly", 1.0)])
    r = ct.summarize(sim, SMALL)[0]
    assert r["p_first"] == pytest.approx(1 / SMALL.n_players, abs=4 * r["p_first_se"])
    assert r["p_top3"] == pytest.approx(3 / SMALL.n_players, abs=4 * r["p_top3_se"])


def test_kelly_maximises_the_median_when_beliefs_are_accurate():
    cfg = SMALL.with_(n_players=5, user_noise_ratio=0.02, n_rounds=10, n_sims=4000)
    pols = [ct.Policy(f"{c:g}x", "kelly", c) for c in (0.5, 1.0, 2.0)] + [ct.Policy("all-in", "allin")]
    rows = {r["policy"]: r for r in ct.summarize(ct.simulate(cfg, pols), cfg)}
    assert rows["1x"]["median_bankroll"] > rows["0.5x"]["median_bankroll"]
    assert rows["1x"]["median_bankroll"] > rows["2x"]["median_bankroll"]
    wealth_ok = ct.simulate(cfg.with_(n_sims=300), [ct.Policy("all-in", "allin")])["policies"]["all-in"]["wealth"]
    assert np.all((wealth_ok == 0) | (wealth_ok > 1))            # all-in either busts or grows


def test_adaptive_rule_never_escalates_in_a_three_player_field():
    cfg = SMALL.with_(n_players=3, n_sims=500)
    base = ct.Policy("half", "kelly", 0.5)
    adapt = ct.Policy("half then all-in", "kelly", 0.5, 1.0, "allin")
    sim = ct.simulate(cfg, [base, adapt])
    # with only two rivals you always hold a top-3 place, so the rule stays at half Kelly
    assert np.array_equal(sim["policies"]["half"]["wealth"], sim["policies"]["half then all-in"]["wealth"])
    assert ct.late_rounds(adapt, 10) == 10 and ct.late_rounds(ct.Policy("x", late_share=0.2), 10) == 2


def test_simulation_is_deterministic():
    cfg = SMALL.with_(n_sims=200)
    a = ct.simulate(cfg, ct.spec_policies())
    b = ct.simulate(cfg, ct.spec_policies())
    for name in a["policies"]:
        assert np.array_equal(a["policies"][name]["wealth"], b["policies"][name]["wealth"])


def test_study_and_report_pipeline(tmp_path, monkeypatch):
    for mod, attr in ((reporting, "REPORTS"), (results, "RESULTS"), (plotting, "FIGURES")):
        d = tmp_path / attr.lower()
        d.mkdir()
        monkeypatch.setattr(mod, attr, d)
    monkeypatch.setattr(ct, "SENSITIVITY", {"n_players": (8, 12), "n_rounds": (4, 5)})
    monkeypatch.setattr(ct, "EXTRA_SCENARIOS", {"cautious field (no all-in, c <= 1)": {"field_allin_share": 0.0}})
    monkeypatch.setattr(ct, "SENS_BUDGET", 2e4)
    monkeypatch.setattr(ct, "SENS_SIMS", (100, 200))
    base = ct.ContestConfig(n_players=12, n_rounds=5, markets_per_round=3, n_sims=300, seed=11)
    C, fd = ct.run_study(base)
    C["first_generated_utc"] = C["generated_utc"]
    ct._figures(C, fd)
    path = ct.write_report(journal_path=tmp_path / "missing.csv", C=C)
    text = path.read_text()
    assert "The pre-registered policy" in text and "Pending" in text
    assert len(ct.scenarios(C)) == 1 + 2 + 1              # base once, two non-base sweep values, one extra
    rob = ct.robustness(C, [p.name for p in ct.spec_policies()])
    assert all(0 <= r["worst_ratio"] <= 1 for r in rob if np.isfinite(r["worst_ratio"]))
    assert (plotting.FIGURES / "p_sizing.png").exists() and (plotting.FIGURES / "p_sensitivity_dark.png").exists()
    assert results.load("predictions_cup")["contest"]["config"]["n_players"] == 12
