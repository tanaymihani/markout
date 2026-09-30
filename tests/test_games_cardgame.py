"""Card market game: engine rules, Bayes benchmark vs exact enumeration, --auto mode."""

import numpy as np
import pytest

from markout.games import cardgame as cg

SMALL = cg.Config(n_cards=3, bots_per_round=2, p_informed=0.5, max_width=3.0,
                  deck=(1, 2, 3, 4, 5, 6, 7, 8))


def test_prior_mean():
    cfg = cg.Config()
    assert cg.prior_mean(cfg, []) == pytest.approx(5 * 7.0)
    assert cg.prior_mean(cfg, [13, 13]) == pytest.approx(26 + 3 * (364 - 26) / 50)
    assert cg.prior_mean(cfg, [1, 2, 3, 4, 5]) == 15


def test_posterior_without_trades_is_the_prior():
    cfg = cg.Config()
    out = cg.posterior_mean(cfg, [7, 12], [], n_samples=40_000, rng=np.random.default_rng(0))
    assert out["mean"] == pytest.approx(cg.prior_mean(cfg, [7, 12]), abs=4 * out["se"] + 1e-9)


@pytest.mark.parametrize("revealed, events", [
    ([], [cg.Event(0, 11.0, 14.0, 1), cg.Event(0, 11.0, 14.0, 0)]),
    ([6], [cg.Event(0, 11.0, 14.0, 1), cg.Event(0, 11.0, 14.0, 0),
           cg.Event(1, 14.0, 17.0, -1), cg.Event(1, 14.0, 17.0, 0)]),
    ([6, 2], [cg.Event(0, 11.0, 14.0, 1), cg.Event(0, 11.0, 14.0, 0),
              cg.Event(1, 12.0, 15.0, -1), cg.Event(1, 12.0, 15.0, 1)]),
])
def test_monte_carlo_posterior_matches_exact_enumeration(revealed, events):
    exact = cg.exact_posterior_mean(SMALL, revealed, events)
    mc = cg.posterior_mean(SMALL, revealed, events, n_samples=200_000,
                           rng=np.random.default_rng(1))
    assert mc["mean"] == pytest.approx(exact, abs=4 * mc["se"] + 1e-6)
    # and the trades moved the estimate away from the prior
    assert abs(exact - cg.prior_mean(SMALL, revealed)) > 0.05


def test_informed_bot_trades_only_when_the_quote_is_wrong():
    cfg = cg.Config()
    value = cg.bot_value(cfg, [], 13, 5)          # knows a King: E[S] = 13 + 4·mean(rest)
    assert value == pytest.approx(13 + 4 * (364 - 13) / 51)
    assert cg.bot_action(value, value - 5, value - 1) == 1      # ask too low → buys
    assert cg.bot_action(value, value + 1, value + 5) == -1     # bid too high → sells
    assert cg.bot_action(value, value - 2, value + 2) == 0      # quote brackets its value


def test_engine_rules_and_pnl_accounting():
    cfg = cg.Config(n_cards=3, bots_per_round=2, max_width=4.0)
    d = cg.Deal((10, 2, 7), ((cg.Bot(False, 0.0, 1), cg.Bot(False, 0.0, -1)),
                             (cg.Bot(True, 0.0, 1), cg.Bot(False, 0.0, 1)),
                             (cg.Bot(False, 0.0, -1), cg.Bot(True, 0.9, 1))))
    g = cg.Game(cfg, d)
    with pytest.raises(ValueError):
        g.play_round(20.0, 25.0)                  # too wide
    with pytest.raises(ValueError):
        g.play_round(21.0, 20.0)                  # crossed
    ev, card = g.play_round(18.0, 22.0)
    assert card == 10 and [e.action for e in ev] == [1, -1]
    ev, card = g.play_round(24.0, 27.0)           # informed bot sees the 2: value 12 + 2 + 7·… < 24
    assert card == 2 and ev[0].action == -1 and ev[1].action == 1
    ev, card = g.play_round(18.0, 20.0)           # informed bot now knows S = 19 exactly
    assert card == 7 and ev[0].action == -1 and ev[1].action == 0
    assert g.done
    s = 19
    expected = [(22 - s) + (s - 18), (s - 24) + (27 - s), (s - 18)]
    assert g.pnl_by_round() == pytest.approx(expected)


def test_auto_game_is_deterministic_and_consistent():
    cfg = cg.Config()
    d = cg.deal(cfg, np.random.default_rng(5))
    g1, r1 = cg.play(cfg, d, cg.bayes_quoter(5_000, seed=5), n_samples=5_000, seed=5)
    g2, r2 = cg.play(cfg, d, cg.bayes_quoter(5_000, seed=5), n_samples=5_000, seed=5)
    assert [r.pnl for r in r1] == [r.pnl for r in r2]
    for r in r1:   # the Bayes quoter centres its market on the benchmark mid
        assert 0.5 * (r.bid + r.ask) == pytest.approx(r.bayes_mid)
        assert r.ask - r.bid == pytest.approx(cfg.max_width)
    assert sum(r.pnl for r in r1) == pytest.approx(sum(g1.pnl_by_round()))


def test_cli_auto_mode_runs():
    out = []
    assert cg.main(["--auto", "--seed", "11", "--samples", "4000"], write=out.append) == 0
    text = "\n".join(out)
    assert "Bayes mid" in text and "total PnL" in text


def test_cli_interactive_mode_with_scripted_input():
    # one unparsable answer and one too-wide market are rejected, then five valid rounds
    answers = iter(["oops", "30 38", "33 37", "31 35", "30 34", "28 32", "27 31"])
    out = []
    assert cg.main(["--seed", "4", "--samples", "3000"], read=lambda _: next(answers),
                   write=out.append) == 0
    text = "\n".join(out)
    assert text.count("invalid market") == 2
    assert "The Bayes quoter on the same deal" in text


def test_cli_exits_cleanly_on_end_of_input():
    def eof(_):
        raise EOFError
    out = []
    assert cg.main(["--seed", "4", "--samples", "3000"], read=eof, write=out.append) == 1
    assert "abandoned" in "\n".join(out)
