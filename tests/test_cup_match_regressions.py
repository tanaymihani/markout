"""Matcher regressions from the live dry run (Kalshi stand-ins vs live references).

Before the hard filters, every "must reject" pair below was a top suggestion: the other
team, another stat line, a different threshold, or identical text for a different game."""

import pytest

from markout.cup import match

MUST_REJECT = [
    ("Atlanta wins first 5 innings by over 1.5 runs?", "Houston wins first 5 innings by over 1.5 runs?", "entities"),
    ("Ty France: 1+ home runs?", "Ty France: 1+ hits + runs + RBIs?", "different measure"),
    ("San Diego wins by over 4.5 runs?", "San Diego wins by over 5.5 runs?", "thresholds differ"),
    ("Will over 4.5 goals be scored?", "Will over 4.5 goals be scored?", "no named entity"),
    ("Will over 4.5 goals be scored? Over 4.5 goals scored", "Will over 4.5 goals be scored? Over 4.5 goals scored",
     "no named entity"),
    ("Kirk Cousins: 1+ touchdowns", "Kirk Cousins: 1+ passing touchdowns", "different measure"),
    ("Niels Visker wins", "Constantin Bittoun Kouzmine / Niels Visker wins", "also names"),
    ("Will the WTI crude oil settlement price be above 90.99 USD/Bbl on Oct 3?",
     "Will the WTI crude oil settlement price be above 91.99 USD/Bbl on Oct 3?", "thresholds differ"),
]
MUST_REJECT += [  # from the Polymarket stand-in dry run
    ("Will there be no change in Fed interest rates after the October 2026 meeting?",
     "Will the Fed decrease interest rates by 25 bps after the October 2026 meeting?", "thresholds differ"),
    ("Will the Fed increase interest rates by 25 bps after the October 2026 meeting?",
     "Will the Fed decrease interest rates by 25 bps after the October 2026 meeting?", "different measure"),
    ("US announces end of Iranian blockade by October 31, 2026?",
     "US announces end of Iranian blockade by September 30, 2026?", "dates differ"),
    ("Will Ethereum reach $2,800 in September?", "Will Ethereum reach $2,800 on September 29?", "dates differ"),
    ("Will the highest temperature in Shanghai be 24°C on September 30?",
     "Will the lowest temperature in Shanghai be 24°C on September 30?", "different measure"),
]
MUST_REJECT += [
    ("Will the Republican Party win the Senate in 2026?", "2028 Senate winner Republican party", "years differ"),
    ("Will the highest temperature in Paris be 24°C on September 30?",
     "Will the highest temperature in Paris be 20°C on September 30?", "thresholds differ"),
]
MUST_REJECT += [  # reversals: the same names, the opposite question
    ("Will Benjamin Netanyahu be the next Prime Minister of Israel?",
     "[ACX 2026] Will Benjamin Netanyahu cease to be Prime Minister of Israel during 2026?", "different measure"),
    ("Will Apple miss Q3 EPS estimates?", "Apple Q3 earnings: beat EPS consensus?", "different measure"),
]
MUST_REJECT += [  # a sibling market in the same event: same speech, a different word
    ("What will Donald Trump say during Hispanic Heritage Month Celebration? Rubio",
     "What will Trump say during Hispanic Heritage Month Celebration? Donald Trump - Hispanic Heritage Month "
     "Celebration ICE", "entities"),
]
MUST_MATCH = [
    ("Will Republicans win the Senate in 2026?", "Which party will win the Senate in 2026? Republican"),
    ("Will the Chiefs beat the Bills on Sunday?", "Chiefs vs. Bills Chiefs"),
    ("Will Apple beat Q3 EPS estimates?", "Apple Q3 earnings: beat EPS consensus?"),
    ("Will Taylor Swift release a new album before November?", "New Taylor Swift album before November?"),
    ("Will the Republican Party control the Senate after the 2026 Midterm elections?",
     "Which party will win the U.S. Senate? In 2026 Republican Party"),
    ("Will Alexia Putellas win the 2026 Women's Ballon d'Or?", "Women's Ballon d'Or Winner 2026 Alexia Putellas"),
    ("Will Luiz Inácio Lula da Silva win the 2026 Brazilian presidential election?",
     "Brazil Presidential election winner? In The next presidential election Luiz Inácio Lula da Silva"),
]


@pytest.mark.parametrize("a,b,why", MUST_REJECT)
def test_rejects_look_alikes(a, b, why):
    r = match.hard_reject(a, b)
    assert r is not None and why in r


@pytest.mark.parametrize("a,b", MUST_MATCH)
def test_keeps_true_matches(a, b):
    assert match.hard_reject(a, b) is None


def test_thresholds_ignore_years_and_days():
    assert match.thresholds("Will turnout exceed 55% in the 2026 election on Nov 3?") == {"55"}
    assert match.thresholds("Over 4.5 goals, 1+ touchdowns, above $1,000") == {"4.5", "1", "1000"}


def test_yes_contract_on_a_no_reference_is_inverted():
    from datetime import datetime, timezone

    from markout.cup.sources.base import ExternalQuote
    from markout.cup.types import Contract, Market

    m = Market("m", "Will Ethereum reach $3,000 in October?", "markets",
               (Contract("y", "m", "Yes"), Contract("n", "m", "No")))
    ref = ExternalQuote("polymarket", "1", "Will Ethereum reach $3,000 in October?", "No", 0.8)
    yes, no = match.score(m, m.contracts[0], ref), match.score(m, m.contracts[1], ref)
    assert yes.invert and yes.p == 0.19999999999999996 or abs(yes.p - 0.2) < 1e-12
    assert not no.invert and no.p == 0.8
