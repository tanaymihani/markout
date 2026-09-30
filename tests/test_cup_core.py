"""Predictions Cup bot: sizing, engine, matcher, paper exchange."""

from datetime import datetime, timedelta, timezone

import pytest

from markout.cup import engine, match
from markout.cup import proposals as P
from markout.cup.api import ContestAPI, RateLimiter
from markout.cup.match import Candidate
from markout.cup.paper import PaperContest, PaperMarketSpec
from markout.cup.sources.base import ExternalQuote
from markout.cup.types import Account, Book, Contract, Market

T0 = datetime(2026, 10, 1, 16, tzinfo=timezone.utc)


def mk_market(mid="m", title="Will X happen?", names=("Yes", "No"), hours=48, category="other"):
    cs = tuple(Contract(f"{mid}-{n.lower()}", mid, n) for n in names)
    return Market(mid, title, category, cs, T0 + timedelta(hours=hours - 1), T0 + timedelta(hours=hours))


def book(cid, bid, ask, size=100.0):
    return Book(cid, ((bid, size),), ((ask, size),), ts=T0)


def ref(p, liq=1e6, bid=None, ask=None, source="polymarket", kind="market"):
    return Candidate(ExternalQuote(source, "id", "q", "Yes", p, kind, bid, ask, liq, liq / 5), 1.0, False, "t")


# ---------------------------------------------------------------- engine
def test_engine_pools_toward_trusted_reference_and_needs_one():
    b = book("c", 0.48, 0.52)
    assert engine.estimate(b, []) is None
    est = engine.estimate(b, [ref(0.70)])
    assert 0.5 < est.p < 0.70  # moved toward the reference, never all the way (the contest price counts)
    weak = engine.estimate(b, [ref(0.70, liq=0)])
    assert weak.p < est.p  # an illiquid reference moves it less


def test_band_measures_reference_disagreement_not_contest_mispricing():
    far = engine.estimate(book("c", 0.18, 0.22), [ref(0.70, bid=0.69, ask=0.71)])
    assert far.hi - far.lo == pytest.approx(2 * 0.02)  # min band: the one reference is tight
    split = engine.estimate(book("c", 0.48, 0.52), [ref(0.40), ref(0.80)])
    assert split.hi - split.lo > 0.2  # the references disagree


# ---------------------------------------------------------------- sizing
def ests(*triples):
    out, markets, books = {}, [], {}
    for i, (p, ask, confirmed) in enumerate(triples):
        m = mk_market(f"m{i}")
        c = m.contracts[0]
        markets.append(m)
        books[c.id] = book(c.id, ask - 0.02, ask, size=1e9)  # deep: these tests are about sizing
        out[c.id] = (engine.Estimate(p, p - 0.02, p + 0.02, ask - 0.01), confirmed, ["k"])
    return markets, books, out


def test_screen_requires_edge_and_band_above_ask():
    markets, books, est = ests((0.60, 0.55, True), (0.56, 0.55, True), (0.58, 0.57, True))
    got = P.screen(markets, books, est, P.Config(min_edge=0.03), T0)
    assert [c.contract_id for c in got] == ["m0-yes"]


def test_prize_mode_is_all_in_on_the_best_kelly_outside_top3():
    markets, books, est = ests((0.70, 0.50, True), (0.60, 0.50, True))
    acct = Account(cash=1000, equity=1000, rank=1, n_players=100, start_cash=1000)  # tied at the start
    assert not acct.in_top3
    props = P.build(markets, books, est, acct, P.Config(mode="prize"), T0)
    stakes = {p.contract_id: p.stake for p in props}
    assert stakes["m0-yes"] == pytest.approx(1000.0) and stakes["m1-yes"] == 0.0
    assert props[0].qty == 2000  # floor(1000 / 0.50)


def test_prize_mode_in_top3_and_steady_mode_use_half_kelly():
    markets, books, est = ests((0.70, 0.50, True))
    lead = Account(cash=2000, equity=2000, rank=2, n_players=100, start_cash=1000)
    assert lead.in_top3
    f = (0.70 - 0.50) / (1 - 0.50)
    for acct, cfg in [(lead, P.Config(mode="prize")), (lead, P.Config(mode="steady", max_frac_per_market=1.0))]:
        p = P.build(markets, books, est, acct, cfg, T0)[0]
        assert p.stake == pytest.approx(0.5 * f * 2000, abs=0.5)
    capped = P.build(markets, books, est, lead, P.Config(mode="steady", max_frac_per_market=0.1), T0)[0]
    assert capped.stake <= 200 + 1e-9


def test_unconfirmed_mapping_and_late_resolution_are_flagged():
    markets, books, est = ests((0.70, 0.50, False))
    late = P.Config(contest_end=T0)  # the market resolves after this "contest end"
    p = P.build(markets, books, est, Account(1000, 1000), late, T0)[0]
    assert "mapping not confirmed" in p.flags and "resolves after the contest ends" in p.flags


# ---------------------------------------------------------------- matcher
def q(question, outcome="Yes", p=0.6, end=None):
    return ExternalQuote("polymarket", question[:8], question, outcome, p, end=end)


def test_matcher_prefers_the_same_event_and_penalizes_different_numbers():
    m = mk_market(title="Will Republicans win the Senate in 2026?", names=("Yes", "No"))
    good = match.score(m, m.contracts[0], q("Which party will win the Senate in 2026?", "Republican"))
    house = match.score(m, m.contracts[0], q("Which party will win the House in 2026?", "Republican"))
    assert good.score > house.score
    runs = mk_market(title="Will the Yankees win by over 4.5 runs?")
    same = match.score(runs, runs.contracts[0], q("Yankees win by over 4.5 runs"))
    other = match.score(runs, runs.contracts[0], q("Yankees win by over 5.5 runs"))
    assert same.score - other.score >= 0.25


def test_no_contract_inverts_a_yes_reference():
    m = mk_market(title="Will it rain in Boston on Friday?")
    c = match.score(m, m.contracts[1], q("Will it rain in Boston on Friday?", "Yes", p=0.3))
    assert c.invert and c.p == pytest.approx(0.7)


@pytest.mark.parametrize("title,tk,k,above", [
    ("Will Apple (AAPL) close above $250 on Oct 31?", "AAPL", 250.0, True),
    ("Will Tesla close below $300 this week?", "TSLA", 300.0, False),
    ("Will NVIDIA stock finish over $1,000?", "NVDA", 1000.0, True),
])
def test_price_questions_parse(title, tk, k, above):
    pq = match.parse_price_question(mk_market(title=title))
    assert pq is not None and (pq.ticker, pq.strike, pq.above) == (tk, k, above)


def test_non_price_questions_do_not_parse():
    assert match.parse_price_question(mk_market(title="Will the Blue Party win the governor race?")) is None


# ---------------------------------------------------------------- paper exchange
def paper(truth=0.6, crowd=0.5, exclusive=False, names=("Yes", "No"), outcome=None):
    m = mk_market("m1", names=names, hours=24)
    specs = [PaperMarketSpec(m, {c.id: truth if i == 0 else 1 - truth for i, c in enumerate(m.contracts)},
                             {c.id: crowd if i == 0 else 1 - crowd for i, c in enumerate(m.contracts)},
                             spread=0.04, depth=50.0, exclusive=exclusive, outcome=outcome)]
    clock = [T0]
    api = PaperContest(specs, start_cash=100.0, seed=1, clock=lambda: clock[0], n_field=10)
    return api, m, clock


def test_paper_is_a_contest_api_and_fills_at_the_book():
    api, m, _ = paper()
    assert isinstance(api, ContestAPI)
    cid = m.contracts[0].id
    b = api.books([cid])[cid]
    assert b.best_ask == pytest.approx(0.52) and b.best_bid == pytest.approx(0.48)
    o = api.place_order(cid, "buy", 0.53, 60)  # 50 at 0.52 and 10 at 0.53
    assert o.status == "filled" and o.filled_qty == 60
    assert api.cash == pytest.approx(100 - 50 * 0.52 - 10 * 0.53)
    assert api.positions()[0].qty == 60


def test_paper_rejects_what_it_should():
    api, m, _ = paper()
    cid = m.contracts[0].id
    assert api.place_order(cid, "buy", 0.52, 10_000).status == "rejected"  # not enough cash
    assert api.place_order(cid, "sell", 0.48, 1).status == "rejected"  # long only


def test_paper_resolution_pays_winners():
    api, m, clock = paper(outcome={"m1-yes": 1, "m1-no": 0})
    cid = m.contracts[0].id
    api.place_order(cid, "buy", 0.52, 50)
    cash_after_buy = api.cash
    clock[0] = T0 + timedelta(hours=25)
    api.step()
    assert "m1" in api.resolved and api.cash == pytest.approx(cash_after_buy + 50)
    assert api.markets() == []


def test_exclusive_markets_have_exactly_one_winner():
    for seed in range(5):
        api, m, clock = paper(exclusive=True, names=("A", "B", "C"))
        api.rng = __import__("numpy").random.default_rng(seed)
        clock[0] = T0 + timedelta(hours=25)
        api.step()
        assert sum(api.resolved["m1"].values()) == 1


def test_rate_limiter_spaces_calls():
    import time

    lim = RateLimiter(rate=20.0, burst=1)
    t = time.monotonic()
    for _ in range(5):
        lim.acquire()
    assert time.monotonic() - t >= 4 / 20 * 0.9


def test_orders_are_sized_to_the_book_and_never_rest():
    b = Book("c", ((0.40, 50.0),), ((0.50, 30.0), (0.54, 30.0), (0.60, 30.0)), ts=T0)
    qty, limit, cost = P.fill_plan(b, max_price=0.57, budget=1_000.0)  # p 0.60 - min edge 0.03
    assert (qty, limit) == (60, 0.54) and cost == pytest.approx(30 * 0.50 + 30 * 0.54)
    qty, limit, cost = P.fill_plan(b, max_price=0.57, budget=20.0)  # money runs out first
    assert qty == 30 + 9 and cost <= 20.0 + 1e-9  # 30 at 0.50 = 15, then floor(5 / 0.54) = 9


def test_manifold_quotes_parse_and_weigh_less_than_real_money():
    from markout.cup.sources import manifold

    q = manifold.parse_market({"id": "a1", "question": "Will X happen?", "outcomeType": "BINARY", "probability": 0.3,
                               "closeTime": 1_800_000_000_000, "uniqueBettorCount": 500, "url": "u"})
    assert q is not None and q.source == "manifold" and q.p == 0.3
    assert manifold.parse_market({"outcomeType": "MULTIPLE_CHOICE", "probability": None}) is None
    play = engine.source_weight(Candidate(q, 1.0, False, "t"))
    real = engine.source_weight(ref(0.3, liq=1e5))
    assert play < 1.0 < real
