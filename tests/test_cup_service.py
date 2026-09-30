"""Predictions Cup bot end to end on the paper exchange, and the desk's web security."""

import json
import threading
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

import pytest

from markout.cup import proposals as P
from markout.cup import risk
from markout.cup.demo import SCENARIO, ManualClock, reviewer
from markout.cup.paper import PaperContest, load_scenario
from markout.cup.refs import References
from markout.cup.service import Bot
from markout.cup.sources.static import from_scenario
from markout.cup.store import Store
from markout.cup.web import PAGE, Desk
from markout.decision import journal

T0 = datetime(2026, 10, 1, 16, tzinfo=timezone.utc)


@pytest.fixture
def bot(tmp_path, monkeypatch):
    monkeypatch.setattr(risk, "STOP_FILE", tmp_path / "STOP")
    clock = ManualClock(T0)
    specs = load_scenario(SCENARIO, T0)
    api = PaperContest(specs, start_cash=10_000, seed=3, clock=clock)
    static = {cid: from_scenario([r for r in s.references if r["contract_id"] == cid])
              for s in specs for cid in {r["contract_id"] for r in s.references}}
    b = Bot(api, Store(tmp_path / "t.db"), References(live=False, static=static), P.Config(mode="steady"),
            journal_path=tmp_path / "journal.csv", clock=clock)
    b.clock_ref = clock
    return b


def test_refresh_proposes_only_where_the_band_clears_the_ask(bot):
    info = bot.refresh()
    pend = bot.store.proposals("pending")
    assert info["proposals"] == len(pend) > 0
    for p in pend:
        assert p["edge"] >= bot.cfg.min_edge and p["lo"] > p["ask"]
        assert p["qty"] > 0 and p["limit_price"] <= p["p"] - bot.cfg.min_edge + 1e-9
        assert p["stake"] <= bot.api.cash + 1e-9
    ids = {p["contract_id"] for p in pend}
    assert "m8-yes" not in ids and "m8-no" not in ids  # no reference, no view, no trade


def test_approve_trades_logs_and_resolves(bot):
    bot.refresh()
    p = max(bot.store.proposals("pending"), key=lambda x: x["edge"])
    cash0 = bot.api.cash
    res = bot.approve(p["id"])
    assert res["ok"], res
    done = bot.store.proposal(p["id"])
    assert done["status"] == "executed" and done["fill_qty"] > 0
    assert bot.api.cash == pytest.approx(cash0 - done["fill_qty"] * done["fill_price"])
    rows = journal.load(bot.journal_path)
    assert len(rows) == 1 and rows.iloc[0]["market"] == p["contract_id"]
    assert float(rows.iloc[0]["p_belief"]) == pytest.approx(p["p"], abs=1e-4)
    bot.clock_ref.advance(timedelta(days=8))  # every demo market has resolved by then
    bot.refresh()
    rows = journal.load(bot.journal_path)
    assert rows["outcome"].notna().all()
    assert bot.approve(p["id"])["ok"] is False  # not pending any more


def test_forecast_only_skip_and_deviation(bot):
    bot.refresh()
    a, b, c = bot.store.proposals("pending")[:3]
    assert bot.approve(a["id"], stake=0)["message"].startswith("forecast logged")
    assert bot.skip(b["id"], note="not sure")["ok"]
    assert bot.store.proposal(b["id"])["status"] == "skipped"
    bot.approve(c["id"], stake=round(c["stake"] / 2))
    rows = journal.load(bot.journal_path)
    assert (rows["stake"].astype(float) == 0).sum() == 1
    # declining the policy's stake (forecast only) and halving it are both deviations, and logged as such
    assert rows["reason"].str.startswith("DEVIATION:").sum() == 2


def test_guards_refuse_unconfirmed_mappings_and_the_kill_switch(bot):
    bot.refresh()
    p = bot.store.proposals("pending")[0]
    bot.set_mapping(p["contract_id"], p["mapping_keys"][0], "suggested")  # un-confirm
    bot.refresh()
    q = next(x for x in bot.store.proposals("pending") if x["contract_id"] == p["contract_id"])
    assert not q["mapping_confirmed"]
    assert "confirm" in bot.approve(q["id"])["message"]
    bot.set_mapping(q["contract_id"], q["mapping_keys"][0], "confirmed")
    bot.refresh()
    r = next(x for x in bot.store.proposals("pending") if x["contract_id"] == p["contract_id"])
    risk.STOP_FILE.write_text("stop")
    assert "kill switch" in bot.approve(r["id"])["message"]
    risk.STOP_FILE.unlink()
    assert bot.approve(r["id"])["ok"]


def test_prize_mode_starts_all_in_on_one_market(bot):
    bot.set_mode("prize")
    bot.refresh()
    pend = bot.store.proposals("pending")
    staked = [p for p in pend if p["policy_stake"] > 0]
    assert len(staked) == 1 and staked[0]["kelly"] == max(p["kelly"] for p in pend)
    assert staked[0]["policy_stake"] == pytest.approx(bot.api.cash)  # the policy wants all-in ...
    assert 0 < staked[0]["stake"] <= staked[0]["policy_stake"]  # ... the book decides how much is possible
    assert any("sized to the book" in f for f in staked[0]["flags"])


def test_demo_reviewer_rule():
    assert reviewer({"edge": 0.06, "flags": []})[0] is None
    assert reviewer({"edge": 0.04, "flags": []})[0] == 0.0


# ---------------------------------------------------------------- web desk
@pytest.fixture
def desk(bot):
    d = Desk(bot, port=0)
    srv = d.serve(background_loop=False)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield d, srv.server_address[1]
    srv.shutdown()


def call(port, path, body=None, token=None, host=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=None if body is None else json.dumps(body).encode(),
                                 method="GET" if body is None else "POST")
    if token:
        req.add_header("X-Markout-Token", token)
    if host:
        req.add_header("Host", host)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read() or b"{}") if path != "/" else r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, None


def test_desk_serves_state_and_requires_the_token(desk):
    d, port = desk
    code, page = call(port, "/")
    assert code == 200 and d.token in page
    code, state = call(port, "/api/state")
    assert code == 200 and state["adapter"] == "paper"
    assert call(port, "/api/refresh", {})[0] == 403  # no token
    code, res = call(port, "/api/refresh", {}, token=d.token)
    assert code == 200 and res["ok"]


def test_desk_refuses_foreign_hosts(desk):
    d, port = desk
    assert call(port, "/api/state", host="evil.example.com")[0] == 403
    assert call(port, "/api/refresh", {}, token=d.token, host="evil.example.com")[0] == 403


def test_page_has_no_inline_handlers_with_data():
    assert 'onclick="post(' not in PAGE and "data-act=" in PAGE


def test_kelly_is_a_target_exposure_and_decided_contracts_cool_down(bot):
    bot.refresh()
    p = max(bot.store.proposals("pending"), key=lambda x: x["edge"])
    assert bot.approve(p["id"])["ok"]
    bot.refresh()  # same prices, minutes later: the decided contract is not asked again
    assert all(x["contract_id"] != p["contract_id"] for x in bot.store.proposals("pending"))
    held = {x.contract_id: x.qty for x in bot.api.positions()}
    cands = P.screen(*[bot.api.markets()], bot.api.books([c.id for m in bot.api.markets() for c in m.contracts]),
                     {}, bot.cfg, bot.now())
    assert cands == []  # (no estimates given: nothing to screen)
    from markout.cup.engine import Estimate
    from markout.cup.proposals import Candidate as Cand
    b = bot.api.books([p["contract_id"]])[p["contract_id"]]
    m = next(m for m in bot.api.markets() if m.id == p["market_id"])
    c = Cand(m, p["contract_id"], p["contract_name"], Estimate(p["p"], p["lo"], p["hi"], b.mid), b,
             p["kelly"], p["edge"], True, [], [])
    target = bot.cfg.kelly_mult * p["kelly"] * bot.api.account().equity
    (_, stake_fresh, _), = P.size([c], bot.api.account(), bot.cfg, held={})
    (_, stake_held, why), = P.size([c], bot.api.account(), bot.cfg, held=held)
    assert stake_held < stake_fresh  # what is already held counts toward the target
    assert stake_held == pytest.approx(max(min(target, 0.25 * bot.api.account().equity)
                                           - held[p["contract_id"]] * b.best_ask, 0.0), abs=1e-6) or stake_held == 0
