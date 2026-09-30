"""Predictions Cup desk.

    python -m markout.cup serve --paper                  # paper contest from the demo scenario
    python -m markout.cup serve --paper --live-refs      # paper contest + live Polymarket/Kalshi references
    python -m markout.cup serve --live                   # the real contest (needs the Oct 1 adapter)
    python -m markout.cup refs "senate 2026"             # search the free reference sources
    python -m markout.cup demo                           # offline scripted demo -> docs/demo/
    python -m markout.cup dryrun                         # matcher on live markets -> docs/demo/DRYRUN.md
Open http://127.0.0.1:8765 to review and approve proposals. Nothing trades without approval.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone

from markout.paths import DATA, ROOT

SCENARIO = ROOT / "configs" / "cup_demo_scenario.json"


class SimClock:
    """Paper-mode clock: simulated time runs `speed` times faster than real time."""

    def __init__(self, speed: float = 1.0, start: datetime | None = None):
        self.speed = speed
        self.t0 = datetime.now(timezone.utc)
        self.start = start or self.t0

    def __call__(self) -> datetime:
        return self.start + (datetime.now(timezone.utc) - self.t0) * self.speed


def cmd_serve(a) -> int:
    from markout.cup import proposals as P
    from markout.cup.api import Limited, LiveAPI, RateLimiter
    from markout.cup.paper import PaperContest, load_scenario
    from markout.cup.refs import References
    from markout.cup.service import Bot
    from markout.cup.sources.static import from_scenario
    from markout.cup.store import Store
    from markout.cup.web import Desk

    cfg = P.Config(mode=a.mode, min_edge=a.min_edge, max_stake=a.max_stake)
    if a.live:
        api = Limited(LiveAPI(), RateLimiter(rate=a.rate))  # raises until the adapter is written
        refs, store, clock = References(live=True), Store(DATA / "cup" / "cup.db"), None
        journal_path = DATA / "journal.csv"
    else:
        clock = SimClock(speed=a.speed)
        specs = load_scenario(a.scenario, clock())
        api = PaperContest(specs, start_cash=a.cash, seed=a.seed, clock=clock)
        static = {} if a.live_refs else {cid: from_scenario([r for r in s.references if r["contract_id"] == cid])
                                         for s in specs for cid in {r["contract_id"] for r in s.references}}
        refs = References(live=a.live_refs, static=static)
        store = Store(DATA / "cup" / "paper.db")
        journal_path = DATA / "cup" / "paper_journal.csv"
    from markout.cup.service import Bot  # noqa: F811

    bot = Bot(api, store, refs, cfg, journal_path=journal_path, **({"clock": clock} if clock else {}))
    desk = Desk(bot, port=a.port, interval=a.interval)
    srv = desk.serve()
    print(f"Markout Desk ({api.name}) on http://127.0.0.1:{a.port}  (Ctrl-C to stop; kill switch: touch data/cup/STOP)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        desk.stop()
    return 0


def cmd_refs(a) -> int:
    from markout.cup.sources import kalshi, polymarket

    q = " ".join(a.query).lower()
    rows = [x for x in polymarket.search(q)] + [x for x in kalshi.open_markets(max_pages=2)
                                                  if all(w in x.question.lower() for w in q.split())]
    for x in rows[: a.limit]:
        print(f"{x.source:10s} {x.p:5.2f}  {x.question[:90]} [{x.outcome}]  liq {x.liquidity or 0:,.0f}")
    print(f"{len(rows)} matches")
    return 0


def cmd_dryrun(a) -> int:
    from markout.cup.dryrun import main as dry

    return dry()


def cmd_demo(a) -> int:
    from markout.cup.demo import run

    out = run(screenshot=not a.no_screenshot)
    print(f"demo written to {out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m markout.cup", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    g = s.add_mutually_exclusive_group()
    g.add_argument("--paper", action="store_true", default=True)
    g.add_argument("--live", action="store_true")
    s.add_argument("--live-refs", action="store_true", help="use live Polymarket/Kalshi references in paper mode")
    s.add_argument("--scenario", default=str(SCENARIO))
    s.add_argument("--mode", default="prize", choices=["prize", "steady"])
    s.add_argument("--min-edge", type=float, default=0.03)
    s.add_argument("--max-stake", type=float, default=None, help="platform position limit, once known")
    s.add_argument("--cash", type=float, default=10_000.0)
    s.add_argument("--speed", type=float, default=60.0, help="paper clock speed-up")
    s.add_argument("--interval", type=float, default=30.0, help="seconds between refreshes")
    s.add_argument("--rate", type=float, default=1.0, help="max API calls per second (live)")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_serve)
    r = sub.add_parser("refs")
    r.add_argument("query", nargs="+")
    r.add_argument("--limit", type=int, default=20)
    r.set_defaults(fn=cmd_refs)
    sub.add_parser("dryrun", help="matcher dry run on live markets -> docs/demo/DRYRUN.md").set_defaults(fn=cmd_dryrun)
    d = sub.add_parser("demo")
    d.add_argument("--no-screenshot", action="store_true")
    d.set_defaults(fn=cmd_demo)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
