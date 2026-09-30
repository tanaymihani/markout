"""Offline, scripted demo of the Predictions Cup desk: `python -m markout.cup demo`.

Runs the whole loop (refresh -> estimates -> proposals -> a scripted reviewer's decisions ->
orders -> resolutions -> journal scoring) on the bundled fictional scenario with a
simulated clock, then writes docs/demo/: DEMO.md, figures, a read-only static snapshot of
the desk page (desk.html) and a screenshot of it (desk.png, via headless Chrome when
installed). Everything is synthetic by construction: it demonstrates the mechanics and
the audit trail, not an edge.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from markout import plotting as mp
from markout import reporting as R
from markout.cup import proposals as P
from markout.cup.paper import PaperContest, load_scenario
from markout.cup.refs import References
from markout.cup.service import Bot
from markout.cup.sources.static import from_scenario
from markout.cup.store import Store
from markout.cup.web import PAGE
from markout.decision import journal
from markout.paths import DATA, ROOT

SCENARIO = ROOT / "configs" / "cup_demo_scenario.json"
OUT = ROOT / "docs" / "demo"
CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
START = datetime(2026, 10, 1, 16, 0, tzinfo=timezone.utc)  # the contest opens at 12:00 ET


class ManualClock:
    def __init__(self, t: datetime):
        self.t = t

    def __call__(self) -> datetime:
        return self.t

    def advance(self, dt: timedelta) -> None:
        self.t += dt


def reviewer(p: dict) -> tuple[float | None, str]:
    """The scripted human: take clear edges at the policy stake, log everything else as a forecast."""
    if p["edge"] >= 0.05 and "resolves after the contest ends" not in p["flags"]:
        return None, "demo reviewer: edge >= 5 points, take the policy stake"
    return 0.0, "demo reviewer: small edge, log the forecast only"


def static_page(state: dict) -> str:
    shim = ("<script>window.__STATE__=" + json.dumps(state, default=str).replace("</", "<\\/") + ";"
            "window.fetch=async(u,o)=>new Response(JSON.stringify(u==='/api/state'?window.__STATE__:"
            "{ok:false,message:'Read-only demo snapshot'}));</script>")
    return PAGE.replace("__TOKEN__", "demo-snapshot").replace("<script>", shim + "<script>", 1)


def take_screenshot(html: Path, png: Path, w: int = 1400, h: int = 1150, wait: float = 60.0) -> bool:
    """Headless Chrome with a throwaway profile. Chrome sometimes keeps running after it has
    written the file (the page polls), so wait for the file and then close Chrome."""
    import time

    if not CHROME.exists():
        return False
    png.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory() as prof:
        proc = subprocess.Popen([str(CHROME), "--headless=new", "--disable-gpu", "--hide-scrollbars",
                                 f"--user-data-dir={prof}", f"--window-size={w},{h}", f"--screenshot={png}",
                                 html.resolve().as_uri()], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        t0, last = time.monotonic(), -1
        while time.monotonic() - t0 < wait:
            if png.exists() and png.stat().st_size > 0 and png.stat().st_size == last:
                break
            last = png.stat().st_size if png.exists() else -1
            time.sleep(1.0)
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    return png.exists() and png.stat().st_size > 0


def run(out_dir: Path = OUT, screenshot_page: bool = True, mode: str = "steady", seed: int = 3,
        screenshot: bool | None = None) -> Path:
    if screenshot is not None:
        screenshot_page = screenshot
    work = DATA / "cup" / "demo"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    clock = ManualClock(START)
    specs = load_scenario(SCENARIO, START)
    api = PaperContest(specs, start_cash=10_000.0, seed=seed, clock=clock)
    static = {cid: from_scenario([r for r in s.references if r["contract_id"] == cid])
              for s in specs for cid in {r["contract_id"] for r in s.references}}
    bot = Bot(api, Store(work / "demo.db"), References(live=False, static=static), P.Config(mode=mode),
              journal_path=work / "journal.csv", clock=clock)
    decisions, snap = [], None
    for _ in range(0, 7 * 24, 2):  # every 2 simulated hours for a week
        bot.refresh()
        pending = bot.store.proposals("pending")
        if pending and snap is None:
            snap = bot.state()  # what the desk looks like before anyone decides
        for p in pending:
            stake, why = reviewer(p)
            res = bot.approve(p["id"], stake, note=why)
            decisions.append({"t": clock().isoformat(timespec="minutes"), "market": p["market_title"],
                              "contract": p["contract_name"], "p": p["p"], "ask": p["ask"], "edge": p["edge"],
                              "stake": p["stake"] if stake is None else stake, "result": res["message"]})
        clock.advance(timedelta(hours=2))
    bot.refresh()
    acct = api.account()
    df = journal.load(work / "journal.csv")
    scores = journal.score(df)
    snaps = bot.store.snapshots()

    def equity():
        fig, ax = mp.subplots(h=3.2)
        t = [(datetime.fromisoformat(s["ts"]) - START).total_seconds() / 86400 for s in snaps]
        ax.plot(t, [s["equity"] for s in snaps], color=mp.series(0))
        ax.axhline(10_000, color=mp.baseline_color(), linewidth=0.9, zorder=1)
        ax.set_xlabel("days since the contest opened (simulated)")
        ax.set_ylabel("equity, SUSQies")
        mp.title(ax, "Demo account equity", "Fictional markets, synthetic prices; mechanics, not edge")
        return fig

    mp.render("cup_demo_equity", equity, out_dir=out_dir)
    (out_dir / "desk.html").write_text(static_page(snap or bot.state()))
    shot = screenshot_page and take_screenshot(out_dir / "desk.html", out_dir / "desk.png")
    traded = [d for d in decisions if d["stake"] and "@" in d["result"]]
    logged = [d for d in decisions if not d["stake"]]
    s = scores or {}
    lines = [
        "# Markout Desk: demo walkthrough", "",
        "> Fictional markets and synthetic prices (`configs/cup_demo_scenario.json`), a simulated clock and a "
        "scripted reviewer. It shows how the desk works end to end; it says nothing about real edge.", "",
        "## What the desk does", "",
        R.bullets([
            "Pulls every open contest market and its order book (here: a paper exchange; live: the Predictions Cup API).",
            "Finds reference prices for each contract in free, public sources (Polymarket and Kalshi prices, "
            "option-implied probabilities for price questions, earnings-beat history), and proposes a mapping "
            "that a human must confirm.",
            "Pools the references with the contest's own price in log-odds (explicit weights, printed on every "
            "proposal), and proposes a trade only when the whole uncertainty band clears the ask by at least "
            "3 points.",
            "Sizes it with Kelly: the pre-registered prize policy from report 05, or a steady 0.5x Kelly mode.",
            "Trades only after an explicit click, with a last look at a fresh book; a kill switch stops everything.",
            "Writes every decision, including forecasts it did not trade, to the decision journal, which scores "
            "the beliefs against the market when markets resolve (Brier and log score, calibration).",
        ]), "",
        "## The desk", "",
        ("![The approval desk](desk.png)" if shot else "_(screenshot unavailable: Google Chrome not found)_") +
        "  \n[Open the read-only snapshot](desk.html) in a browser to click around.", "",
        "## This demo run", "",
        R.table([{"what": k, "value": v} for k, v in [
            ("simulated days", "7"),
            ("markets", str(len(specs))),
            ("proposals decided", str(len(decisions))),
            ("trades placed", str(len(traded))),
            ("forecasts logged without trading", str(len(logged))),
            ("final equity (start 10,000)", f"{acct.equity:,.0f}"),
            ("journal rows / resolved", f"{len(df)} / {int(df['outcome'].notna().sum()) if len(df) else 0}"),
            ("Brier: bot vs contest price at entry", f"{s.get('brier_you', float('nan')):.3f} vs "
                                                      f"{s.get('brier_market', float('nan')):.3f}" if s else "n/a"),
        ]]), "",
        mp.picture("cup_demo_equity", "Demo account equity", prefix=""), "",
        "Decisions, in order:", "",
        R.table(decisions, columns=["t", "market", "contract", "p", "ask", "edge", "stake", "result"],
                formats={"p": "{:.2f}", "ask": "{:.2f}", "edge": "{:+.2f}", "stake": "{:,.0f}"}), "",
        "## Run it", "",
        "```bash",
        "python -m markout.cup demo                       # this page (offline, deterministic)",
        "python -m markout.cup serve --paper              # the desk on a paper contest: http://127.0.0.1:8765",
        "python -m markout.cup serve --paper --live-refs  # paper contest, live Polymarket/Kalshi references",
        "python -m markout.cup serve --live               # the real contest, once the Oct 1 adapter exists",
        "```", "",
        "The live adapter is written against the platform's API reference, which is only visible after "
        "registering. The contest's rules allow bots (one account, individual participation, the platform's "
        "rate and position limits).",
    ]
    (out_dir / "DEMO.md").write_text("\n".join(lines) + "\n")
    return out_dir / "DEMO.md"
