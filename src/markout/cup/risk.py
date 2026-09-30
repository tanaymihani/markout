"""Pre-trade checks run on every approval, with a fresh book (a "last look").

The kill switch is a file: `touch data/cup/STOP` stops all trading at once, from any
terminal, even if the web page is unreachable; delete it to resume."""

from __future__ import annotations

from datetime import datetime, timedelta

from markout.cup.proposals import Config
from markout.cup.types import Account, Book, Market
from markout.paths import DATA

STOP_FILE = DATA / "cup" / "STOP"


def check(p: dict, stake: float, account: Account, book: Book | None, market: Market | None, cfg: Config,
          now: datetime) -> tuple[bool, str]:
    if STOP_FILE.exists():
        return False, "kill switch is on (data/cup/STOP)"
    if market is None:
        return False, "market no longer listed"
    if market.closes_at and market.closes_at < now + timedelta(minutes=cfg.min_minutes_to_close):
        return False, "market closes too soon"
    if not p.get("mapping_confirmed"):
        return False, "confirm the reference mapping first"
    if stake < 0 or stake > account.cash + 1e-9:
        return False, f"stake {stake:.2f} exceeds available cash {account.cash:.2f}"
    if cfg.max_stake is not None and stake > cfg.max_stake + 1e-9:
        return False, f"stake exceeds the position limit {cfg.max_stake:g}"
    if stake == 0:
        return True, "forecast only"
    if book is None or book.best_ask is None:
        return False, "no ask in the book"
    a = book.best_ask
    if p["p"] - a < cfg.min_edge:
        return False, f"edge gone: ask is now {a:.2f} against p {p['p']:.2f}"
    return True, "ok"
