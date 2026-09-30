"""The bot: refresh -> suggest mappings -> estimate -> propose; approve -> last look ->
order -> journal. Nothing trades without an explicit approval.

Every approval (including a zero-stake "forecast only" one) is written to the decision
journal (data/journal.csv by default), so `python -m markout.decision.journal score`
grades the bot's beliefs against the market after the contest, report 05 Part 2.
A stake that differs from the policy's stake is marked "DEVIATION:" in the journal,
as the pre-registration requires.
"""

from __future__ import annotations

import math
import threading
from datetime import datetime
from pathlib import Path

from markout.cup import engine, risk
from markout.cup import proposals as P
from markout.cup.refs import References
from markout.cup.store import Store
from markout.cup.types import Account, utcnow
from markout.decision import journal


class Bot:
    def __init__(self, api, store: Store, refs: References, cfg: P.Config | None = None,
                 journal_path: Path = journal.JOURNAL, clock=utcnow):
        self.api, self.store, self.refs, self.clock = api, store, refs, clock
        self.cfg = cfg or P.Config()
        saved = store.get("mode")
        if saved in ("prize", "steady"):
            self.cfg.mode = saved
        self.journal_path = Path(journal_path)
        if not self.journal_path.exists():
            journal.init(self.journal_path)
        self.lock = threading.RLock()
        self.last_refresh: datetime | None = None
        self.last_error = ""
        self._resolved_logged: set[str] = set()

    # ------------------------------------------------------------------ helpers
    def now(self) -> datetime:
        return self.clock()

    def _market_map(self):
        ms = self.api.markets()
        return {m.id: m for m in ms}, {c.id: (m, c) for m in ms for c in m.contracts}

    # ------------------------------------------------------------------ refresh
    def refresh(self) -> dict:
        with self.lock:
            now = self.now()
            if hasattr(self.api, "step"):
                self.api.step(now)
            self._sync_resolutions()
            markets, by_contract = self._market_map()
            ids = list(by_contract)
            books = self.api.books(ids)
            estimates = {}
            for cid, (m, c) in by_contract.items():
                cands = self.refs.candidates(m, c)
                for cand in cands:
                    if self.store.mapping_status(cid, cand.key) is None:
                        auto = cand.why == "bundled reference"  # demo references are pre-matched
                        self.store.put_mapping(cid, cand.key, "confirmed" if auto else "suggested",
                                               {"score": cand.score, "why": cand.why, "question": cand.quote.question,
                                                "outcome": cand.quote.outcome, "source": cand.quote.source,
                                                "p": cand.p, "url": cand.quote.url, "invert": cand.invert})
                usable = [x for x in cands if self.store.mapping_status(cid, x.key) != "rejected"]
                confirmed = [x for x in usable if self.store.mapping_status(cid, x.key) == "confirmed"]
                refs = confirmed or usable[:1]  # propose from the best suggestion, but it must be confirmed to trade
                if cid in books and refs:
                    est = engine.estimate(books[cid], refs)
                    if est is not None:
                        estimates[cid] = (est, bool(confirmed), [x.key for x in refs])
            account = self.api.account()
            held = {p.contract_id: p.qty for p in self.api.positions()}
            new = P.build(list(markets.values()), books, estimates, account, self.cfg, now, held)
            new = self._cool_down(new, now)
            self._merge(new, now)
            self.store.snapshot(now.isoformat(timespec="seconds"),
                                {"cash": account.cash, "equity": account.equity, "rank": account.rank,
                                 "n": account.n_players})
            self.last_refresh = now
            return {"markets": len(markets), "contracts": len(ids), "estimates": len(estimates), "proposals": len(new)}

    def _cool_down(self, new: list[P.Proposal], now: datetime) -> list[P.Proposal]:
        """Drop re-proposals of a contract decided recently unless something moved.
        Without this, every refresh would re-ask the same question and re-log the same forecast."""
        from datetime import timedelta

        cutoff = (now - timedelta(hours=self.cfg.cooldown_hours)).isoformat()
        last: dict[str, dict] = {}
        for p in self.store.proposals():
            if p["status"] in ("executed", "logged", "skipped", "rejected") and (p.get("decided") or "") >= cutoff:
                last.setdefault(p["contract_id"], p)  # proposals() is newest first
        keep = []
        for p in new:
            prev = last.get(p.contract_id)
            moved = prev is None or abs(p.p - prev["p"]) >= self.cfg.requote_move \
                or abs(p.ask - prev["ask"]) >= self.cfg.requote_move
            if moved:
                keep.append(p)
        return keep

    def _merge(self, new: list[P.Proposal], now: datetime) -> None:
        """One pending proposal per contract: replace it when the numbers change, expire stale ones."""
        pending = {p["contract_id"]: p for p in self.store.proposals("pending")}
        fresh = {p.contract_id for p in new}
        for cid, old in pending.items():
            if cid not in fresh or (old.get("expires") and old["expires"] < now.isoformat()):
                old["status"], old["decided"] = "expired", now.isoformat(timespec="seconds")
                self.store.put_proposal(old)
        for p in new:
            old = pending.get(p.contract_id)
            if old and old["status"] == "pending" and abs(old["ask"] - p.ask) < 1e-9 and abs(old["p"] - p.p) < 0.005 \
                    and abs(old["stake"] - p.stake) < 1e-6:
                continue  # unchanged: keep the original (and its id)
            if old and old["status"] == "pending":
                old["status"], old["decided"] = "expired", now.isoformat(timespec="seconds")
                self.store.put_proposal(old)
            self.store.put_proposal(p.to_dict())

    # ------------------------------------------------------------------ decisions
    def approve(self, pid: str, stake: float | None = None, note: str = "") -> dict:
        with self.lock:
            p = self.store.proposal(pid)
            if p is None or p["status"] != "pending":
                return {"ok": False, "message": "proposal is not pending"}
            now = self.now()
            markets, by_contract = self._market_map()
            m = markets.get(p["market_id"])
            books = self.api.books([p["contract_id"]])
            book = books.get(p["contract_id"])
            account = self.api.account()
            stake = p["stake"] if stake is None else float(stake)
            ok, why = risk.check(p, stake, account, book, m, self.cfg, now)
            if not ok:
                return {"ok": False, "message": why}
            deviation = abs(stake - p["stake"]) > 1e-6
            reason = ("DEVIATION: " if deviation else "") + (note + " | " if note else "") + p["rationale"]
            q_ref = book.best_ask if book and book.best_ask else p["ask"]
            if stake == 0:
                journal.add(self.journal_path, market=p["contract_id"], side="YES", p=p["p"], q=q_ref, stake=0,
                            bankroll=max(account.equity, 1e-9), question=f"{p['market_title']} :: {p['contract_name']}",
                            reason=reason, timestamp=now.isoformat(timespec="seconds"))
                p.update(status="logged", decided=now.isoformat(timespec="seconds"), decided_stake=0.0, note=note)
                self.store.put_proposal(p)
                self.store.log(p["decided"], "logged", {"proposal": pid})
                return {"ok": True, "message": "forecast logged (no order)"}
            qty, limit, _ = P.fill_plan(book, p["p"] - self.cfg.min_edge, stake)
            if qty <= 0:
                return {"ok": False, "message": f"no contracts offered at a price with edge (ask {book.best_ask:.2f})"}
            order = self.api.place_order(p["contract_id"], "buy", limit, qty, client_id=pid)
            if order.status in ("open", "partial"):
                self.api.cancel_order(order.id)  # take what the book offers now; never leave a resting order
            filled = order.filled_qty
            fills = [f for f in self.api.fills() if f.order_id == order.id]
            px = sum(f.price * f.qty for f in fills) / filled if filled else None
            p.update(status="executed" if filled else "rejected",
                     decided=now.isoformat(timespec="seconds"), decided_stake=stake, note=note, order_id=order.id,
                     fill_price=px, fill_qty=filled)
            self.store.put_proposal(p)
            self.store.log(p["decided"], "order", {"proposal": pid, "order": order.id, "status": order.status,
                                                   "filled": filled, "price": px})
            if filled:
                journal.add(self.journal_path, market=p["contract_id"], side="YES", p=p["p"], q=px, stake=filled * px,
                            bankroll=max(account.equity, filled * px), question=f"{p['market_title']} :: {p['contract_name']}",
                            reason=reason, timestamp=now.isoformat(timespec="seconds"))
            return {"ok": bool(filled), "message": f"order {order.status}: {filled:g} @ {px:.2f}" if filled
                    else f"order {order.status}"}

    def skip(self, pid: str, note: str = "") -> dict:
        with self.lock:
            p = self.store.proposal(pid)
            if p is None or p["status"] != "pending":
                return {"ok": False, "message": "proposal is not pending"}
            p.update(status="skipped", decided=self.now().isoformat(timespec="seconds"), note=note)
            self.store.put_proposal(p)
            self.store.log(p["decided"], "skipped", {"proposal": pid, "note": note})
            return {"ok": True, "message": "skipped"}

    def set_mapping(self, contract_id: str, key: str, status: str) -> dict:
        assert status in ("confirmed", "rejected", "suggested")
        with self.lock:
            rows = [r for r in self.store.mappings(contract_id) if r["key"] == key]
            if not rows:
                return {"ok": False, "message": "unknown mapping"}
            body = {k: v for k, v in rows[0].items() if k not in ("contract_id", "key", "status")}
            self.store.put_mapping(contract_id, key, status, body)
            for p in self.store.proposals("pending"):  # the numbers change: re-propose on the next refresh
                if p["contract_id"] == contract_id:
                    p.update(status="expired", decided=self.now().isoformat(timespec="seconds"))
                    self.store.put_proposal(p)
            return {"ok": True, "message": f"mapping {status}"}

    def set_mode(self, mode: str) -> dict:
        if mode not in ("prize", "steady"):
            return {"ok": False, "message": "mode must be prize or steady"}
        with self.lock:
            self.cfg.mode = mode
            self.store.set("mode", mode)
            for p in self.store.proposals("pending"):
                p.update(status="expired", decided=self.now().isoformat(timespec="seconds"))
                self.store.put_proposal(p)
        return {"ok": True, "message": f"mode set to {mode}"}

    # ------------------------------------------------------------------ resolutions
    def _sync_resolutions(self) -> None:
        """Paper mode knows outcomes; live mode resolves when the platform reports them."""
        resolved = getattr(self.api, "resolved", {}) or {}
        for mid, outcomes in resolved.items():
            for cid, y in outcomes.items():
                if cid in self._resolved_logged:
                    continue
                try:
                    journal.resolve(self.journal_path, cid, int(y), resolved_at=self.now().isoformat(timespec="seconds"))
                except (ValueError, KeyError):
                    pass  # no journal rows for that contract
                self._resolved_logged.add(cid)

    # ------------------------------------------------------------------ view
    def state(self) -> dict:
        with self.lock:
            acct: Account = self.api.account()
            markets, by_contract = self._market_map()
            pos = []
            books = self.api.books([p.contract_id for p in self.api.positions()])
            for p in self.api.positions():
                m, c = by_contract.get(p.contract_id, (None, None))
                mid = books.get(p.contract_id).mid if p.contract_id in books else None
                pos.append({"contract_id": p.contract_id, "market": m.title if m else p.contract_id,
                            "contract": c.name if c else "", "qty": p.qty, "avg_price": p.avg_price, "mid": mid,
                            "pnl": (mid - p.avg_price) * p.qty if mid is not None else None})
            maps = {}
            for r in self.store.mappings():
                maps.setdefault(r["contract_id"], []).append(r)
            pend = self.store.proposals("pending")
            for p in pend:
                p["mappings"] = maps.get(p["contract_id"], [])
            return {"now": self.now().isoformat(timespec="seconds"), "adapter": self.api.name, "mode": self.cfg.mode,
                    "config": self.cfg.to_dict(), "account": acct.__dict__, "in_top3": acct.in_top3,
                    "kill_switch": risk.STOP_FILE.exists(), "last_refresh": self.last_refresh.isoformat(timespec="seconds")
                    if self.last_refresh else None, "last_error": self.last_error, "pending": pend,
                    "recent": [p for p in self.store.proposals() if p["status"] != "pending"][:30],
                    "positions": pos, "snapshots": self.store.snapshots()[-500:],
                    "source_errors": self.refs.errors[-5:]}
