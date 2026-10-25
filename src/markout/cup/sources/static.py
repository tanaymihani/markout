"""Bundled reference quotes for the offline paper/demo scenario (no network)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from markout.cup.sources.base import ExternalQuote

if TYPE_CHECKING:
    from markout.cup.paper import PaperMarketSpec


def from_scenario(refs: list[dict]) -> list[ExternalQuote]:
    return [
        ExternalQuote(
            r.get("source", "static"),
            r["id"],
            r["question"],
            r.get("outcome", "Yes"),
            float(r["p"]),
            r.get("kind", "market"),
            r.get("bid"),
            r.get("ask"),
            r.get("liquidity"),
            r.get("volume_24h"),
            None,
            r.get("url", ""),
            r.get("n"),
            r.get("note", "bundled demo reference (synthetic)"),
        )
        for r in refs
    ]


def bundled(specs: list[PaperMarketSpec]) -> dict[str, list[ExternalQuote]]:
    """The scenario's references, keyed by contest contract (what `References(static=...)` takes)."""
    return {
        cid: from_scenario([r for r in s.references if r["contract_id"] == cid])
        for s in specs
        for cid in {r["contract_id"] for r in s.references}
    }
