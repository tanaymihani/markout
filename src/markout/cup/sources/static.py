"""Bundled reference quotes for the offline paper/demo scenario (no network)."""

from __future__ import annotations

from markout.cup.sources.base import ExternalQuote


def from_scenario(refs: list[dict]) -> list[ExternalQuote]:
    return [ExternalQuote(r.get("source", "static"), r["id"], r["question"], r.get("outcome", "Yes"), float(r["p"]),
                          r.get("kind", "market"), r.get("bid"), r.get("ask"), r.get("liquidity"), r.get("volume_24h"),
                          None, r.get("url", ""), r.get("n"), r.get("note", "bundled demo reference (synthetic)"))
            for r in refs]
