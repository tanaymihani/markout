"""Round-trip trading costs in bps.

A trade here is a 60 s round trip in one stock, hedged with the index:
  - cross the spread to get in and to get out: half the entry spread + half the exit
    spread, both read from the data's own bid and ask;
  - exchange fees on both legs (an assumption, varied in the sensitivity analysis);
  - the cost of the index hedge that makes the PnL index-relative, like the target
    (an assumption: the index is synthetic, so this leg cannot be observed).
The whole stack is scaled by a scenario multiplier (1x, 2x, 3x).
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class CostModel:
    fee_bps_per_side: float = 0.5
    hedge_bps_round_trip: float = 0.5

    def expr(self, multiplier: float = 1.0) -> pl.Expr:
        spread = (pl.col("spread_entry_bps") + pl.col("spread_exit_bps")) / 2
        return (multiplier * (spread + 2 * self.fee_bps_per_side + self.hedge_bps_round_trip)).alias("cost_bps")

    def to_dict(self) -> dict:
        return {"fee_bps_per_side": self.fee_bps_per_side, "hedge_bps_round_trip": self.hedge_bps_round_trip}
