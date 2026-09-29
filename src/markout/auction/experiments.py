"""The experiment grid. Every (model, decision policy) pair is one trial, and every
trial's daily out-of-sample PnL goes into the registry, including the losers, so
the Deflated Sharpe and PBO see the true amount of searching that was done.

Selection happens at 1x cost on the research period only. Cost multiples are a
robustness check on the selected trial, never a selection criterion.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from markout.auction.pipeline import ModelSpec
from markout.backtest.costs import CostModel
from markout.backtest.engine import Policy, backtest

ALL = ("book", "auction", "dynamics", "xs", "hist")

MODELS: list[ModelSpec] = [
    ModelSpec("zero", groups=()),
    ModelSpec("median", groups=()),
    ModelSpec("ridge", groups=ALL, params={"alpha": 10.0}),
    ModelSpec("lgbm", groups=("book", "auction", "dynamics"),
              params={"num_leaves": 63, "min_data_in_leaf": 500}),
    ModelSpec("lgbm", groups=ALL, params={"num_leaves": 63, "min_data_in_leaf": 500}),
    ModelSpec("lgbm", groups=ALL, params={"num_leaves": 255, "min_data_in_leaf": 2000}),
]

POLICIES: list[Policy] = [Policy(threshold=t, sizing=s) for t in (1.0, 1.25, 1.5, 2.0) for s in ("flat", "ramp")]

COSTS = CostModel()


def model_label(spec: ModelSpec) -> str:
    if spec.kind != "lgbm":
        return spec.kind
    g = "all" if tuple(spec.groups) == ALL else "+".join(spec.groups)
    return f"lgbm[{g}, {spec.params.get('num_leaves', 63)} leaves]"


@dataclass
class Trial:
    spec: ModelSpec
    policy: Policy
    summary: dict
    daily: pl.DataFrame
    trial_id: str | None = None

    @property
    def config(self) -> dict:
        return {"model": self.spec.to_dict(), "policy": self.policy.to_dict(), "costs": COSTS.to_dict(),
                "cost_mult": 1.0}


def run_policies(spec: ModelSpec, dec: pl.DataFrame, days: tuple[int, int], registry=None,
                 policies: list[Policy] = POLICIES) -> list[Trial]:
    """Backtest every policy on one model's decision frame; log each as a trial."""
    out = []
    for pol in policies:
        _, day, summary = backtest(dec, pol, COSTS, days)
        t = Trial(spec, pol, summary, day)
        if registry is not None:
            pnl = day.select("date_id", "pnl_usd").to_pandas().set_index("date_id")["pnl_usd"]
            metrics = {k: summary[k] for k in ("sharpe_daily", "sharpe_annualized", "total_pnl_usd", "trades",
                                               "hit_rate", "net_bps_per_trade")}
            t.trial_id = registry.log(t.config, metrics, pnl, tags={"model": model_label(spec)})
        out.append(t)
    return out
