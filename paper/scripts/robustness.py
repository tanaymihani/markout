"""Robustness of the post-or-cross results (gross of fees), not in report 02.

(a) horizon H = 10, 30, 60 s (the posted order lives H seconds; both actions are marked
    to the mid at t + H); (b) queue buckets scaled by the morning-only median queue
    instead of the full-day median. Writes paper/generated/robustness.json.
Needs the LOBSTER sample files (`make lobster`); run from the repository root.
"""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from markout.lob.lobster import SPLIT, TICKERS, load_day
from markout.lob.markouts import queue_buckets
from markout.lob.postcross import block_bootstrap_ci, decisions, evaluate_rule

OUT = Path(__file__).resolve().parents[1] / "generated" / "robustness.json"
SEEDS = {"touch": 0, "fifo": 1}


def summary(d, model, horizon):
    r = asdict(evaluate_rule(d, model, horizon=horizon, seed=SEEDS[model]))
    test = d[(d["model"] == model) & (d["t"] >= SPLIT)]
    r["always_post_ci"] = block_bootstrap_ci(test["t"].to_numpy(), test["ev_post"].to_numpy(), seed=SEEDS[model])
    return r


rows = []
for ticker in TICKERS:
    day = load_day(ticker)
    for horizon in (10.0, 30.0, 60.0):
        pc = decisions(day, horizon=horizon)
        for model in SEEDS:
            rows.append({"ticker": ticker, "check": "horizon", "horizon": horizon, **summary(pc, model, horizon)})
        if horizon == 30.0:
            d = pc.copy()
            morning = d[(d["model"] == "fifo") & (d["t"] + horizon <= SPLIT)]["queue"].to_numpy()
            d["queue_bucket"] = queue_buckets(d["queue"].to_numpy(), float(np.median(morning)))
            for model in SEEDS:
                rows.append(
                    {"ticker": ticker, "check": "morning_median", "horizon": horizon, **summary(d, model, horizon)}
                )
    print(ticker, "done")
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(rows, indent=1))
print("wrote", OUT)
