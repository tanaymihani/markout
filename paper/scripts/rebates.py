"""Fee and rebate sensitivity for the post-or-cross rule (not in report 02).

Re-runs report 02's decision frames, then charges every cross the Reg NMS access-fee cap
(0.3 ticks per share) and credits every posted fill a maker rebate of r ticks per filled
share, *before* the rule is chosen, so the selection can react to the rebate. The rule is
chosen before 12:45 and scored after, with the same block bootstrap and seeds as report 02.
Writes reports/results/fill_rebates.json (the site's fee and rebate toggles read it).
Needs the LOBSTER sample files (`make lobster`); run from the repository root.
"""

import json
from dataclasses import asdict

from markout.lob.fills import MODELS
from markout.lob.lobster import SPLIT, TICKERS, load_day
from markout.lob.postcross import TAKER_FEE_TICKS, block_bootstrap_ci, decisions, evaluate_rule
from markout.paths import RESULTS

OUT = RESULTS / "fill_rebates.json"
REBATES = (0.0, 0.2, 0.3)


def always_post_ci(d, model, seed):
    """95% block-bootstrap CI of always posting, on the test decisions (after 12:45)."""
    test = d[(d["model"] == model) & (d["t"] >= SPLIT)]
    return block_bootstrap_ci(test["t"].to_numpy(), test["ev_post"].to_numpy(), seed=seed)


rows = []
for ticker in TICKERS:
    pc = decisions(load_day(ticker))
    for r in REBATES:
        d = pc.copy()
        d["ev_post"] = d["ev_post"] + r * d["fill_frac"]
        d["ev_cross"] = d["ev_cross"] - TAKER_FEE_TICKS
        d["diff"] = d["ev_post"] - d["ev_cross"]
        for k, model in enumerate(MODELS):
            res = asdict(evaluate_rule(d, model, seed=k, fee_ticks=0.0))
            rows.append(
                {
                    "ticker": ticker,
                    "rebate_ticks": r,
                    "fee_ticks": TAKER_FEE_TICKS,
                    **res,
                    "always_post_ci": always_post_ci(d, model, k),
                }
            )
    # gross reproduction check: no fees, no rebates, must equal report 02's table
    for k, model in enumerate(MODELS):
        res = asdict(evaluate_rule(pc, model, seed=k))
        rows.append(
            {
                "ticker": ticker,
                "rebate_ticks": None,
                "fee_ticks": 0.0,
                **res,
                "always_post_ci": always_post_ci(pc, model, k),
            }
        )
    print(ticker, "done")
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(rows, indent=1))
print("wrote", OUT)
