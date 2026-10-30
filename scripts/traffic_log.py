"""Append the repository's GitHub traffic to a CSV: `python scripts/traffic_log.py` (weekly is enough).

GitHub only keeps traffic for 14 days, so a log is the only way the numbers survive. Each run
adds one row: views and unique visitors, clones, stars, forks and the top referrers over the
last 14 days. Needs the gh CLI, logged in with push access. The CSV is local (data/ is not
committed).
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

REPO = "tanaymihani/markout"
OUT = Path(__file__).resolve().parents[1] / "data" / "traffic.csv"


def gh(path: str):
    return json.loads(subprocess.run(["gh", "api", path], capture_output=True, text=True, check=True).stdout)


def main(out: Path = OUT) -> int:
    views, clones = gh(f"repos/{REPO}/traffic/views"), gh(f"repos/{REPO}/traffic/clones")
    refs, repo = gh(f"repos/{REPO}/traffic/popular/referrers"), gh(f"repos/{REPO}")
    row = {
        "date": date.today().isoformat(),
        "views_14d": views["count"],
        "visitors_14d": views["uniques"],
        "clones_14d": clones["count"],
        "cloners_14d": clones["uniques"],
        "stars": repo["stargazers_count"],
        "forks": repo["forks_count"],
        "top_referrers": "; ".join(f"{r['referrer']} {r['count']}" for r in refs[:5]),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with out.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print(", ".join(f"{k} {v}" for k, v in row.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
