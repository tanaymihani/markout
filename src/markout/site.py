"""Build the GitHub Pages site in docs/: `python -m markout.site`.

The page (template: src/markout/site_template.html) is an interactive lab: a queue simulator,
the replay results for five NASDAQ stocks, a cost slider on the closing-auction strategy and
a Kelly-sizing lab on the S&P 500 volatility premium, plus the market-making game and the
Predictions Cup desk. Every number on it is read from saved results here and embedded in
the page as JSON; only derived results are published, never raw market data. The browser
code lives in docs/js/ and docs/game.js and is tested against the Python code.

docs/desk/ is the desk itself, running in the visitor's browser: the page, the loader
(docs/desk/loader.js, which starts Pyodide) and markout-desk.zip, the desk's own modules.
"""

from __future__ import annotations

import html
import io
import json
import math
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import date
from pathlib import Path
from typing import Any

from markout import results
from markout.paths import FIGURES, RESULTS, ROOT

DOCS = ROOT / "docs"
TEMPLATE = Path(__file__).with_name("site_template.html")
REPO = "https://github.com/tanaymihani/markout"
SITE = "https://tanaymihani.github.io/markout/"
PAPER_URL: str | None = None  # SSRN link, once the paper is public
DESK = DOCS / "desk"  # the desk, running in the visitor's browser on Pyodide
DESK_URL = SITE + "desk/"
VIDEO = DOCS / "media" / "markout-demo.mp4"
DESCRIPTION = (
    "A reality check for trading backtests: how much of an edge survives the spread, a real order queue, "
    "competing traders and selection bias, on real NASDAQ and S&P 500 data. Interactive labs run in the browser."
)
PREVIEW = "img/markout-card.png"  # link-preview image; falls back to a report figure if missing
ICON = (
    "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'>"
    "<rect width='32' height='32' rx='7' fill='%232a78d6'/><polyline points='6,22 12,15 17,19 26,9' fill='none' "
    "stroke='white' stroke-width='3' stroke-linecap='round' stroke-linejoin='round'/></svg>"
)
TICKERS = ("INTC", "MSFT", "AAPL", "AMZN", "GOOG")
BUILT_WITH = (
    "Python",
    "NumPy",
    "pandas",
    "Polars",
    "SciPy",
    "LightGBM",
    "C++17",
    "pybind11",
    "JavaScript",
    "SVG",
    "Pyodide",
    "WebAssembly",
    "Docker",
    "Playwright",
    "GitHub Actions",
    "GitHub Pages",
    "LOBSTER data",
    "Kaggle",
)


def _r(x, d=4):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), d)


def _ci(row, key="ci"):
    lo, hi = row[key]
    return [_r(lo), _r(hi)]


def micro_data() -> dict | None:
    m = results.load("microstructure")
    if not m:
        return None
    out: dict[str, Any] = {"tickers": list(TICKERS), "horizons": [0.1, 1, 10, 60], "stocks": {}}
    rb = _json(RESULTS / "fill_rebates.json")  # paper/scripts/rebates.py
    rob = _json(RESULTS / "fill_robustness.json")  # paper/scripts/robustness.py
    for t in TICKERS:
        fa, summ = m["facts"][t], {r["model"]: r for r in m["fills"][t]["summary"]}
        s = {
            "class": fa["tick_class"].replace("-tick", ""),
            "price": _r(fa["mean_price_usd"], 2),
            "tick_bps": _r(fa["tick_bps"], 2),
            "spread": _r(fa["spread_ticks"], 2),
            "depth": round(fa["depth_best"]),
            "auc": _r(m["predictive"][t]["imbalance"]["auc"], 3),
            "uncond": _r(summ["touch"]["uncond_10"], 3),
            "fills": {
                mod: {
                    "p": _r(r["p_fill"], 3),
                    "mk": [_r(r[f"mk_{h:g}"], 3) for h in out["horizons"]],
                    "mk10bps": _r(r["mk_10_bps"], 2),
                }
                for mod, r in summ.items()
            },
        }
        if rb:

            def rule(model, rebate, rows=rb, ticker=t):
                r = next(
                    x for x in rows if x["ticker"] == ticker and x["model"] == model and x["rebate_ticks"] == rebate
                )
                return {
                    "rule": [_r(r["out_of_sample"])] + _ci(r),
                    "always": [_r(r["always_post"])] + _ci(r, "always_post_ci"),
                    "post": _r(r["share_post"], 3),
                    "cross": _r(r["share_cross"], 3),
                }

            s["rules"] = {
                "gross": {mod: rule(mod, None) for mod in ("touch", "fifo")},
                "r02": {mod: rule(mod, 0.2) for mod in ("touch", "fifo")},
                "r03": {mod: rule(mod, 0.3) for mod in ("touch", "fifo")},
            }
        if rob:
            for h in (10.0, 60.0):
                s.setdefault("rules", {})[f"h{int(h)}"] = {
                    mod: (
                        lambda r: {
                            "rule": [_r(r["out_of_sample"])] + _ci(r),
                            "always": [_r(r["always_post"])] + _ci(r, "always_post_ci"),
                            "post": _r(r["share_post"], 3),
                            "cross": _r(r["share_cross"], 3),
                        }
                    )(
                        next(
                            x
                            for x in rob
                            if x["ticker"] == t and x["model"] == mod and x["check"] == "horizon" and x["horizon"] == h
                        )
                    )
                    for mod in ("touch", "fifo")
                }
        out["stocks"][t] = s
    return out


def auction_data() -> dict | None:
    a = results.load("auction")
    daily = ROOT / "reports" / "results" / "auction_daily_research.csv"
    if not a or not daily.exists():
        return None
    import pandas as pd

    d = pd.read_csv(daily)
    rows = {r["cost_mult"]: r for r in a["robustness"]["cost_rows"]}
    h = (a.get("holdout") or {}).get("adapted_1x", {})
    return {
        "gross": [round(x, 2) for x in d["gross_usd"]],
        "cost": [round(x, 2) for x in d["cost_usd"]],
        "days": int(len(d)),
        "sharpe": _r(a["selected"]["summary"]["sharpe_annualized"], 2),
        "adapted": {str(int(k)): _r(v["adapted"]["sharpe_annualized"], 2) for k, v in rows.items()},
        "holdout": {"sharpe": _r(h.get("sharpe_annualized"), 2), "days": h.get("days")},
    }


def vol_data() -> dict | None:
    v, d = results.load("vol_premium"), results.load("vol_monthly")  # both written by markout.vol.report
    if not v or not d:
        return None
    kf, kp = v["kelly_full"], v["kelly_pre2008"]
    return {
        "dates": d["dates"],
        "pnl": [round(x, 4) for x in d["var_swap"]],
        "kelly": {
            "full": _r(100 * kf["v_star"], 2),
            "ruin": _r(100 * kf["v_max"], 2),
            "cont": _r(100 * kf["v_continuous"], 2),
            "pre": _r(100 * kp["v_star"], 2),
        },
        "survivable_post2008": _r(v["survivable_vega_per_100"], 2),
    }


def _json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def lab_data() -> dict:
    return {"repo": REPO, "micro": micro_data(), "auction": auction_data(), "vol": vol_data()}


def findings() -> list[tuple[str, str, str]]:
    out = []
    a = results.load("auction")
    if a:
        s, h = a["selected"]["summary"], (a.get("holdout") or {}).get("adapted_1x", {})
        two = next(r for r in a["robustness"]["cost_rows"] if r["cost_mult"] == 2.0)["adapted"]["sharpe_annualized"]
        out.append(
            (
                "Closing auction",
                "01_auction",
                (
                    f"LightGBM beats a per-stock baseline on every walk-forward day. Over {s['days']} walk-forward "
                    f"days, traded net of costs, it has an annualized Sharpe of "
                    f"<span class=num>{s['sharpe_annualized']:.1f}</span> but nets only "
                    f"<span class=num>${s['total_pnl_usd']:,.0f}</span>, and at twice the assumed costs the Sharpe is "
                    f"<span class=num>{two:.1f}</span>. A {h.get('days', 0)}-day holdout, scored once, gave "
                    f"<span class=num>{h.get('sharpe_annualized', float('nan')):.1f}</span>."
                ),
            )
        )
    m = results.load("microstructure")
    if m:
        intc = {r["model"]: r for r in m["fills"]["INTC"]["summary"]}
        out.append(
            (
                "Realistic fills",
                "02_microstructure",
                (
                    f"Resting orders in INTC look profitable if any trade at your price fills you "
                    f"(<span class=num>{intc['touch']['mk_10_bps']:+.2f}</span> bps ten seconds later). Tracking queue "
                    f"position order by order turns that into <span class=num>{intc['fifo']['mk_10_bps']:+.2f}</span> bps: "
                    "the fills you actually get are the ones the market was about to run over."
                ),
            )
        )
    ar = results.load("arena")
    if ar:
        k = {r["K"]: r for r in ar["arena"]["k_sweep"]}
        out.append(
            (
                "Competition",
                "03_arena",
                (
                    f"A lone market maker quotes a <span class=num>{100 * k[1]['quoted_spread']['mean']:.0f}</span>-tick "
                    f"spread; one identical rival brings it to <span class=num>{100 * k[2]['quoted_spread']['mean']:.1f}</span>, "
                    f"next to the zero-profit level of <span class=num>{100 * k[2]['zero_profit_spread']['mean']:.1f}</span>. "
                    "Competition removes the rent, not the cost of informed traders."
                ),
            )
        )
    v = results.load("vol_premium")
    if v:
        kf = v["kelly_full"]
        out.append(
            (
                "Volatility premium",
                "06_vol_premium",
                (
                    f"VIX sat above the volatility that followed <span class=num>{100 * v['share_implied_above']:.0f}%</span> "
                    f"of the time since 1990. But the mean/variance Kelly formula asks for "
                    f"<span class=num>{kf['v_continuous'] / kf['v_star']:.1f}x</span> the growth-optimal size, and every "
                    "Kelly fraction fitted on 1993–2007 was wiped out after 2008."
                ),
            )
        )
    b = results.load("cpp_bench")
    if b:
        out.append(
            (
                "C++ core",
                "02_microstructure",
                (
                    f"The order-book replay was ported to C++17: <span class=num>{b['speedup_min']:.0f}–"
                    f"{b['speedup_max']:.0f}x</span> faster than Python, with identical fills on every test."
                ),
            )
        )
    return out


def hero_tiles() -> str:
    tiles = []
    m, a, v = results.load("microstructure"), results.load("auction"), results.load("vol_premium")
    if m:
        intc = {r["model"]: r for r in m["fills"]["INTC"]["summary"]}
        tiles.append(
            (
                f"{intc['touch']['mk_10_bps']:+.2f} → {intc['fifo']['mk_10_bps']:+.2f} bps",
                "INTC resting orders ten seconds after the fill: touch-fill backtest vs a real queue",
            )
        )
    if a:  # one period throughout: the 240 walk-forward days
        s = a["selected"]["summary"]
        tiles.append(
            (
                f"${s['total_pnl_usd']:,.0f}",
                f"What the Sharpe-{s['sharpe_annualized']:.1f} closing-auction model nets in {s['days']} walk-forward "
                f"days: costs take {s['cost_usd'] / s['gross_usd']:.0%} of the gross",
            )
        )
    if v:
        kf = v["kelly_full"]
        tiles.append(
            (
                f"{kf['v_continuous'] / kf['v_star']:.1f}×",
                "How much the textbook Kelly formula overbets selling S&P 500 volatility",
            )
        )
    return "".join(
        f'<div class="tile"><b>{html.escape(n).replace("-", "&minus;")}</b><span>{html.escape(t)}</span></div>'
        for n, t in tiles
    )


def render() -> str:
    has_video = VIDEO.exists()
    preview = PREVIEW if (DOCS / PREVIEW).exists() else "img/h_kelly.png"
    cards = "".join(
        f'<div class="card"><h3><a href="{REPO}/blob/main/reports/{rep}.md">{html.escape(t)}</a></h3>'
        f"<p>{body}</p></div>"
        for t, rep, body in findings()
    )
    video_section = (
        ""
        if not has_video
        else '<section class="lab" id="video"><p class="eyebrow">Demo</p><h2>Watch the walkthrough</h2>'
        '<video controls preload="metadata" playsinline poster="media/markout-demo.jpg" '
        'src="media/markout-demo.mp4"></video></section>'
    )
    data = json.dumps(lab_data(), separators=(",", ":")).replace("</", "<\\/")
    fills = {
        "description": html.escape(DESCRIPTION),
        "site": SITE,
        "preview": preview,
        "icon": ICON,
        "repo": REPO,
        "paper_href": PAPER_URL or f"{REPO}/blob/main/reports/02_microstructure.md",
        "paper_label": "Read the paper" if PAPER_URL else "Read the study",
        "video_button": '<a class="btn" href="#video">Watch the demo</a>' if has_video else "",
        "hero_tiles": hero_tiles(),
        "findings_cards": cards,
        "video_section": video_section,
        "built_with": "".join(f"<span>{html.escape(b)}</span>" for b in BUILT_WITH),
        "generated": date.today().strftime("%B %Y"),
        "lab_json": data,
    }
    page = TEMPLATE.read_text()
    for key, value in fills.items():
        page = page.replace("{{" + key + "}}", value)
    if "{{" in page:
        raise ValueError("unfilled placeholder in site_template.html")
    return page


DESK_NOTICE = (
    '<div style="padding:8px 20px;font-size:13px;background:var(--warn-bg);color:var(--warn-ink)">'
    "<b>A demo that runs in your browser,</b> on a fictional 8-market scenario with paper money. The desk's own Python runs "
    "in this tab, so nothing you do is sent anywhere. Move the clock with <b>+2 h</b> / <b>+1 day</b>: prices drift, "
    'markets resolve and the journal scores your calls. <a href="../">Back to Markout</a>. <span id="sbx"></span></div>'
)


def desk_modules() -> list[Path]:
    """The package files the in-browser desk imports, found by importing it in a fresh interpreter."""
    probe = (
        "import sys, markout.cup.browser\n"
        "for m in list(sys.modules.values()):\n"
        "    if getattr(m, '__name__', '').split('.')[0] == 'markout' and getattr(m, '__file__', None):\n"
        "        print(m.__file__)"
    )
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True, env=env).stdout
    return sorted(Path(f).resolve() for f in out.split())


def desk_bundle() -> bytes:
    """markout-desk.zip: those modules plus the scenario and the journal template, laid out as
    in the repository (Pyodide unpacks it to /app, so markout.paths.ROOT is /app)."""
    from markout.cup.paper import SCENARIO
    from markout.decision.journal import TEMPLATE

    files = [p.relative_to(ROOT) for p in desk_modules()] + [SCENARIO.relative_to(ROOT), TEMPLATE.relative_to(ROOT)]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for rel in sorted(files):
            info = zipfile.ZipInfo(rel.as_posix(), date_time=(1980, 1, 1, 0, 0, 0))  # no timestamps: reproducible
            info.compress_type, info.external_attr = zipfile.ZIP_DEFLATED, 0o644 << 16
            z.writestr(info, (ROOT / rel).read_bytes())
    return buf.getvalue()


def desk_page() -> str:
    """docs/desk/index.html: the desk's own page, with the time controls and the Pyodide loader."""
    from markout.cup.public import CONTROLS, KILL_HELP
    from markout.cup.web import page

    html_page = page("in-browser", controls=CONTROLS, notice=DESK_NOTICE, kill_help=KILL_HELP, icon=ICON)
    description = (
        '<meta name="description" content="The Markout trading desk, running in your browser: '
        'fictional markets, paper money and the real decision loop.">'
    )
    return html_page.replace("</title>", "</title>" + description, 1).replace(
        "<script>", '<script src="loader.js"></script><script>', 1
    )


def build() -> str:
    DOCS.mkdir(parents=True, exist_ok=True)
    DESK.mkdir(exist_ok=True)
    (DESK / "index.html").write_text(desk_page())
    (DESK / "markout-desk.zip").write_bytes(desk_bundle())
    (DOCS / "img").mkdir(exist_ok=True)
    for suffix in ("", "_dark"):  # fallback link-preview image
        src = FIGURES / f"h_kelly{suffix}.png"
        if src.exists():
            shutil.copy(src, DOCS / "img" / src.name)
    (DOCS / ".nojekyll").write_text("")
    return render()


def main() -> None:
    (DOCS / "index.html").write_text(build())
    print("wrote docs/index.html")


if __name__ == "__main__":
    main()
