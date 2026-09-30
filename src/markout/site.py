"""Build the GitHub Pages site in docs/: `python -m markout.site`.

A landing page with the main findings (numbers read from reports/results/), a few
figures, the playable market-making game (docs/game.js), and the desk snapshot.
"""

from __future__ import annotations

import html
import shutil

from markout import results
from markout.paths import FIGURES, ROOT

DOCS = ROOT / "docs"
REPO = "https://github.com/tanaymihani/markout"
SITE = "https://tanaymihani.github.io/markout/"
DESCRIPTION = ("Does a short-horizon trading edge survive the spread, realistic fills, competition and selection bias? "
               "Six studies, mostly on real market data, and a trading desk for SIG's Predictions Cup.")
PREVIEW = "h_kelly"  # link-preview image (about the 1.91:1 shape preview cards use)
ICON = ("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'>"
        "<rect width='32' height='32' rx='7' fill='%232a78d6'/><polyline points='6,22 12,15 17,19 26,9' fill='none' "
        "stroke='white' stroke-width='3' stroke-linecap='round' stroke-linejoin='round'/></svg>")
FIGS = [("b_voi", "Net edge per trade against forecast quality (report 01)", "01_auction"),
        ("d_markouts", "Markouts of filled orders under three fill models (report 02)", "02_microstructure"),
        ("e_arena_competition", "Spread and maker profit as market makers compete (report 03)", "03_arena"),
        ("h_kelly", "Kelly sizing chosen before 2008, lived through it (report 06)", "06_vol_premium"),
        ("h_straddle", "A delta-hedged straddle against a variance swap, month by month since 1993 (report 06)",
         "06_vol_premium")]

CSS = """
:root{--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;
--border:rgba(11,11,11,.10);--accent:#2a78d6;--good:#006300;--bad:#d03b3b}
@media (prefers-color-scheme:dark){:root{--page:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;
--muted:#898781;--grid:#2c2c2a;--border:rgba(255,255,255,.10);--accent:#3987e5;--good:#0ca30c;--bad:#e66767}}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);
font:16px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:960px;margin:0 auto;padding:40px 20px 80px}
h1{font-size:34px;margin:0 0 6px;letter-spacing:-.01em}h2{font-size:21px;margin:44px 0 12px}
p{margin:0 0 14px;color:var(--ink)}.lede{font-size:18px;color:var(--ink2)}a{color:var(--accent)}
.links{display:flex;gap:16px;flex-wrap:wrap;margin:14px 0 8px;font-size:15px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:16px 18px}
.card h3{margin:0 0 6px;font-size:15px}.card p{font-size:15px;margin:0;color:var(--ink2)}
.num{font-variant-numeric:tabular-nums;color:var(--ink);font-weight:600}
figure{margin:18px 0 26px}figure img{width:100%;height:auto;border-radius:10px;border:1px solid var(--border)}
figcaption{font-size:14px;color:var(--muted);margin-top:6px}
.game{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:18px}
.cards{display:flex;gap:10px;flex-wrap:wrap;margin:10px 0 14px}
.card.down,.cards .card{display:inline-flex;flex-direction:column;align-items:center;justify-content:center;width:54px;height:76px;
border-radius:8px;border:1px solid var(--border);background:var(--page);font-size:22px;font-weight:600;padding:0}
.cards .card small{font-size:16px;font-weight:400}.cards .card.red{color:var(--bad)}
.cards .card.down{background:repeating-linear-gradient(45deg,var(--grid),var(--grid) 6px,var(--surface) 6px,var(--surface) 12px)}
.controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:8px 0}
input[type=number]{width:96px;font:inherit;padding:6px 8px;border-radius:8px;border:1px solid var(--border);
background:var(--page);color:var(--ink);font-variant-numeric:tabular-nums}
button{font:inherit;border-radius:8px;border:1px solid var(--border);background:var(--surface);color:var(--ink);
padding:7px 14px;cursor:pointer}button.primary{background:var(--accent);color:#fff;border-color:transparent;font-weight:600}
button:disabled{opacity:.45;cursor:not-allowed}.err{color:var(--bad);font-size:14px;min-height:20px}
.muted{color:var(--muted);font-size:14px}.log .entry{font-size:14px;padding:8px 0;border-top:1px solid var(--grid)}
.hint{color:var(--ink2)}.final{margin-top:12px;padding:12px;border-radius:10px;background:var(--page);border:1px solid var(--border)}
footer{margin-top:60px;font-size:14px;color:var(--muted)}
"""


def picture(name: str, alt: str) -> str:
    return (f'<picture><source media="(prefers-color-scheme: dark)" srcset="img/{name}_dark.png">'
            f'<img src="img/{name}.png" alt="{html.escape(alt)}" loading="lazy"></picture>')


def findings() -> list[tuple[str, str, str]]:
    out = []
    a = results.load("auction")
    if a:
        s, h = a["selected"]["summary"], (a.get("holdout") or {}).get("adapted_1x", {})
        two = next(r for r in a["robustness"]["cost_rows"] if r["cost_mult"] == 2.0)["adapted"]["sharpe_annualized"]
        out.append(("Closing auction", "01_auction",
                    f"LightGBM beats a per-stock baseline on every walk-forward day. Traded net of costs it earns an "
                    f"annualized Sharpe of <span class=num>{s['sharpe_annualized']:.1f}</span>, and "
                    f"<span class=num>{h.get('sharpe_annualized', float('nan')):.1f}</span> on a holdout scored once. "
                    f"At twice the assumed costs it drops to <span class=num>{two:.1f}</span>: the edge is about one "
                    "spread wide."))
    m = results.load("microstructure")
    if m:
        intc = {r["model"]: r for r in m["fills"]["INTC"]["summary"]}
        out.append(("Realistic fills", "02_microstructure",
                    f"Resting orders in INTC look profitable if any trade at your price fills you "
                    f"(<span class=num>{intc['touch']['mk_10_bps']:+.2f}</span> bps ten seconds later). Tracking queue "
                    f"position order by order turns that into <span class=num>{intc['fifo']['mk_10_bps']:+.2f}</span> bps: "
                    "the fills you actually get are the ones the market was about to run over."))
    ar = results.load("arena")
    if ar:
        k = {r["K"]: r for r in ar["arena"]["k_sweep"]}
        out.append(("Competition", "03_arena",
                    f"A lone market maker quotes a <span class=num>{100 * k[1]['quoted_spread']['mean']:.0f}</span>-tick "
                    f"spread; one identical rival brings it to <span class=num>{100 * k[2]['quoted_spread']['mean']:.1f}</span>, "
                    f"next to the zero-profit level of <span class=num>{100 * k[2]['zero_profit_spread']['mean']:.1f}</span>. "
                    "Competition removes the rent, not the cost of informed traders."))
    v = results.load("vol_premium")
    if v:
        kf = v["kelly_full"]
        out.append(("Volatility premium", "06_vol_premium",
                    f"VIX sat above the volatility that followed <span class=num>{100 * v['share_implied_above']:.0f}%</span> "
                    f"of the time since 1990. But the mean/variance Kelly formula asks for "
                    f"<span class=num>{kf['v_continuous'] / kf['v_star']:.1f}x</span> the growth-optimal size, and every "
                    "Kelly fraction fitted on 1993–2007 was wiped out after 2008."))
    b = results.load("cpp_bench")
    if b:
        out.append(("C++ core", "02_microstructure",
                    f"The order-book replay was ported to C++17: <span class=num>{b['speedup_min']:.0f}–"
                    f"{b['speedup_max']:.0f}x</span> faster than Python, with identical fills on every test."))
    return out


GAME = """
<div class="game" id="game">
  <div class="controls"><b id="round"></b><span class="muted" id="known"></span></div>
  <div class="cards" id="cards"></div>
  <div class="controls">
    <label>Bid <input type="number" id="bid" step="0.5"></label>
    <label>Ask <input type="number" id="ask" step="0.5"></label>
    <button class="primary" id="quote">Quote</button>
    <button id="new">New game</button>
  </div>
  <div class="err" id="err"></div>
  <div id="result"></div>
  <div class="log" id="log"></div>
</div>
"""


def build() -> str:
    DOCS.mkdir(parents=True, exist_ok=True)
    img = DOCS / "img"
    img.mkdir(exist_ok=True)
    for name in dict.fromkeys([n for n, _, _ in FIGS] + [PREVIEW]):
        for suffix in ("", "_dark"):
            src = FIGURES / f"{name}{suffix}.png"
            if src.exists():
                shutil.copy(src, img / src.name)
    (DOCS / "site.css").write_text(CSS.strip() + "\n")
    (DOCS / ".nojekyll").write_text("")
    cards = "".join(f'<div class="card"><h3><a href="{REPO}/blob/main/reports/{rep}.md">{html.escape(t)}</a></h3>'
                    f"<p>{body}</p></div>" for t, rep, body in findings())
    figs = "".join(f'<figure><a href="{REPO}/blob/main/reports/{rep}.md">{picture(n, alt)}</a>'
                   f"<figcaption>{html.escape(alt)}</figcaption></figure>"
                   for n, alt, rep in FIGS if (FIGURES / f"{n}.png").exists())
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Markout</title><meta name="description" content="{html.escape(DESCRIPTION)}">
<meta property="og:type" content="website"><meta property="og:title" content="Markout">
<meta property="og:description" content="{html.escape(DESCRIPTION)}"><meta property="og:url" content="{SITE}">
<meta property="og:image" content="{SITE}img/{PREVIEW}.png"><meta name="twitter:card" content="summary_large_image">
<link rel="icon" href="{ICON}"><link rel="stylesheet" href="site.css"></head>
<body><main>
<h1>Markout</h1>
<p class="lede">Does a short-horizon trading edge survive the spread, realistic fills, competing traders, and the bias
that comes from trying many strategies? These are the results of trying to find out, mostly on real market data.</p>
<div class="links"><a href="{REPO}">Code on GitHub</a><a href="{REPO}#results">All results</a>
<a href="demo/desk.html">Predictions Cup desk (snapshot)</a><a href="#play">Play the market-making game</a></div>

<h2>What came out of it</h2>
<div class="grid">{cards}</div>

<h2>A few figures</h2>
{figs}

<h2 id="play">Play: make a market on five hidden cards</h2>
<p>Five cards are dealt face down from a standard deck (A = 1 to K = 13). Quote a bid and an ask on their sum, at most
four apart. Three bots trade against you each round: some know one of the hidden cards and only trade when your quote
is wrong, the rest trade at random. One card is turned up after every round, and every trade settles at the final sum.
After each round the game shows the best estimate given everything you have seen, and at the end it replays the same
cards and bots with two scripted quoters so you can compare.</p>
{GAME}

<h2>The Predictions Cup desk</h2>
<p>SIG's student prediction contest allows bots. The desk pools free reference prices (Polymarket, Kalshi,
option-implied probabilities, earnings history) with the contest's own price, proposes trades sized by Kelly and by
what the order book can fill, and never trades without a click. <a href="demo/desk.html">Open a read-only snapshot</a>
or read <a href="{REPO}/blob/main/docs/demo/DEMO.md">the walkthrough</a>.</p>
<figure><img src="demo/desk.png" alt="The approval desk" loading="lazy"><figcaption>The approval desk on a paper
contest</figcaption></figure>

<footer>Markout · <a href="{REPO}">github.com/tanaymihani/markout</a></footer>
</main><script src="game.js"></script></body></html>
"""


def main() -> None:
    (DOCS / "index.html").write_text(build())
    print("wrote docs/index.html")


if __name__ == "__main__":
    main()
