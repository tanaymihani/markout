// Markout lab: wires the page to the simulator (queue-sim.js), the maths (lab-math.js), the
// charts (charts.js) and the results embedded by `python -m markout.site`.
(function () {
  "use strict";
  const D = JSON.parse(document.getElementById("lab-data").textContent);
  const Q = window.MarkoutQueue, M = window.MarkoutMath, C = window.MarkoutCharts;
  const $ = (s) => document.querySelector(s);
  const minus = (s) => String(s).replace(/-/g, "−");
  const signed = (x, d = 2) => minus((x > 0 ? "+" : "") + x.toFixed(d));
  const pct = (x, d = 0) => (100 * x).toFixed(d) + "%";
  const money = (x) => minus((x < 0 ? "-$" : "$") + Math.abs(Math.round(x)).toLocaleString("en-US"));
  const color = (i) => C.cssVar(`--s${i}`);
  const MODEL = { touch: "Touch-fill backtest", fifo: "Real queue (FIFO)", through: "Trade-through" };

  function seg(container, options, current, onPick) {
    container.innerHTML = "";
    for (const [key, label] of options) {
      const b = document.createElement("button");
      b.type = "button"; b.textContent = label; b.dataset.key = key;
      b.setAttribute("aria-pressed", String(key === current));
      b.addEventListener("click", () => onPick(key));
      container.appendChild(b);
    }
  }
  const pressed = (container, key) => container.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.key === key)));

  function fillKeys() {
    const intc = D.micro && D.micro.stocks.INTC, a = D.auction, v = D.vol;
    const vals = {};
    if (intc) Object.assign(vals, { intc_touch_fill: pct(intc.fills.touch.p), intc_touch_mk: signed(intc.fills.touch.mk[2]),
      intc_fifo_fill: pct(intc.fills.fifo.p), intc_fifo_mk: signed(intc.fills.fifo.mk[2]) });
    if (a) Object.assign(vals, { auction_sharpe: a.sharpe.toFixed(1), auction_days: String(a.days),
      auction_adapt2: minus(a.adapted["2"].toFixed(2)), auction_adapt3: minus(a.adapted["3"].toFixed(2)),
      auction_holdout: `${a.holdout.sharpe.toFixed(1)} over ${a.holdout.days} days` });
    if (v) Object.assign(vals, { kelly_pre: v.kelly.pre.toFixed(2), kelly_cont: v.kelly.cont.toFixed(2) });
    document.querySelectorAll("[data-k]").forEach((el) => { if (vals[el.dataset.k] !== undefined) el.textContent = vals[el.dataset.k]; });
  }

  // ------------------------------------------------------------ lab 1: queue simulator
  const queue = (() => {
    const ahead = $("#q-ahead"), informed = $("#q-informed");
    let seed = 7, anim = null;
    const params = () => ({ queueAhead: +ahead.value, informedShare: +informed.value / 100 });
    const outputs = () => {
      $("#q-ahead-out").textContent = (+ahead.value).toLocaleString("en-US") + " shares";
      $("#q-informed-out").textContent = informed.value + "%";
    };
    function summary() {
      const r = Q.runMany(params(), 2000, seed);
      $("#q-results").hidden = false;
      for (const model of ["touch", "fifo"]) {
        const s = r[model];
        $(`#q-${model}-mk`).textContent = Number.isFinite(s.meanMarkout) ? `${signed(s.meanMarkout)} ticks` : "no fills";
        $(`#q-${model}-fill`).textContent = `average price move 10 s after the fill; ${pct(s.pFill)} of 2,000 orders filled`;
      }
    }
    function setStatus(model, text, tone) {
      const el = $(`#q-${model}`);
      el.classList.remove("good", "bad");
      if (tone) el.classList.add(tone);
      el.querySelector(".what").textContent = text;
    }
    function reset(p) {
      const bar = $("#q-aheadbar").parentElement;
      bar.classList.remove("swept", "left");
      $("#q-aheadbar").style.width = "92%";
      $("#q-askbar").style.width = "60%";
      $("#q-log").innerHTML = "";
      $("#q-clock").textContent = "0.0 s";
      $("#q-aheadtxt").textContent = `${p.queueAhead.toLocaleString("en-US")} shares ahead of you`;
      setStatus("touch", "waiting"); setStatus("fifo", "waiting");
    }
    function watch() {
      if (anim) cancelAnimationFrame(anim);
      const p = params(), r = Q.runOne(p, Q.mulberry32((Math.random() * 2 ** 31) | 0), true);
      const P = r.params, end = P.lifetime + P.horizon, speed = 9;
      reset(p);
      let shown = 0, t0 = null;
      const bar = $("#q-aheadbar").parentElement;
      const aheadAt = (t) => {
        let last = r.log[0];
        for (const e of r.log) { if (e.t <= t) last = e; else break; }
        if (r.fifo && t >= r.fifo.t) return 0;
        return last.ahead * Math.exp(-P.cancelRate * (t - last.t));
      };
      const frame = (now) => {
        if (t0 === null) t0 = now;
        const t = Math.min(((now - t0) / 1000) * speed, end);
        while (shown < r.log.length && r.log[shown].t <= t) {
          const e = r.log[shown++];
          if (e.kind !== "join") {
            const li = document.createElement("li");
            li.innerHTML = `<b>${e.t.toFixed(1)} s</b> ${e.text}`;
            $("#q-log").prepend(li);
          }
          if (e.kind === "trade") { bar.classList.remove("flash"); void bar.offsetWidth; bar.classList.add("flash"); }
          if (e.kind === "sweep") bar.classList.add("swept");
          if (e.kind === "up") { bar.classList.add("left"); $("#q-askbar").style.width = "0%"; }
        }
        const a = aheadAt(t);
        $("#q-aheadbar").style.width = `${Math.max(0, (a / Math.max(p.queueAhead, 1)) * 92)}%`;
        $("#q-clock").textContent = `${Math.min(t, end).toFixed(1)} s`;
        $("#q-aheadtxt").textContent = r.fifo && t >= r.fifo.t ? "your order has reached the front" : `${Math.round(a).toLocaleString("en-US")} shares ahead of you`;
        for (const model of ["touch", "fifo"]) {
          const f = r[model];
          if (f && t >= f.t + P.horizon) {
            setStatus(model, `filled at ${f.t.toFixed(1)} s; 10 s later the price is ${signed(f.markout, 1)} ticks from your bid`, f.markout > 0 ? "good" : "bad");
          } else if (f && t >= f.t) {
            setStatus(model, `filled at ${f.t.toFixed(1)} s${f.how === "sweep" ? " (the level was swept)" : ""}`);
          } else if (t >= P.lifetime || (!f && r.log.some((e) => e.kind === "up" && e.t <= t))) {
            setStatus(model, r.log.some((e) => e.kind === "up" && e.t <= t) ? "not filled: the price moved away" : "not filled within 60 s");
          } else if (model === "fifo") {
            setStatus(model, "waiting in the queue");
          }
        }
        if (t < end) anim = requestAnimationFrame(frame);
      };
      anim = requestAnimationFrame(frame);
    }
    function init() {
      outputs(); reset(params()); summary();
      for (const el of [ahead, informed]) el.addEventListener("input", () => { outputs(); summary(); reset(params()); });
      $("#q-watch").addEventListener("click", watch);
      $("#q-run").addEventListener("click", () => { seed = (seed * 16807) % 2147483647; summary(); });
    }
    return { init };
  })();

  // ------------------------------------------------------------ lab 2: replay results
  const fills = (() => {
    if (!D.micro) return { init() {}, draw() {} };
    const S = D.micro.stocks;
    let stock = "INTC", costs = "gross", horizon = "gross";
    const COSTS = [["gross", "No fees"], ["r02", "Fee + 0.2-tick rebate"], ["r03", "Fee + 0.3-tick rebate"]];
    const HORIZ = [["h10", "10 s"], ["gross", "30 s"], ["h60", "60 s"]];
    function summary() {
      const s = S[stock];
      $("#f-summary").innerHTML = s.class === "large"
        ? `<b>${stock}</b> is a large-tick stock: its spread is almost always one tick (${s.spread.toFixed(2)} on average) and about ${s.depth.toLocaleString("en-US")} shares wait at each best quote. Queue imbalance predicted its next price move with an AUC of <b>${s.auc.toFixed(3)}</b> on the afternoon.`
        : `<b>${stock}</b> is a small-tick stock: its spread averages ${s.spread.toFixed(1)} ticks and only about ${s.depth.toLocaleString("en-US")} shares wait at each best quote. Queue imbalance predicted its next price move with an AUC of <b>${s.auc.toFixed(3)}</b> on the afternoon.`;
    }
    function drawMarkouts() {
      const s = S[stock], H = D.micro.horizons;
      const series = [["touch", 0, ""], ["fifo", 1, "7 5"], ["through", 2, "2 4"]].map(([m, i, dash]) => (
        { name: MODEL[m], color: color(i), dash, markers: true, points: H.map((h, k) => [h, s.fills[m].mk[k]]) }));
      C.lineChart($("#f-markouts"), {
        series, height: 290, label: `Markouts of filled orders for ${stock}`,
        x: { type: "log", min: 0.08, max: 75, ticks: H, format: (v) => `${v} s`, label: "time after the fill" },
        y: { format: (v) => signed(v, Math.abs(v) < 1 ? 2 : 1), label: "ticks per share, your side" },
        refs: [{ y: s.uncond, label: "a randomly timed fill earns half the spread", dash: "2 5" }, { y: 0, dash: "", width: 1, color: C.cssVar("--baseline") }],
        tipX: (v) => `${v} s after the fill`,
      });
    }
    function drawRules() {
      const s = S[stock];
      const key = costs === "gross" ? horizon : costs;
      const r = s.rules && s.rules[key];
      if (!r) { $("#f-rules").innerHTML = "<p class='small'>Run paper/scripts/rebates.py and robustness.py to add these results.</p>"; return; }
      const item = (m, which) => ({ name: MODEL[m], color: color(m === "touch" ? 0 : 1), marker: m === "touch" ? "circle" : "square",
        value: r[m][which][0], lo: r[m][which][1], hi: r[m][which][2] });
      C.dotWhisker($("#f-rules"), {
        rows: [{ label: "Always post", items: [item("touch", "always"), item("fifo", "always")] },
               { label: "Chosen rule", items: [item("touch", "rule"), item("fifo", "rule")] }],
        format: (v) => signed(v, 2), xLabel: "out-of-sample value per decision, ticks per share (95% interval)",
        label: `Out-of-sample results for ${stock}`,
      });
      const at = costs === "gross" ? `with ${HORIZ.find((h) => h[0] === horizon)[1]} horizons and no fees` : `with the 0.3-tick fee on crosses and a ${costs === "r02" ? "0.2" : "0.3"}-tick rebate on posted fills`;
      $("#f-rules-summary").innerHTML = `For ${stock} ${at}, always posting earns <b>${signed(r.touch.always[0], 3)}</b> ticks per decision under touch fills and <b>${signed(r.fifo.always[0], 3)}</b> under the real queue. The chosen rule posts on ${pct(r.touch.post)} of decisions under touch fills and ${pct(r.fifo.post)} under the real queue.`;
    }
    function draw() { summary(); drawMarkouts(); drawRules(); }
    function init() {
      seg($("#f-stock"), D.micro.tickers.map((t) => [t, t]), stock, (k) => { stock = k; pressed($("#f-stock"), k); draw(); });
      seg($("#f-costs"), COSTS, costs, (k) => {
        costs = k; pressed($("#f-costs"), k);
        if (k !== "gross") { horizon = "gross"; pressed($("#f-horizon"), "gross"); }
        $("#f-horizon").querySelectorAll("button").forEach((b) => { b.disabled = k !== "gross" && b.dataset.key !== "gross"; });
        drawRules();
      });
      seg($("#f-horizon"), HORIZ, horizon, (k) => { horizon = k; pressed($("#f-horizon"), k); drawRules(); });
      draw();
    }
    return { init, draw };
  })();

  // ------------------------------------------------------------ lab 3: cost slider
  const auction = (() => {
    if (!D.auction) return { init() {}, draw() {} };
    const A = D.auction, el = $("#a-mult"), base = M.costCurve(A.gross, A.cost, 1);
    function draw() {
      const mult = +el.value, c = M.costCurve(A.gross, A.cost, mult);
      $("#a-mult-out").textContent = `${mult.toFixed(2)}× the assumed spread and fees`;
      $("#a-sharpe").textContent = minus(c.sharpe.toFixed(2));
      $("#a-total").textContent = money(c.total);
      const days = c.cum.map((v, i) => [i + 1, v]);
      C.lineChart($("#a-chart"), {
        height: 280, label: "Cumulative profit of the closing-auction strategy",
        series: [{ name: `at ${mult.toFixed(2)}× costs`, color: color(0), points: days },
                 ...(Math.abs(mult - 1) > 1e-9 ? [{ name: "at the assumed costs", color: C.cssVar("--muted"), dash: "6 5", width: 1.6, points: base.cum.map((v, i) => [i + 1, v]) }] : [])],
        x: { min: 1, max: A.days, ticks: [1, 40, 80, 120, 160, 200, 240], format: (v) => `day ${v}`, label: "trading day of the research period" },
        y: { format: money, label: "cumulative profit" },
        refs: [{ y: 0, dash: "", width: 1, color: C.cssVar("--baseline") }],
      });
    }
    return { init() { el.addEventListener("input", draw); draw(); }, draw };
  })();

  // ------------------------------------------------------------ lab 4: Kelly sizing
  const kelly = (() => {
    if (!D.vol) return { init() {}, draw() {} };
    const V = D.vol, el = $("#k-size");
    const years = V.dates.map((d) => { const [y, m, dd] = d.split("-").map(Number); return y + (m - 1) / 12 + (dd - 1) / 365; });
    const PRESETS = [[V.kelly.pre, "Kelly fitted on 1993–2007"], [V.kelly.pre / 2, "half of that"], [V.kelly.pre / 4, "a quarter"],
                     [V.kelly.full, "exact Kelly, with hindsight"], [V.kelly.cont, "textbook mean/variance"]];
    function draw() {
      const v = +el.value, w = M.wealthPath(V.pnl, v / 100);
      $("#k-size-out").textContent = `$${v.toFixed(2)} of vega per $100`;
      const ruinIdx = w.findIndex((x) => x <= 0);
      const floor = 1e-3;
      const pts = w.map((x, i) => [years[i], Math.max(x, floor)]);
      $("#k-final").textContent = ruinIdx >= 0 ? "wiped out" : `×${w[w.length - 1] >= 100 ? Math.round(w[w.length - 1]).toLocaleString("en-US") : w[w.length - 1].toFixed(2)}`;
      $("#k-dd").textContent = minus((100 * M.maxDrawdown(w)).toFixed(0) + "%");
      $("#k-ruin").textContent = ruinIdx >= 0 ? `wiped out in ${new Date(V.dates[ruinIdx]).toLocaleString("en-US", { month: "long", year: "numeric", timeZone: "UTC" })}` : "survived";
      const ymax = Math.max(10, ...pts.map((p) => p[1]));
      C.lineChart($("#k-chart"), {
        height: 300, label: "Wealth path of the variance-selling strategy",
        series: [{ name: `$${v.toFixed(2)} of vega per $100`, color: color(0), points: pts }],
        x: { min: years[0], max: years[years.length - 1], ticks: [1995, 2000, 2005, 2010, 2015, 2020, 2025], format: (y) => String(Math.round(y)), label: "" },
        y: { type: "log", min: floor, max: ymax * 1.5, format: (x) => (x >= 1 ? "×" + Math.round(x).toLocaleString("en-US") : "×" + x), label: "wealth (start = 1, log scale)" },
        refs: [{ y: 1, dash: "", width: 1, color: C.cssVar("--baseline") }],
        vlines: [{ x: 2008.7, label: "2008" }, { x: 2020.15, label: "2020" }],
        tipX: (y) => `${Math.floor(y)}`,
      });
    }
    function init() {
      const box = $("#k-presets");
      for (const [val, label] of PRESETS) {
        const b = document.createElement("button");
        b.type = "button"; b.textContent = `${label} · ${val.toFixed(2)}`;
        b.addEventListener("click", () => { el.value = val.toFixed(2); draw(); });
        box.appendChild(b);
      }
      el.addEventListener("input", draw);
      draw();
    }
    return { init, draw };
  })();

  document.addEventListener("DOMContentLoaded", () => {
    fillKeys(); queue.init(); fills.init(); auction.init(); kelly.init();
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const redraw = () => { fills.draw(); auction.draw(); kelly.draw(); };
    if (mq.addEventListener) mq.addEventListener("change", redraw);
  });
})();
