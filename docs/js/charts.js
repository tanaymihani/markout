// Small SVG charts for the Markout lab: line charts with a hover crosshair, and dot-and-interval
// charts with per-mark tooltips. No dependencies. Colors come from CSS variables, so the
// charts follow the page's light and dark themes; identity never rests on color alone
// (lines also differ in dash pattern, dots in shape, and every chart has a legend).

(function (root) {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";

  function svgEl(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
    if (parent) parent.appendChild(e);
    return e;
  }

  const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  function niceTicks(lo, hi, count = 5) {
    if (!(hi > lo)) return [lo];
    const span = hi - lo, raw = span / count, mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= count) || 10 * mag;
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9 * step; v += step) out.push(Math.abs(v) < 1e-12 ? 0 : v);
    return out;
  }

  function makeScale(type, d0, d1, r0, r1) {
    if (type === "log") {
      const l0 = Math.log10(d0), l1 = Math.log10(d1);
      const f = (v) => r0 + ((Math.log10(Math.max(v, 1e-300)) - l0) / (l1 - l0)) * (r1 - r0);
      f.invert = (p) => Math.pow(10, l0 + ((p - r0) / (r1 - r0)) * (l1 - l0));
      return f;
    }
    const f = (v) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);
    f.invert = (p) => d0 + ((p - r0) / (r1 - r0)) * (d1 - d0);
    return f;
  }

  function tooltipFor(container) {
    let tip = container.querySelector(".chart-tip");
    if (!tip) { tip = document.createElement("div"); tip.className = "chart-tip"; tip.hidden = true; container.appendChild(tip); }
    return tip;
  }

  function placeTip(container, tip, px, py, html) {
    tip.innerHTML = html; tip.hidden = false;
    const cw = container.clientWidth, tw = tip.offsetWidth;
    tip.style.left = Math.min(Math.max(px + 12, 0), cw - tw - 4) + "px";
    tip.style.top = Math.max(py - 10, 0) + "px";
  }

  function legend(container, items) {
    let lg = container.querySelector(".chart-legend");
    if (!lg) { lg = document.createElement("div"); lg.className = "chart-legend"; container.appendChild(lg); }
    lg.innerHTML = items.map((it) =>
      `<span class="lg-item"><svg width="26" height="12" aria-hidden="true">` +
      (it.marker
        ? `<rect x="8" y="1" width="10" height="10" rx="${it.marker === "circle" ? 5 : 1.5}" fill="${it.filled === false ? "none" : it.color}" stroke="${it.color}" stroke-width="2"/>`
        : `<line x1="1" y1="6" x2="25" y2="6" stroke="${it.color}" stroke-width="2.5" stroke-dasharray="${it.dash || ""}"/>`) +
      `</svg>${it.name}</span>`).join("");
  }

  // series: [{name, color, dash, points: [[x, y], ...], hide?}], x/y: {type, min, max, ticks, format, label}
  function lineChart(container, opts) {
    const W = opts.width || 720, H = opts.height || 300, m = Object.assign({ l: 58, r: 18, t: 14, b: 44 }, opts.margin);
    container.classList.add("chart");
    container.querySelectorAll("svg.plot").forEach((n) => n.remove());
    const svg = svgEl("svg", { class: "plot", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": opts.label || "chart" });
    container.prepend(svg);
    const xs = opts.series.flatMap((s) => s.points.map((p) => p[0]));
    const ys = opts.series.flatMap((s) => s.points.map((p) => p[1])).concat((opts.refs || []).map((r) => r.y));
    const xo = opts.x || {}, yo = opts.y || {};
    const x0 = xo.min ?? Math.min(...xs), x1 = xo.max ?? Math.max(...xs);
    let y0 = yo.min ?? Math.min(...ys), y1 = yo.max ?? Math.max(...ys);
    if (yo.type !== "log") { const pad = (y1 - y0 || 1) * 0.08; y0 = yo.min ?? y0 - pad; y1 = yo.max ?? y1 + pad; }
    const yTicks = yo.ticks || (yo.type === "log" ? logTicks(y0, y1) : niceTicks(y0, y1, 5));
    const widest = Math.max(...yTicks.map((v) => String((yo.format || String)(v)).length));
    if (!(opts.margin && opts.margin.l)) m.l = Math.max(m.l, widest * 7.4 + (yo.label ? 34 : 14));
    const X = makeScale(xo.type, x0, x1, m.l, W - m.r), Y = makeScale(yo.type, y0, y1, H - m.b, m.t);
    const ink2 = cssVar("--ink2"), grid = cssVar("--grid"), base = cssVar("--baseline");
    for (const v of yTicks) {
      svgEl("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), stroke: grid, "stroke-width": 1 }, svg);
      const tx = svgEl("text", { x: m.l - 8, y: Y(v) + 4, "text-anchor": "end", class: "tick" }, svg);
      tx.textContent = (yo.format || String)(v);
    }
    const xTicks = xo.ticks || niceTicks(x0, x1, 6);
    for (const v of xTicks) {
      const tx = svgEl("text", { x: X(v), y: H - m.b + 18, "text-anchor": "middle", class: "tick" }, svg);
      tx.textContent = (xo.format || String)(v);
    }
    svgEl("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, stroke: base, "stroke-width": 1 }, svg);
    if (xo.label) { const t = svgEl("text", { x: (m.l + W - m.r) / 2, y: H - 6, "text-anchor": "middle", class: "axis" }, svg); t.textContent = xo.label; }
    if (yo.label) { const t = svgEl("text", { x: 14, y: (m.t + H - m.b) / 2, "text-anchor": "middle", class: "axis", transform: `rotate(-90 14 ${(m.t + H - m.b) / 2})` }, svg); t.textContent = yo.label; }
    for (const r of opts.refs || []) {
      svgEl("line", { x1: m.l, x2: W - m.r, y1: Y(r.y), y2: Y(r.y), stroke: r.color || ink2, "stroke-width": r.width || 1.2, "stroke-dasharray": r.dash ?? "5 4" }, svg);
      if (r.label) { const t = svgEl("text", { x: W - m.r, y: Y(r.y) - 6, "text-anchor": "end", class: "ref" }, svg); t.textContent = r.label; }
    }
    for (const v of opts.vlines || []) {
      svgEl("line", { x1: X(v.x), x2: X(v.x), y1: m.t, y2: H - m.b, stroke: ink2, "stroke-width": 1, "stroke-dasharray": "3 3" }, svg);
      if (v.label) { const t = svgEl("text", { x: X(v.x) + 5, y: m.t + 12, class: "ref" }, svg); t.textContent = v.label; }
    }
    for (const s of opts.series) {
      if (!s.points.length) continue;
      const d = s.points.map((p, i) => `${i ? "L" : "M"}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join("");
      svgEl("path", { d, fill: "none", stroke: s.color, "stroke-width": s.width || 2.2, "stroke-dasharray": s.dash || "", "stroke-linejoin": "round", "stroke-linecap": "round" }, svg);
      if (s.markers) for (const p of s.points) svgEl("circle", { cx: X(p[0]), cy: Y(p[1]), r: 4.5, fill: s.color, stroke: cssVar("--surface"), "stroke-width": 1.5 }, svg);
    }
    legend(container, opts.series.filter((s) => s.name).map((s) => ({ name: s.name, color: s.color, dash: s.dash })));
    // hover: crosshair at the nearest x, values of every series there
    const tip = tooltipFor(container);
    const cross = svgEl("line", { y1: m.t, y2: H - m.b, stroke: ink2, "stroke-width": 1, visibility: "hidden" }, svg);
    const hit = svgEl("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent" }, svg);
    const allX = [...new Set(xs)].sort((a, b) => a - b);
    const move = (ev) => {
      const box = svg.getBoundingClientRect(), sx = ((ev.clientX - box.left) / box.width) * W;
      const xv = X.invert(sx);
      let best = allX[0];
      for (const v of allX) if (Math.abs(X(v) - sx) < Math.abs(X(best) - sx)) best = v;
      cross.setAttribute("x1", X(best)); cross.setAttribute("x2", X(best)); cross.setAttribute("visibility", "visible");
      const rows = opts.series.map((s) => { const p = s.points.find((q) => q[0] === best) || nearest(s.points, best); return p ? `<div><i style="background:${s.color}"></i>${s.name}: <b>${(yo.format || String)(p[1])}</b></div>` : ""; }).join("");
      const head = (opts.tipX || xo.format || String)(best, xv);
      placeTip(container, tip, ((X(best)) / W) * box.width, ((ev.clientY - box.top)), `<div class="tip-h">${head}</div>${rows}`);
    };
    hit.addEventListener("mousemove", move);
    hit.addEventListener("mouseleave", () => { tip.hidden = true; cross.setAttribute("visibility", "hidden"); });
    return { X, Y, svg };
  }

  function nearest(points, x) {
    let best = null;
    for (const p of points) if (best === null || Math.abs(p[0] - x) < Math.abs(best[0] - x)) best = p;
    return best;
  }

  function logTicks(lo, hi) {
    const out = [];
    for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) { const v = Math.pow(10, e); if (v >= lo * 0.999 && v <= hi * 1.001) out.push(v); }
    return out.length ? out : [lo, hi];
  }

  // rows: [{label, items: [{name, color, marker, filled, value, lo, hi}]}]
  function dotWhisker(container, opts) {
    const rowH = opts.rowHeight || 64, per = Math.max(...opts.rows.map((r) => r.items.length));
    const W = opts.width || 720, m = Object.assign({ l: 92, r: 24, t: 10, b: 44 }, opts.margin);
    const H = m.t + m.b + rowH * opts.rows.length;
    container.classList.add("chart");
    container.querySelectorAll("svg.plot").forEach((n) => n.remove());
    const svg = svgEl("svg", { class: "plot", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": opts.label || "chart" });
    container.prepend(svg);
    const vals = opts.rows.flatMap((r) => r.items.flatMap((i) => [i.lo, i.hi, i.value])).concat([0]);
    let x0 = Math.min(...vals), x1 = Math.max(...vals); const pad = (x1 - x0 || 1) * 0.08; x0 -= pad; x1 += pad;
    const X = makeScale("linear", x0, x1, m.l, W - m.r), grid = cssVar("--grid"), ink2 = cssVar("--ink2");
    for (const v of niceTicks(x0, x1, 6)) {
      svgEl("line", { x1: X(v), x2: X(v), y1: m.t, y2: H - m.b, stroke: grid, "stroke-width": 1 }, svg);
      const t = svgEl("text", { x: X(v), y: H - m.b + 18, "text-anchor": "middle", class: "tick" }, svg);
      t.textContent = (opts.format || String)(v);
    }
    svgEl("line", { x1: X(0), x2: X(0), y1: m.t, y2: H - m.b, stroke: ink2, "stroke-width": 1.4 }, svg);
    if (opts.xLabel) { const t = svgEl("text", { x: (m.l + W - m.r) / 2, y: H - 6, "text-anchor": "middle", class: "axis" }, svg); t.textContent = opts.xLabel; }
    const tip = tooltipFor(container), surface = cssVar("--surface");
    opts.rows.forEach((row, j) => {
      const yc = m.t + rowH * (j + 0.5);
      const t = svgEl("text", { x: m.l - 12, y: yc + 4, "text-anchor": "end", class: "row" }, svg); t.textContent = row.label;
      row.items.forEach((it, k) => {
        const y = yc + (k - (per - 1) / 2) * Math.min(14, (rowH - 16) / per);
        svgEl("line", { x1: X(it.lo), x2: X(it.hi), y1: y, y2: y, stroke: it.color, "stroke-width": 2.5 }, svg);
        const fill = it.filled === false ? surface : it.color;
        const mark = it.marker === "square"
          ? svgEl("rect", { x: X(it.value) - 5, y: y - 5, width: 10, height: 10, rx: 1.5, fill, stroke: it.color, "stroke-width": 2 }, svg)
          : svgEl("circle", { cx: X(it.value), cy: y, r: 5.5, fill, stroke: it.color, "stroke-width": 2 }, svg);
        const hitbox = svgEl("rect", { x: Math.min(X(it.lo), X(it.value) - 8), y: y - 8, width: Math.max(X(it.hi) - X(it.lo), 16), height: 16, fill: "transparent" }, svg);
        const show = (ev) => {
          const box = svg.getBoundingClientRect();
          placeTip(container, tip, ev.clientX - box.left, ev.clientY - box.top,
            `<div class="tip-h">${row.label}: ${it.name}</div><div><b>${(opts.format || String)(it.value)}</b> (95% interval ${(opts.format || String)(it.lo)} to ${(opts.format || String)(it.hi)})</div>`);
        };
        for (const target of [mark, hitbox]) { target.addEventListener("mousemove", show); target.addEventListener("mouseleave", () => { tip.hidden = true; }); }
      });
    });
    const seen = new Map();
    for (const it of opts.rows.flatMap((r) => r.items)) if (!seen.has(it.name)) seen.set(it.name, { name: it.name, color: it.color, marker: it.marker || "circle", filled: it.filled });
    legend(container, [...seen.values()]);
    return { X, svg };
  }

  root.MarkoutCharts = { lineChart, dotWhisker, cssVar, niceTicks };
})(typeof window !== "undefined" ? window : globalThis);
