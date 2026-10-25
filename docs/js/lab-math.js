// Numbers behind the Markout lab's sliders. Each function mirrors the Python code that
// produced the reports, so the browser and the reports agree:
//   costCurve   <- reports/results/auction_daily_research.csv (same trades, costs scaled)
//   wealthPath  <- markout.vol.strategy.wealth_path (ruin is absorbing)
//   kellyExact  <- markout.vol.strategy.kelly_exact (grid search of mean log growth)

(function (root) {
  "use strict";

  const mean = (x) => x.reduce((a, b) => a + b, 0) / x.length;
  const variance = (x) => { const m = mean(x); return x.reduce((a, b) => a + (b - m) ** 2, 0) / (x.length - 1); };

  function sharpe(x, periodsPerYear) {
    const sd = Math.sqrt(variance(x));
    return sd > 0 ? (mean(x) / sd) * Math.sqrt(periodsPerYear) : 0;
  }

  // Daily PnL of the same trades when every cost is multiplied by `mult`.
  function costCurve(gross, cost, mult) {
    const pnl = gross.map((g, i) => g - mult * cost[i]);
    let c = 0;
    const cum = pnl.map((v) => (c += v));
    return { pnl, cum, total: c, sharpe: sharpe(pnl, 252) };
  }

  // Wealth after each period at v dollars of exposure per dollar of capital.
  function wealthPath(pnl, v) {
    let w = 1;
    return pnl.map((x) => (w = w * Math.max(1 + v * x, 0)));
  }

  function maxDrawdown(w) {
    let peak = 1e-300, worst = 0;
    for (const x of w) { peak = Math.max(peak, x); worst = Math.min(worst, x / peak - 1); }
    return worst;
  }

  // Growth-optimal v on the empirical outcomes; v must keep 1 + v * min(pnl) > 0.
  function kellyExact(pnl, points = 2001) {
    const worst = Math.min(...pnl);
    const vmax = 0.999 / Math.max(-worst, 1e-9);
    let best = { v: 0, g: 0 };
    for (let i = 0; i < points; i++) {
      const v = (vmax * i) / (points - 1);
      const g = mean(pnl.map((x) => Math.log1p(v * x)));
      if (g > best.g) best = { v, g };
    }
    const vr = variance(pnl);
    return { vStar: best.v, growth: best.g, vMax: vmax, vContinuous: vr > 0 ? mean(pnl) / vr : null };
  }

  root.MarkoutMath = { mean, variance, sharpe, costCurve, wealthPath, maxDrawdown, kellyExact };
})(typeof window !== "undefined" ? window : globalThis);
