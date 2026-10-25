// Queue simulator for the Markout lab: why queue position decides which fills you get.
//
// A small, transparent model of one price level, the best bid, with our 100-share buy order
// joining the back of the queue. Time runs in steps of dt seconds.
//   - Sell orders hit the bid at rate sellRate, with exponential sizes. A share informedShare
//     of them come from informed sellers: an informed seller keeps selling and clears the
//     whole level within a few seconds (a "sweep"), the mid falls half a tick, and sometimes
//     (followThrough) it falls one more tick.
//   - The queue ahead of us shrinks through cancellations (cancelRate per second).
//   - Buyers may clear the ask instead (upRate): the mid rises a tick and our bid is left
//     behind, so it can no longer fill.
// A touch fill happens at the first trade at our price. A FIFO fill happens only once the
// queue ahead of us is used up, or when the level is swept. Our order never affects the
// market, as in the replay study. The markout is side * (mid 10 s after the fill - price),
// in ticks. The defaults were tuned on a small grid so that this toy model roughly reproduces
// INTC's measured fill rates and 10 s markouts on 21 June 2012 (touch 68%, +0.09; FIFO 48%, -0.44).

(function (root) {
  "use strict";

  const DEFAULTS = {
    queueAhead: 13600,    // shares ahead of us when we join (INTC's mean depth at the best)
    sellRate: 0.04,       // sell orders hitting our level, per second
    sellSize: 1200,       // their mean size, shares
    informedShare: 0.4,   // share of sell orders that are informed and end in a sweep
    sweepDelay: 4,        // seconds within which an informed seller clears the level
    followThrough: 0.05,  // chance that a sweep is followed by a further one-tick fall
    cancelRate: 0.04,     // share of the queue ahead that cancels, per second
    upRate: 0.014,        // the ask is cleared and the price moves up, per second
    lifetime: 60,         // seconds the order rests
    horizon: 10,          // markout horizon after the fill, seconds
    dt: 0.05,
  };

  function mulberry32(seed) {
    return function () {
      seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  const fmt = (n) => Math.round(n).toLocaleString("en-US");

  // One order. Returns the outcome under each fill model, the mid path and an event log.
  function runOne(params, rng, keepLog = false) {
    const p = Object.assign({}, DEFAULTS, params);
    const steps = Math.round((p.lifetime + p.horizon) / p.dt);
    let ahead = p.queueAhead, alive = true, mid = 0.5;    // our bid is at price 0, the ask at 1
    let touch = null, fifo = null, sweepAt = null, fallAt = null;
    const mids = [[0, mid]], log = [];
    const note = (t, kind, text) => { if (keepLog) log.push({ t, kind, text, ahead: Math.max(ahead, 0) }); };
    note(0, "join", `you join the bid behind ${fmt(ahead)} shares`);
    for (let k = 1; k <= steps; k++) {
      const t = k * p.dt, open = t <= p.lifetime + 1e-9;
      if (fallAt !== null && t >= fallAt) {
        mid -= 1; fallAt = null; mids.push([t, mid]);
        note(t, "move", "the price falls another tick");
      }
      if (!alive) continue;
      if (sweepAt !== null && t >= sweepAt) {             // the informed seller clears the level
        if (open && touch === null) touch = { t, how: "sweep" };
        if (open && fifo === null) fifo = { t, how: "sweep" };
        alive = false; mid = -0.5; mids.push([t, mid]);
        if (rng() < p.followThrough) fallAt = t + rng() * p.horizon;
        note(t, "sweep", "an informed seller clears the whole bid; the price falls");
        continue;
      }
      ahead -= ahead * p.cancelRate * p.dt;
      if (rng() < p.upRate * p.dt) {                      // buyers clear the ask: we are left behind
        alive = false; mid = 1.5; mids.push([t, mid]);
        note(t, "up", "buyers clear the ask; the price rises and leaves your bid behind");
        continue;
      }
      if (rng() < p.sellRate * p.dt) {                    // a sell order hits the bid
        const size = Math.max(100, Math.round(-Math.log(1 - rng()) * p.sellSize / 100) * 100);
        const informed = rng() < p.informedShare;
        if (informed && sweepAt === null) sweepAt = t + rng() * p.sweepDelay;
        if (open && touch === null) touch = { t, how: "trade" };
        if (open && fifo === null) {
          ahead -= size;
          if (ahead < 0) { fifo = { t, how: "queue" }; ahead = 0; }
        }
        note(t, "trade", `a seller hits the bid for ${fmt(size)} shares`);
      }
    }
    const midAt = (t) => { let m = mids[0][1]; for (const [s, v] of mids) { if (s <= t + 1e-9) m = v; else break; } return m; };
    const outcome = (f) => (f === null ? null : { t: f.t, how: f.how, markout: midAt(f.t + p.horizon) });
    return { touch: outcome(touch), fifo: outcome(fifo), mids, log, params: p };
  }

  // Many independent orders: fill rates and mean markouts for each model.
  function runMany(params, n = 2000, seed = 7) {
    const rng = mulberry32(seed);
    const acc = { touch: { fills: 0, sum: 0, markouts: [] }, fifo: { fills: 0, sum: 0, markouts: [] } };
    for (let i = 0; i < n; i++) {
      const r = runOne(params, rng);
      for (const model of ["touch", "fifo"]) {
        if (r[model]) { acc[model].fills += 1; acc[model].sum += r[model].markout; acc[model].markouts.push(r[model].markout); }
      }
    }
    const summary = (a) => ({ pFill: a.fills / n, meanMarkout: a.fills ? a.sum / a.fills : NaN, markouts: a.markouts });
    return { n, touch: summary(acc.touch), fifo: summary(acc.fifo) };
  }

  root.MarkoutQueue = { DEFAULTS, mulberry32, runOne, runMany };
})(typeof window !== "undefined" ? window : globalThis);
