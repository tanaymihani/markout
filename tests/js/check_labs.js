// Runs the browser lab code in Node and prints the numbers the Python tests compare against.
// Usage: node check_labs.js <docs dir> <input.json>
const fs = require("fs");
const path = require("path");
const [docs, input] = process.argv.slice(2);
for (const f of ["queue-sim.js", "lab-math.js"]) eval(fs.readFileSync(path.join(docs, "js", f), "utf8"));
const Q = globalThis.MarkoutQueue, M = globalThis.MarkoutMath;
const data = JSON.parse(fs.readFileSync(input, "utf8"));

const run = (p) => { const r = Q.runMany(p, 3000, 11); return { touch: [r.touch.pFill, r.touch.meanMarkout], fifo: [r.fifo.pFill, r.fifo.meanMarkout] }; };
const out = {
  defaults: run({}),
  again: run({}),
  short_queue: run({ queueAhead: 1000 }),
  long_queue: run({ queueAhead: 30000 }),
  no_informed: run({ informedShare: 0 }),
  kelly: M.kellyExact(data.pnl),
  wealth_last: M.wealthPath(data.pnl, data.v)[data.pnl.length - 1],
  cost: [1, 2].map((m) => M.costCurve(data.gross, data.cost, m).sharpe),
};
console.log(JSON.stringify(out));
