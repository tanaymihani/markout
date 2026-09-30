const fs = require("fs");
global.window = {}; global.document = { addEventListener() {} };
eval(fs.readFileSync(process.argv[2], "utf8"));
const G = window.markoutGame;
const cases = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
let ok = true;
for (const c of cases) {
  const d = { cards: c.cards, suits: c.cards.map(() => "♠"), bots: c.bots };
  const g = new G.Game(d);
  for (const [b, a] of c.quotes) g.play(b, a);
  const acts = g.events.map((e) => e.action);
  const same = JSON.stringify(acts) === JSON.stringify(c.actions);
  const m = G.posteriorMean(g.revealed, g.events, 300000, G.mulberry32(3));
  const diff = Math.abs(m - c.post);
  console.log(`actions match: ${same}  js ${m.toFixed(3)} vs py ${c.post.toFixed(3)}  |diff| ${diff.toFixed(3)}`);
  if (!same || diff > 0.1) ok = false;
}
console.log(ok ? "JS port agrees with the Python engine" : "MISMATCH");
process.exit(ok ? 0 : 1);
