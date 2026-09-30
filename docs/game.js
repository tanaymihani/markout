// Market-making card game: a browser port of src/markout/games/cardgame.py.
//
// Five cards are dealt face down from a standard deck (A = 1 ... K = 13). You quote a
// two-sided market on their sum S. Each round three bots trade against your quote: an
// informed bot (probability 0.4) knows one face-down card and trades only when your quote
// is wrong relative to it; a noise bot buys or sells at random. Then one card is turned
// face up. At the end every trade settles at S.
//
// The benchmark is E[S | revealed cards, every trade seen so far], computed by importance
// sampling over the face-down cards (the same model as the Python engine).

(function () {
  "use strict";

  const CFG = { nCards: 5, botsPerRound: 3, pInformed: 0.4, maxWidth: 4 };
  const DECK = [];
  for (let r = 1; r <= 13; r++) for (let s = 0; s < 4; s++) DECK.push(r);
  const RANK = { 1: "A", 11: "J", 12: "Q", 13: "K" };
  const SUITS = ["♠", "♥", "♦", "♣"];
  const name = (v) => RANK[v] || String(v);

  function mulberry32(seed) {
    return function () {
      seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function deal(rng) {
    const idx = DECK.map((_, i) => i);
    for (let i = idx.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [idx[i], idx[j]] = [idx[j], idx[i]];
    }
    const picked = idx.slice(0, CFG.nCards);
    const cards = picked.map((i) => DECK[i]);
    const suits = picked.map((i) => SUITS[i % 4]);
    const bots = [];
    for (let r = 0; r < CFG.nCards; r++) {
      const row = [];
      for (let b = 0; b < CFG.botsPerRound; b++) {
        row.push({ informed: rng() < CFG.pInformed, slotU: rng(), side: rng() < 0.5 ? 1 : -1 });
      }
      bots.push(row);
    }
    return { cards, suits, bots };
  }

  const sum = (a) => a.reduce((x, y) => x + y, 0);

  function restMean(known) {
    const rest = DECK.slice();
    for (const c of known) rest.splice(rest.indexOf(c), 1);
    return sum(rest) / rest.length;
  }

  function botValue(revealed, own, nHidden) {
    return sum(revealed) + own + (nHidden - 1) * restMean([...revealed, own]);
  }

  function act(value, bid, ask) {
    if (ask < value) return 1; // the bot buys at your ask
    if (bid > value) return -1; // the bot sells at your bid
    return 0;
  }

  class Game {
    constructor(d) { this.d = d; this.round = 0; this.events = []; this.quotes = []; }
    get revealed() { return this.d.cards.slice(0, this.round); }
    get done() { return this.round >= CFG.nCards; }
    play(bid, ask) {
      const r = this.round;
      const hidden = this.d.cards.slice(r);
      const out = this.d.bots[r].map((bot) => {
        let a;
        if (bot.informed) {
          const own = hidden[Math.min(Math.floor(bot.slotU * hidden.length), hidden.length - 1)];
          a = act(botValue(this.revealed, own, hidden.length), bid, ask);
        } else a = bot.side;
        return { round: r, bid, ask, action: a };
      });
      this.events.push(...out);
      this.quotes.push([bid, ask]);
      this.round += 1;
      return out;
    }
    pnl() {
      const s = sum(this.d.cards);
      let p = 0;
      for (const e of this.events) {
        if (e.action === 1) p += e.ask - s;
        else if (e.action === -1) p += s - e.bid;
      }
      return p;
    }
  }

  // P(event | full deal) for one sampled deal, as in the Python _event_likelihood
  function likelihood(e, cards) {
    const r = e.round, nHidden = CFG.nCards - r;
    const revSum = sum(cards.slice(0, r));
    const deckSum = sum(DECK), nDeck = DECK.length;
    let inf = 0;
    for (let j = r; j < CFG.nCards; j++) {
      const own = cards[j];
      const value = revSum + own + (nHidden - 1) * (deckSum - revSum - own) / (nDeck - r - 1);
      if (act(value, e.bid, e.ask) === e.action) inf += 1;
    }
    inf /= nHidden;
    const noise = e.action !== 0 ? 0.5 : 0;
    return CFG.pInformed * inf + (1 - CFG.pInformed) * noise;
  }

  function posteriorMean(revealed, events, n = 20000, rng = mulberry32(7)) {
    const nHidden = CFG.nCards - revealed.length;
    if (nHidden === 0) return sum(revealed);
    const rest = DECK.slice();
    for (const c of revealed) rest.splice(rest.indexOf(c), 1);
    let wsum = 0, wsx = 0;
    const logs = new Float64Array(n), sums = new Float64Array(n);
    let maxLog = -Infinity;
    const pool = rest.slice();
    for (let k = 0; k < n; k++) {
      for (let i = 0; i < nHidden; i++) { // partial Fisher-Yates: draw the face-down cards
        const j = i + Math.floor(rng() * (pool.length - i));
        [pool[i], pool[j]] = [pool[j], pool[i]];
      }
      const cards = revealed.concat(pool.slice(0, nHidden));
      let lw = 0;
      for (const e of events) {
        const l = likelihood(e, cards);
        lw += l > 0 ? Math.log(l) : -Infinity;
      }
      logs[k] = lw; sums[k] = sum(cards);
      if (lw > maxLog) maxLog = lw;
    }
    for (let k = 0; k < n; k++) {
      if (!isFinite(logs[k])) continue;
      const w = Math.exp(logs[k] - maxLog);
      wsum += w; wsx += w * sums[k];
    }
    return wsum > 0 ? wsx / wsum : sum(revealed) + nHidden * restMean(revealed);
  }

  const priorMean = (revealed) => sum(revealed) + (CFG.nCards - revealed.length) * restMean(revealed);

  function replay(d, quoter) { // a scripted quoter on the same deal (the bots are fixed in advance)
    const g = new Game(d);
    while (!g.done) {
      const m = quoter(g);
      g.play(m - CFG.maxWidth / 2, m + CFG.maxWidth / 2);
    }
    return g.pnl();
  }

  // ------------------------------------------------------------------ UI
  const $ = (s) => document.querySelector(s);
  let game, seed;

  function cardHTML(v, suit, up) {
    if (!up) return '<span class="card down" aria-label="face-down card"></span>';
    const red = suit === "♥" || suit === "♦";
    return `<span class="card${red ? " red" : ""}">${name(v)}<small>${suit}</small></span>`;
  }

  function render(msg) {
    const d = game.d;
    $("#cards").innerHTML = d.cards.map((v, i) => cardHTML(v, d.suits[i], i < game.round)).join("");
    $("#round").textContent = game.done ? "Game over" : `Round ${game.round + 1} of ${CFG.nCards}`;
    $("#known").textContent = `Sum of face-up cards: ${sum(game.revealed)} · face down: ${CFG.nCards - game.round}`;
    $("#quote").disabled = game.done;
    if (msg !== undefined) $("#log").insertAdjacentHTML("afterbegin", msg);
  }

  function describe(ev) {
    return ev.map((e, i) => {
      const what = e.action === 1 ? `bought at your ask ${e.ask}` : e.action === -1 ? `sold at your bid ${e.bid}` : "passed";
      return `bot ${i + 1} ${what}`;
    }).join(" · ");
  }

  function newGame() {
    seed = Math.floor(Math.random() * 1e9);
    game = new Game(deal(mulberry32(seed)));
    $("#log").innerHTML = "";
    $("#result").innerHTML = "";
    const m = priorMean([]);
    $("#bid").value = (m - 2).toFixed(1);
    $("#ask").value = (m + 2).toFixed(1);
    render();
  }

  function quote() {
    const bid = Number($("#bid").value), ask = Number($("#ask").value);
    const err = !isFinite(bid) || !isFinite(ask) ? "Enter a bid and an ask."
      : bid >= ask ? "The bid must be below the ask."
      : ask - bid > CFG.maxWidth + 1e-9 ? `The market can be at most ${CFG.maxWidth} wide.` : "";
    $("#err").textContent = err;
    if (err) return;
    const r = game.round;
    const ev = game.play(bid, ask);
    const card = game.d.cards[r], suit = game.d.suits[r];
    const bayes = posteriorMean(game.revealed, game.events);
    render(`<div class="entry"><b>Round ${r + 1}:</b> you quoted ${bid} / ${ask}. ${describe(ev)}. `
      + `Turned up: ${name(card)}${suit}. <span class="hint">Best estimate of S now, given every trade: `
      + `${bayes.toFixed(1)}</span></div>`);
    if (game.done) finish();
  }

  function finish() {
    const s = sum(game.d.cards);
    const you = game.pnl();
    const bayes = replay(game.d, (g) => posteriorMean(g.revealed, g.events));
    const prior = replay(game.d, (g) => priorMean(g.revealed));
    $("#result").innerHTML = `<div class="final">S = <b>${s}</b>. Your PnL: <b>${you >= 0 ? "+" : ""}${you.toFixed(1)}</b>.`
      + ` On the same cards and bots, a quoter centred on the Bayes estimate makes ${bayes >= 0 ? "+" : ""}${bayes.toFixed(1)},`
      + ` and one that ignores the trades makes ${prior >= 0 ? "+" : ""}${prior.toFixed(1)}.</div>`;
  }

  document.addEventListener("DOMContentLoaded", () => {
    $("#quote").addEventListener("click", quote);
    $("#new").addEventListener("click", newGame);
    for (const el of [$("#bid"), $("#ask")]) el.addEventListener("keydown", (e) => { if (e.key === "Enter") quote(); });
    newGame();
  });

  // exposed for tests in the browser console
  window.markoutGame = { deal, Game, posteriorMean, priorMean, replay, mulberry32, CFG };
})();
