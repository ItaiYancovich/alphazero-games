// Every game: a new game against its strongest bot, a few moves, a take-back.
import { start } from "./harness.mjs";

const t0 = performance.now();
const h = await start();
console.log(`boot ${((performance.now() - t0) / 1000).toFixed(1)}s`);
const bots = h.api("/api/bots");
for (const game of bots.games) {
  const t = performance.now();
  const opp = game.default_opponent;
  let st = h.api("/api/new", { game: game.key, black: "human", white: opp, seed: 1 });
  if (st.error || st.problem) console.log("  problem:", st.error || st.problem);
  // Human plays the first legal move each turn; the bot answers.
  let plies = 0;
  for (let turn = 0; turn < 3 && !st.winner && !st.drawn; turn++) {
    const legal = firstLegal(game.key, st);
    if (legal == null) break;
    st = h.api("/api/move", legal);
    if (st.problem) { console.log("  move refused:", st.problem, JSON.stringify(legal)); break; }
    h.drain();
    st = h.api("/api/state");
    plies = st.moves.length;
  }
  const info = JSON.stringify(st.info);
  console.log(`${game.key.padEnd(9)} vs ${opp.padEnd(10)} plies=${plies} `
    + `err=${st.error} info=${info} ${((performance.now() - t) / 1000).toFixed(1)}s`);
}
// Live analysis on Connect Four.
let st = h.api("/api/new", { game: "c4", black: "human", white: "human" });
h.api("/api/analysis", { enabled: true, max_sims: 500 });
h.drain(200);
st = h.api("/api/state");
console.log("analysis:", st.analysis.sims, st.analysis.value_black?.toFixed(3),
            st.analysis.top?.slice(0, 3).map(m => m.label).join(","));
h.api("/api/analysis", { enabled: false, max_sims: 500 });
console.log("net calls", h.stats());
await h.close();
process.exit(0);

function firstLegal(key, st) {
  if (key === "bg") {
    const o = (st.options || [])[0];
    return o ? { hops: o.hops } : null;
  }
  if (key === "splendor") {
    const a = (st.takes || [])[0] || null;
    if (a) return { action: a.action };
    if (st.pass_action != null) return { action: st.pass_action };
    return null;
  }
  if (key === "rps2") {
    const from = Object.keys(st.legal_from || {})[0];
    if (from == null) return null;
    const to = st.legal_from[from][0];
    return { from: +from, to: typeof to === "object" ? to.to ?? to[0] : to };
  }
  const legal = st.legal || null;
  if (legal && legal.length) return { cell: legal[0] };
  const cells = st.cells || [];
  if (key === "c4") {
    for (let c = 0; c < st.cols; c++) {
      for (let r = st.rows - 1; r >= 0; r--) if (cells[r * st.cols + c] === 0) return { cell: r * st.cols + c };
    }
  }
  const i = cells.findIndex(v => v === 0);
  return i >= 0 ? { cell: i } : null;
}
