// The UTTT v3 bot in the web engine: it builds, answers moves, ponders.
import { start } from "./harness.mjs";

const h = await start();
const bots = h.api("/api/bots");
const uttt = bots.games.find(g => g.key === "uttt");
console.log("uttt bots:", uttt.bots.map(b => b.id).join(" "));
let st = h.api("/api/new", { game: "uttt", black: "human", white: "v3:3", seed: 1 });
console.log("new:", st.error || st.problem || "ok", st.seats.map(s => s.name));
for (let turn = 0; turn < 4 && !st.winner && !st.drawn; turn++) {
  const cell = st.legal[Math.floor(st.legal.length / 2)];
  st = h.api("/api/move", { cell });
  if (st.problem) { console.log("refused", st.problem); break; }
  const t0 = performance.now();
  h.drain(1);                                     // the bot's move
  st = h.api("/api/state");
  const ms = performance.now() - t0;
  // Ponder a few ticks, the way the worker does while the human thinks.
  let pondered = 0;
  for (let i = 0; i < 5 && h.web.work() === "ponder"; i++) { h.web.step("ponder"); pondered++; }
  console.log(`move ${turn + 1}: bot ${st.moves.at(-1)?.label} in ${(ms / 1000).toFixed(1)}s`
              + ` info=${JSON.stringify(st.info)} err=${st.error} ponder ticks=${pondered}`);
}
console.log("net", h.stats());
await h.close();
process.exit(0);
