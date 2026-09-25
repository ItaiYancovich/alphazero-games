import { start } from "./harness.mjs";
const h = await start();
let st = h.api("/api/new", { game: "uttt", black: "human", white: "v3:3", seed: 1 });
for (let turn = 0; turn < 4 && !st.winner && !st.drawn; turn++) {
  st = h.api("/api/move", { cell: st.legal[Math.floor(st.legal.length / 2)] });
  h.drain(1);
  st = h.api("/api/state");
  console.log(`after ${st.moves.at(-1).label}:`, JSON.stringify(st.info));
  // "The human thinks": the bot ponders for a few steps.
  for (let i = 0; i < 6 && h.web.work() === "ponder"; i++) h.web.step("ponder");
}
await h.close(); process.exit(0);
