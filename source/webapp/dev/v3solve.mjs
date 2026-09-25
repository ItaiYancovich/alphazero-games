import { start } from "./harness.mjs";
import { readFileSync } from "node:fs";
const cases = JSON.parse(readFileSync("parity_positions.json", "utf8"));
const h = await start();
h.py.globals.set("cases_json", JSON.stringify(cases.slice(0, 3)));
const res = h.py.runPython(`
import json, time
import webgui
g = webgui.g
from alphazero_uttt.uttt_game import UltimateBoard
out = []
for c in json.loads(cases_json):
    board = g.GAMES["uttt"].new_board("9", 0)
    for m in c["moves"]:
        board.play(m)
    agent = g.GAMES["uttt"].build_agent("v3:3", 1, board, "", 400, False)
    t0 = time.time()
    mv = agent.select_move(board.copy())
    info = agent.last_info
    out.append(dict(kind=type(agent).__name__, move=mv, legal=board.is_legal(mv),
                    exact=info.get("exact"), exact_moves=info.get("exact_moves"),
                    desktop_solve=sorted(v for _, v in c["solve"])[-1],
                    seconds=round(time.time() - t0, 2), sims=info.get("sims")))
json.dumps(out)
`);
for (const r of JSON.parse(res)) console.log(JSON.stringify(r));
await h.close(); process.exit(0);
