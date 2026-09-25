import { start } from "./harness.mjs";
import { readFileSync } from "node:fs";
const cases = JSON.parse(readFileSync("parity_positions.json", "utf8"));
const h = await start();
h.py.globals.set("cases_json", JSON.stringify(cases));
const res = h.py.runPython(`
import json
from alphazero_uttt import uttt_rs
cases = json.loads(cases_json)
report = []
for c in cases:
    s = uttt_rs.Searcher(1)
    s.set_root_moves(0, c["moves"])
    solve = [list(p) for p in s.root_solve(0, 2**62, False, 20.0)]
    raw, ids = s.collect(8, 8)
    report.append(dict(
        empties=s.root_open_empties(0) == c["empties"],
        solve=solve == [list(p) for p in c["solve"]],
        canonical=list(uttt_rs.canonical(c["moves"])) == c["canonical"],
        legal=list(uttt_rs.legal_moves(c["moves"])[0]) == c["legal"],
        types=[type(uttt_rs.legal_moves(c["moves"])[0]).__name__, type(s.pv(0, 4)).__name__] == c["types"],
        exact=uttt_rs.solve(c["moves"], 2**40) == c["exact"],
        collected=list(raw[:81]) == c["collected"]))
json.dumps(report)
`);
for (const [i, r] of JSON.parse(res).entries()) console.log(i, cases[i].empties, "empties", JSON.stringify(r));
await h.close(); process.exit(0);
