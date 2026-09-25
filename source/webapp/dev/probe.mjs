// Can the project's Python engine run inside Pyodide?  Mount the project,
// import the GUI module, and play a few moves with classical bots.
import { loadPyodide } from "pyodide";

const ROOT = "C:/Users/itai/Downloads/alphazero_hex_project";
const py = await loadPyodide();
await py.loadPackage(["numpy"]);
py.FS.mkdirTree("/proj");
py.FS.mount(py.FS.filesystems.NODEFS, { root: ROOT }, "/proj");
const t0 = Date.now();
const out = await py.runPythonAsync(`
import sys, time
sys.path.insert(0, "/proj")
t = time.time()
import game_gui as g
res = [f"import {time.time()-t:.1f}s"]
for key in ("hex", "c4", "uttt", "uxx", "rps2", "bg", "splendor"):
    a = g.GAMES[key]
    b = a.new_board(a.default_size, 1)
    ids = [x["id"] for x in a.catalog() if not x["torch"] and x["id"] != "human"]
    res.append(f"{key}: bots {ids}")
res
`);
console.log(out.toJs().join("\n"), `\ntotal ${(Date.now() - t0) / 1000}s`);
