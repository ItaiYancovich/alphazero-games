# Source of the web build

The repo root is the published site (GitHub Pages serves it as is). This
folder holds what that site is **built from**, laid out like the main
project, so changes to the engine can be made here too. The site itself only
has build outputs.

| What | Where |
|---|---|
| Game server and all game logic (runs in Pyodide) | `game_gui.py`, `alphazero_*/` |
| PUCT search (Hex, Connect Four, UTTT "az" bots, analysis) | `alphazero_core/mcts.py`, `alphazero_core/agents.py`, `alphazero_core/evaluator.py` |
| UTTT v3 match bot | `alphazero_uttt/rs_agent.py`, Rust core in `uttt_rs/` |
| Web layer: loaders, v3 agent without threads, session stepping | `webapp/py/webgui.py` (see `WebPonderingAgent.run`, `ponder_tick`) |
| `uttt_rs` for the browser (WASM wrapper + Python shim) | `webapp/uttt_wasm/`, `webapp/py/alphazero_uttt/uttt_rs.py` |
| Engine worker, network workers, pool, tuning | `webapp/static/engine-worker.js`, `ort-worker.js`, `pool.js` |
| Page: bridge, online play, extra UI | `webapp/static/bridge.js`, `head.html`, `body.html`, `relay.js`; base page `web/index.html` |
| Speed test | `webapp/static/bench.html` |
| Build | `webapp/build.py` |

## Rules

* **Edit here, then build. Don't edit the build outputs at the repo root by
  hand** (`engine.zip`, `index.html`, `*.js`, `bench.html`, `engine.json`,
  `uttt_wasm.wasm`). The build regenerates them from this folder, so a change
  made only at the root is lost on the next build.
* Commit the source change and the rebuilt outputs together.
* `webapp/py/**` goes into `engine.zip` at the zip root, so it shadows
  project modules of the same name. `webapp/py/torch` is a numpy shim, and
  `webapp/py/alphazero_uttt/uttt_rs.py` stands in for the native extension.
* The owner syncs this folder with the main project using
  `webapp/sync_source.py` (`push` sends it here, `pull --apply` brings changes
  back). Keep files in their places so that works.

## Build

```bash
python source/webapp/build.py
```

When it runs from `source/`, the build writes into the repo root. It builds:

* `engine.zip` from the Python here.
* `index.html` and the static files.
* `uttt_wasm.wasm`. This needs `cargo` and `rustup target add wasm32-wasip1`;
  without them the step is skipped and the wasm already in the repo stays.
* The vendored runtimes (`pyodide/`, `ort/`, `nostr.bundle.js`) are already
  in the repo. `npm install` in `source/webapp/dev` is only needed to upgrade
  them.

**Networks:** `models/*.onnx` and `models/models.json` come from
`webapp/export_onnx.py`. That script needs the PyTorch checkpoints, which are
not in this repo, so don't run `build.py --models`. To change a network here
(an 8-bit version, say), work from the `.onnx` files in `models/` and add or
adjust its entry in `models/models.json`. If a new checkpoint name is used,
also update `PLACEMENTS` in `build.py` and rebuild so `engine.json` matches.

## Test

```bash
cd source/webapp/dev && npm install && node smoke.mjs
```

This plays every game against its bot in Node, using Pyodide and
onnxruntime-web against the built site. `parity.mjs` checks the WASM
`uttt_rs` against saved positions. The `v3*.mjs` scripts exercise the v3 bot.
Locally, `python source/webapp/serve.py` serves the site with the
cross-origin isolation headers on port 8765.

On a real device, `bench.html` times the bots.
