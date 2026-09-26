# Source of the web build

The repo root is the published site (GitHub Pages serves it as is). This
folder holds what that site is **built from**, laid out like the main
project, so changes to the engine can be made here too. The site itself only
has build outputs.

| What | Where |
|---|---|
| Game server and all game logic (runs in Pyodide) | `game_gui.py`, `alphazero_*/` |
| PUCT search (Hex, Connect Four, UTTT "az" bots, analysis) | `alphazero_core/mcts.py`, `alphazero_core/agents.py`, `alphazero_core/evaluator.py` |
| Search beside the network (web only, off: `BACKGROUND` in `webgui.py`) | `run_search` in `alphazero_core/mcts.py`; `evaluate_start` in the evaluators; `WebNet.start` in `webgui.py` |
| UTTT analysis and review on the v3 network (web only) | `V3Judge`, `_judge_uttt_with_v3` in `webapp/py/webgui.py` |
| Evaluation cache | `EvalCache` in `webapp/py/webgui.py` |
| GPU (WebGPU) worker | `load` in `webapp/static/ort-worker.js`, `gpuSlot` / `"gpu:f"` in `engine-worker.js`, `pool.js` |
| 8-bit networks | `webapp/quantize.py` -> `models/*_int8.onnx` |
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

**GPU:** where the browser has WebGPU, `pool.js` adds one more network
worker that loads onnxruntime-web's WebGPU build (`ort/ort.webgpu.min.mjs` and
the `.jsep` files) and runs the float networks on the graphics card. It is one
more choice for the tuning (`"gpu:f"`), timed like the rest; a choice more
than 3x slower than the best seen is dropped after one timing, and where the
GPU loses that badly it is not tried again for that network at smaller
batches. A software GPU (the browser's CPU fallback) is not used at all;
`bench.html?gpu=force` uses it anyway, for testing. The GPU worker rounds
batches up to 8, 16, 32, 64 or a multiple of 32, so it compiles shaders for
only a few shapes. When the tuning has put the v3 network on the GPU, the v3
bot searches with the desktop's batches of 128 instead of 48.

**8-bit networks:** Hex, Connect Four and UTTT (the "az" bots) have 8-bit
versions: `models/<name>_int8.onnx`, with the float file kept as `float_file`
in `models.json` (bench.html's "before" runs it). Which one runs is decided
per device: the first batch a network is asked for is timed both ways on the
first worker, and the faster is kept with the rest of the tuning
(`precisionFor` in `engine-worker.js`). On a desktop the 8-bit one wins by
about 1.6x; on an 8-core Android phone it did not. Only the residual tower is
8-bit; the stem and heads stay float. `webapp/quantize.py` makes them
(`pip install onnxruntime onnx`), calibrating on positions from bot games and
reporting how closely the result follows the float network. The shipped
files were made the same way (from a separate calibration run) and each
scored 50-53% in 24-40 game matches against the float network at 400
simulations. Intransitive's value head lost too much, and the v3 and
Splendor networks are not convolutional towers, so those stay float. A
re-export from the main project (`export_onnx.py`) writes a fresh
`models.json`; run `quantize.py` again after it.

## Test

```bash
cd source/webapp/dev && npm install && node smoke.mjs
```

This plays every game against its bot in Node, using Pyodide and
onnxruntime-web against the built site. `parity.mjs` checks the WASM
`uttt_rs` against saved positions. The `v3*.mjs` scripts exercise the v3 bot.
Locally, `python source/webapp/serve.py` serves the site with the
cross-origin isolation headers on port 8765.

On a real device, `bench.html` times the bots: each plays the rule-based bot,
before (the old site: one network worker, float networks, no cache, one batch
at a time) and after, alternating after-before-before-after.

Two things in `build.py` to know: text outputs are written with CRLF on every
system, so builds match wherever they run; and when `node_modules` exists in
`webapp/dev` (after `npm install` for the tests), the vendored runtimes are
copied from it -- if that is not an upgrade you meant, `git checkout pyodide
ort nostr.bundle.js` afterwards.
