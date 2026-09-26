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
| GPU (WebGPU) worker | `load`, `infer` in `webapp/static/ort-worker.js`; `gpuSlot`, `planOf` in `engine-worker.js`; `pool.js` |
| GPU versions of the v3 network | `webapp/gpu_models.py` -> `models/uttt_v3_gpu.onnx`, `uttt_v3_gpu16.onnx` |
| v3 exact solver on its own worker | `webapp/static/solver-worker.js`, `uttt-core.js`; `RemoteSolver` in `webgui.py` |
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
batches up to 16, 32, 64, 128 or a multiple of 64, so it sets up sessions for
only a few shapes. When the tuning has put the v3 network on the GPU, the v3
bot searches with the desktop's batches of 128 instead of 48.

On the GPU each batch size gets its own session with the batch fixed
(`freeDimensionOverrides`) and, where it proves correct, graph capture (the
first run is recorded, the rest replay it, sparing the per-step overhead of a
network of many small steps), its input kept in one GPU buffer.
onnxruntime-web's replay ignored new input for Hex and Connect Four (it
answered the recorded positions again), so every captured session is replayed
once on fresh random positions and compared with an uncaptured one before it
is used; where they disagree, that size runs uncaptured. The tuning's GPU
choices are `gpu:f` and `mix:f<cpu>` (e.g. `mix:f8`): the GPU takes a share
of the batch in proportion to its measured speed and the whole CPU pool the
rest, from the shared counter. Half precision (`gpu:h`, `mix:h?`: only where
the GPU has `shader-f16` and the network a `gpu16_file`) is timed by
bench.html but not used by the bots: on an Iris Xe its values strayed up to
0.04 from float.

The v3 network has GPU versions (`webapp/gpu_models.py`): onnxruntime-web has
no WebGPU Softplus, so the original ran four steps of every call on the CPU
and could not be captured; `uttt_v3_gpu.onnx` computes Softplus from ops the
GPU has (outputs within 4e-6), and `uttt_v3_gpu16.onnx` is that in half
precision (value within 0.006, like the desktop's). The CPU keeps the
original, so its moves stay the desktop's.

**The v3 bot's exact solver** runs on a worker of its own (`solver-worker.js`,
a second copy of the native core), as it runs on a thread of its own on the
desktop: beside the bot's search, and on the position you must answer while
you think. `RemoteSolver` in `webgui.py` stands in for the agent's solver; the
long solves go to the worker and the search only looks for the answer between
batches, instead of giving the solver 0.08 s slices of its own time. With
`workers: 0` (bench.html's "before") the solver stays in the engine, as before.

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
