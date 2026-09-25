// The network worker: runs the exported ONNX models for the engine worker.
//
// The engine (Python, in engine-worker.js) needs each network answer
// *synchronously* -- the project's searches call `evaluate()` and expect the
// numbers back -- but onnxruntime only answers asynchronously.  So the two
// meet in shared memory: the engine writes a batch of input planes into the
// shared buffer, posts a job here and blocks on `Atomics.wait`; this worker
// runs the model, writes the outputs into the same buffer and wakes it.
//
// Besides the first of these there is a pool of them, one per core the
// device offers (see pool.js).  A job may go to several at once, and then
// they share it out as they go: each takes the next few positions of the
// batch from a shared counter until none are left.  A worker on a fast core
// comes back for more sooner, so a phone's slow cores need not hold up the
// whole batch.
//
// Shared layout (offsets from pool.js):
//   job block, Int32: [0] job number, [1] next position to take
//   one control block per worker, Int32:
//     [0] state: 0 idle, 1 working, 2 done, 3 failed, 4 cannot be cut by row
//     [1] number of outputs, [2 + 5i] rank of output i, then up to four dims
//     [ROWS] positions this worker answered
//   the input, Float32: the batch, position after position
//   the output, Float32: output j of position r at
//     n * (sizes of outputs before j) + r * (size of output j)
//     -- for a batch in one piece, simply the outputs one after another.
import * as ort from "./ort/ort.wasm.min.mjs";

const CTL_INTS = 64, ROWS = CTL_INTS - 1;
let job = null, ctl = null, input = null, output = null, models = null;
const sessions = {};

ort.env.wasm.wasmPaths = new URL("./ort/", import.meta.url).href;
// Threads only exist where the page is cross-origin isolated.  The first
// worker splits each network call over a few threads of its own, as the lone
// worker always did: it is exactly the old setup.  The pool beside it runs on
// one thread a worker (pool.js says so).
ort.env.wasm.numThreads = self.crossOriginIsolated
  ? Math.max(1, Math.min(4, (navigator.hardwareConcurrency || 2) - 1)) : 1;

// A network with an 8-bit version has two files: `file` (8-bit) and
// `float_file`.  Which is faster depends on the device, and the engine decides
// (see `precisionFor` in engine-worker.js); `variant` names the one to run.
function fileOf(name, variant) {
  const m = models[name];
  return variant === "float" && m.float_file ? m.float_file : m.file;
}

function session(name, variant) {
  const key = `${name}:${fileOf(name, variant)}`;
  if (!sessions[key]) {
    sessions[key] = (async () => {
      const url = new URL("./" + fileOf(name, variant), import.meta.url);
      const bytes = new Uint8Array(await (await fetch(url)).arrayBuffer());
      return ort.InferenceSession.create(bytes, {
        executionProviders: ["wasm"], graphOptimizationLevel: "all",
      });
    })();
  }
  return sessions[key];
}

async function run({ id, model, variant, dims, n, chunk }) {
  const row = dims.slice(1).reduce((a, b) => a * b, 1);
  let answered = 0;
  try {
    const s = await session(model, variant);
    const names = models[model].outputs;
    for (;;) {
      if (Atomics.load(job, 0) !== id) break;           // given up on by the engine
      const from = Atomics.add(job, 1, chunk);
      if (from >= n) break;
      const rows = Math.min(chunk, n - from);
      const d = dims.slice();
      d[0] = rows;
      const out = await s.run({ x: new ort.Tensor("float32", input.slice(from * row, (from + rows) * row), d) });
      // Where each output goes: see the layout above.
      const whole = rows === n;
      if (!whole && names.some(name => out[name].dims[0] !== rows)) {
        Atomics.store(ctl, 0, 4);
        Atomics.notify(ctl, 0);
        return;
      }
      let base = 0;
      ctl[1] = names.length;
      names.forEach((name, i) => {
        const t = out[name];
        ctl[2 + i * 5] = t.dims.length;
        t.dims.forEach((v, k) => { ctl[3 + i * 5 + k] = v; });
        const size = whole ? t.data.length : n * (t.data.length / rows);
        if (base + size > output.length) throw new Error("output too large for the shared buffer");
        output.set(t.data, whole ? base : base + from * (t.data.length / rows));
        base += size;
      });
      answered += rows;
    }
    ctl[ROWS] = answered;
    Atomics.store(ctl, 0, 2);
  } catch (err) {
    console.error("inference failed", err);
    Atomics.store(ctl, 0, 3);
  }
  Atomics.notify(ctl, 0);
}

self.onmessage = async ({ data: msg }) => {
  if (msg.type === "init") {
    const { sab, layout } = msg;
    job = new Int32Array(sab, layout.job, 16);
    ctl = new Int32Array(sab, layout.ctl, CTL_INTS);
    input = new Float32Array(sab, layout.input, layout.inputFloats);
    output = new Float32Array(sab, layout.output, layout.outputFloats);
    models = msg.models;
    if (msg.threads) ort.env.wasm.numThreads = msg.threads;
    // Requests arrive on a port straight from the engine worker, which is
    // blocked while it waits and cannot go through the page.
    msg.port.onmessage = ({ data: req }) => run(req);
    // The first worker loads every model up front, off the critical path of
    // the first move.  The rest of the pool loads a model when it is first
    // asked for it: most visits play one or two games, and every model in
    // every worker would cost a phone well over a hundred megabytes a worker.
    const all = Object.keys(models).flatMap(name =>
      models[name].float_file ? [session(name, "int8"), session(name, "float")] : [session(name)]);
    Promise.all(msg.lazy ? [] : all)
      .then(() => self.postMessage({ type: "models-ready" }))
      .catch(err => self.postMessage({ type: "error", message: String(err) }));
    self.postMessage({ type: "ready" });
  }
};
