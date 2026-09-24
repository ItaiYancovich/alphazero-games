// The network worker: runs the exported ONNX models for the engine worker.
//
// The engine (Python, in engine-worker.js) needs each network answer
// *synchronously* -- the project's searches call `evaluate()` and expect the
// numbers back -- but onnxruntime only answers asynchronously.  So the two
// meet in shared memory: the engine writes the input planes into the shared
// buffer, posts a request here and blocks on `Atomics.wait`; this worker runs
// the model, writes the outputs into the same buffer and wakes it.
//
// Shared layout (see engine-worker.js): an Int32 control block --
//   [0] state: 0 idle, 1 request, 2 done, 3 failed
//   [1] number of outputs
//   [2 + 5i] rank of output i, then up to four dims
// -- followed by a Float32 data area holding the input, then the outputs.
import * as ort from "./ort/ort.wasm.min.mjs";

const CTL_INTS = 64;
let ctl = null, data = null, models = null;
const sessions = {};

ort.env.wasm.wasmPaths = new URL("./ort/", import.meta.url).href;
// Threads only exist where the page is cross-origin isolated; four is plenty
// for networks this size, and leaves the machine responsive.
ort.env.wasm.numThreads = self.crossOriginIsolated
  ? Math.max(1, Math.min(4, (navigator.hardwareConcurrency || 2) - 1)) : 1;

async function session(name) {
  if (!sessions[name]) {
    const url = new URL("./" + models[name].file, import.meta.url);
    const bytes = new Uint8Array(await (await fetch(url)).arrayBuffer());
    sessions[name] = await ort.InferenceSession.create(bytes, {
      executionProviders: ["wasm"], graphOptimizationLevel: "all",
    });
  }
  return sessions[name];
}

async function run({ model, dims, n }) {
  try {
    const s = await session(model);
    const input = new ort.Tensor("float32", data.slice(0, n), dims);
    const out = await s.run({ x: input });
    const names = models[model].outputs;
    let at = 0;
    ctl[1] = names.length;
    names.forEach((name, i) => {
      const t = out[name];
      ctl[2 + i * 5] = t.dims.length;
      t.dims.forEach((d, k) => { ctl[3 + i * 5 + k] = d; });
      data.set(t.data, at);
      at += t.data.length;
    });
    Atomics.store(ctl, 0, 2);
  } catch (err) {
    console.error("inference failed", err);
    Atomics.store(ctl, 0, 3);
  }
  Atomics.notify(ctl, 0);
}

self.onmessage = async ({ data: msg }) => {
  if (msg.type === "init") {
    ctl = new Int32Array(msg.sab, 0, CTL_INTS);
    data = new Float32Array(msg.sab, CTL_INTS * 4);
    models = msg.models;
    // Requests arrive on a port straight from the engine worker, which is
    // blocked while it waits and cannot go through the page.
    msg.port.onmessage = ({ data: req }) => run(req);
    // Load every model up front, off the critical path of the first move.
    Promise.all(Object.keys(models).map(session))
      .then(() => self.postMessage({ type: "models-ready" }))
      .catch(err => self.postMessage({ type: "error", message: String(err) }));
    self.postMessage({ type: "ready" });
  }
};
