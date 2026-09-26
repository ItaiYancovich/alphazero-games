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
//   job block, Int32: [0] job number, [1] next position to take, [2] GPU
//     status (see GPU below)
//   one control block per worker, Int32:
//     [0] state: 0 idle, 1 working, 2 done, 3 failed, 4 cannot be cut by row
//     [1] number of outputs, [2 + 5i] rank of output i, then up to four dims
//     [ROWS] positions this worker answered
//   the input, Float32: the batch, position after position
//   the output, Float32: output j of position r at
//     n * (sizes of outputs before j) + r * (size of output j)
//     -- for a batch in one piece, simply the outputs one after another.

const CTL_INTS = 64, ROWS = CTL_INTS - 1;
const GPU = 2;             // job block: 0 no GPU worker, 1 GPU ready, 2 no GPU after all
let job = null, ctl = null, input = null, output = null, models = null;
let ort = null, provider = "wasm";
let started = null;        // resolves once onnxruntime is loaded
const sessions = {};

// The CPU workers load onnxruntime's WebAssembly build.  One more worker, when
// the browser has WebGPU, loads its WebGPU build and runs the networks on the
// graphics card (pool.js makes it; the engine's tuning decides when it is used).
async function load(backend, threads, force) {
  if (backend === "webgpu") {
    const adapter = navigator.gpu && await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("no WebGPU adapter");
    // A software GPU (a browser's CPU fallback) is far slower than the CPU
    // workers: not worth even trying.  `force` is for testing without a GPU.
    const info = adapter.info || {};
    if ((adapter.isFallbackAdapter || info.isFallbackAdapter
         || /swiftshader|llvmpipe|software/i.test(`${info.vendor} ${info.architecture} ${info.description}`))
        && !force) {
      throw new Error("only a software GPU");
    }
    ort = await import("./ort/ort.webgpu.min.mjs");
    provider = "webgpu";
  } else {
    ort = await import("./ort/ort.wasm.min.mjs");
  }
  ort.env.wasm.wasmPaths = new URL("./ort/", import.meta.url).href;
  // Threads only exist where the page is cross-origin isolated.  The first
  // worker splits each network call over a few threads of its own, as the lone
  // worker always did: it is exactly the old setup.  The pool beside it runs on
  // one thread a worker, and so does the GPU worker's CPU side (pool.js).
  ort.env.wasm.numThreads = !self.crossOriginIsolated ? 1
    : threads || Math.max(1, Math.min(4, (navigator.hardwareConcurrency || 2) - 1));
}

// A network with an 8-bit version has two files: `file` (8-bit) and
// `float_file`.  Which is faster depends on the device, and the engine decides
// (the tuning in engine-worker.js); `variant` names the one to run.
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
        executionProviders: [provider], graphOptimizationLevel: "all",
      });
    })();
  }
  return sessions[key];
}

// On the GPU, every new batch size can mean compiling new shaders -- slow, and
// the bots' batches come in every size.  So the GPU runs batches rounded up
// to a few sizes (8, 16, 32, 64, then multiples of 32), padded with empty
// positions whose answers are dropped.  A network whose outputs do not lead
// with the batch cannot be cut back, and runs as it comes.
const unpadded = new Set();
const padTo = rows => (rows <= 64 ? Math.max(8, 2 ** Math.ceil(Math.log2(rows))) : Math.ceil(rows / 32) * 32);

async function infer(s, model, x, dims) {
  const rows = dims[0];
  const size = padTo(rows);
  if (provider !== "webgpu" || size === rows || unpadded.has(model)) {
    return s.run({ x: new ort.Tensor("float32", x, dims) });
  }
  const padded = new Float32Array(size * (x.length / rows));
  padded.set(x, 0);
  const out = await s.run({ x: new ort.Tensor("float32", padded, [size, ...dims.slice(1)]) });
  const cut = {};
  for (const [name, t] of Object.entries(out)) {
    if (t.dims[0] !== size) { unpadded.add(model); return infer(s, model, x, dims); }
    cut[name] = { dims: [rows, ...t.dims.slice(1)], data: t.data.subarray(0, (t.data.length / size) * rows) };
  }
  return cut;
}

async function run({ id, model, variant, dims, n, chunk }) {
  const row = dims.slice(1).reduce((a, b) => a * b, 1);
  let answered = 0;
  try {
    await started;
    const s = await session(model, variant);
    const names = models[model].outputs;
    for (;;) {
      if (Atomics.load(job, 0) !== id) break;           // given up on by the engine
      const from = Atomics.add(job, 1, chunk);
      if (from >= n) break;
      const rows = Math.min(chunk, n - from);
      const d = dims.slice();
      d[0] = rows;
      const out = await infer(s, model, input.slice(from * row, (from + rows) * row), d);
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
    // Requests arrive on a port straight from the engine worker, which is
    // blocked while it waits and cannot go through the page.
    msg.port.onmessage = ({ data: req }) => run(req);
    started = load(msg.backend, msg.threads, msg.forceGpu);
    if (msg.backend === "webgpu") {
      // Tell the engine whether it has a GPU to choose.
      started.then(() => Atomics.store(job, GPU, 1),
                   err => { Atomics.store(job, GPU, 2); console.warn("WebGPU:", err.message || err); });
    }
    try { await started; } catch (err) { return; }
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
