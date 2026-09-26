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
const HALF = 3;            // job block: 1 when the GPU runs half precision
let job = null, ctl = null, input = null, output = null, models = null;
let ort = null, provider = "wasm", gpuHalf = false;
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
    // onnxruntime asks for 16-bit shaders whenever the adapter has them.
    gpuHalf = adapter.features.has("shader-f16");
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
// (the tuning in engine-worker.js); `variant` names the one to run.  The GPU
// runs float ("float") or half precision ("half"), from the files made for it
// where there are any (models.json `gpu_file`, `gpu16_file`; gpu_models.py).
function fileOf(name, variant) {
  const m = models[name];
  if (provider === "webgpu") {
    if (variant === "half" && m.gpu16_file) return m.gpu16_file;
    return m.gpu_file || m.float_file || m.file;
  }
  return variant === "float" && m.float_file ? m.float_file : m.file;
}

const files = {};
function bytesOf(file) {
  if (!files[file]) {
    files[file] = fetch(new URL("./" + file, import.meta.url))
      .then(r => r.arrayBuffer()).then(b => new Uint8Array(b));
  }
  return files[file];
}

function session(name, variant) {
  const key = `${name}:${fileOf(name, variant)}`;
  if (!sessions[key]) {
    sessions[key] = bytesOf(fileOf(name, variant)).then(bytes => ort.InferenceSession.create(bytes, {
      executionProviders: [provider], graphOptimizationLevel: "all",
    }));
  }
  return sessions[key];
}

// On the GPU, every new batch size can mean compiling new shaders -- slow, and
// the bots' batches come in every size.  So the GPU runs batches rounded up
// to a few sizes (16, 32, 64, 128, then multiples of 64), padded with empty
// positions whose answers are dropped.  (engine-worker.js's `gpuPad` agrees.)
//
// Each size gets a session of its own with the batch fixed, and graph capture
// where it works: the first run is recorded and the rest replay it, which
// saves the GPU the overhead of setting up each of a network's many small
// steps every time.  The input stays in one GPU buffer, rewritten each call,
// as capture needs.
//
// Capture is checked before it is trusted.  onnxruntime-web's replay was
// found to ignore new input for some networks (Hex and Connect Four: every
// replay answered the recorded positions again) while getting others right
// (Ultimate, v3).  So a captured session is recorded with one set of random
// positions, replayed with another, and used only if that replay agrees with
// the same network run without capture; otherwise the size runs uncaptured.
const unpadded = new Set();  // networks whose outputs do not lead with the batch
const padTo = rows => (rows <= 128 ? Math.max(16, 2 ** Math.ceil(Math.log2(rows))) : Math.ceil(rows / 64) * 64);
const runners = {};          // "model:variant:size" -> promise of (input) -> outputs

function randomInput(n) {
  const x = new Float32Array(n);
  for (let i = 0; i < n; i++) x[i] = Math.random() < 0.3 ? 1 : 0;
  return x;
}

function gpuRunner(model, variant, size, rest) {
  const key = `${model}:${variant}:${size}`;
  if (!(key in runners)) {
    runners[key] = (async () => {
      const bytes = await bytesOf(fileOf(model, variant));
      const dims = [size, ...rest];
      const floats = size * rest.reduce((a, b) => a * b, 1);
      let fixed;
      try {
        const s = await ort.InferenceSession.create(bytes, {
          executionProviders: ["webgpu"], graphOptimizationLevel: "all",
          freeDimensionOverrides: { batch: size },
        });
        fixed = async x => s.run({ x: new ort.Tensor("float32", x, dims) });
      } catch (err) {
        // The dynamic session, fed the padded batch.
        const s = await session(model, variant);
        fixed = async x => s.run({ x: new ort.Tensor("float32", x, dims) });
      }
      try {
        const s = await ort.InferenceSession.create(bytes, {
          executionProviders: ["webgpu"], graphOptimizationLevel: "all",
          freeDimensionOverrides: { batch: size }, enableGraphCapture: true,
          preferredOutputLocation: "gpu-buffer",
        });
        const device = ort.env.webgpu.device;
        const buffer = device.createBuffer({
          size: floats * 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        const input = ort.Tensor.fromGpuBuffer(buffer, { dataType: "float32", dims });
        const replay = async x => {
          device.queue.writeBuffer(buffer, 0, x);
          const got = await s.run({ x: input });
          const out = {};
          for (const [name, t] of Object.entries(got)) out[name] = { dims: t.dims.slice(), data: await t.getData() };
          return out;
        };
        await replay(randomInput(floats));                   // recorded
        const probe = randomInput(floats);
        const a = await replay(probe), b = await fixed(probe);  // replayed, and not
        let worst = 0;
        for (const name of Object.keys(b)) {
          for (let i = 0; i < b[name].data.length; i++) {
            worst = Math.max(worst, Math.abs(a[name].data[i] - b[name].data[i]));
          }
        }
        if (!(worst < 1e-3)) throw new Error(`replay disagrees by ${worst}`);
        return replay;
      } catch (err) {
        console.warn(`no graph capture for ${model} (${variant}, ${size}):`, err.message || err);
        return fixed;
      }
    })();
  }
  return runners[key];
}

async function infer(model, variant, x, dims) {
  const rows = dims[0];
  if (provider !== "webgpu") {
    return (await session(model, variant)).run({ x: new ort.Tensor("float32", x, dims) });
  }
  const size = unpadded.has(model) ? rows : padTo(rows);
  let padded = x;
  if (size !== rows) {
    padded = new Float32Array(size * (x.length / rows));
    padded.set(x, 0);
  }
  const run = unpadded.has(model) ? null : await gpuRunner(model, variant, size, dims.slice(1));
  const out = run ? await run(padded)
    : await (await session(model, variant)).run({ x: new ort.Tensor("float32", padded, [size, ...dims.slice(1)]) });
  if (size === rows) return out;
  const cut = {};
  for (const [name, t] of Object.entries(out)) {
    if (t.dims[0] !== size) { unpadded.add(model); return infer(model, variant, x, dims); }
    cut[name] = { dims: [rows, ...t.dims.slice(1)], data: t.data.subarray(0, (t.data.length / size) * rows) };
  }
  return cut;
}

// A job: take positions from the shared counter until none are left -- or,
// with `first` and `count`, exactly those (the GPU's share of a batch it
// splits with the CPU workers, who take the rest from the counter).
async function run({ id, model, variant, dims, n, chunk, first, count }) {
  const row = dims.slice(1).reduce((a, b) => a * b, 1);
  let answered = 0;
  try {
    await started;
    const names = models[model].outputs;
    let fixed = count !== undefined;
    for (;;) {
      if (Atomics.load(job, 0) !== id) break;           // given up on by the engine
      let from, rows;
      if (fixed) {
        if (count <= 0) break;
        from = first; rows = count; count = 0;
      } else {
        from = Atomics.add(job, 1, chunk);
        if (from >= n) break;
        rows = Math.min(chunk, n - from);
      }
      const d = dims.slice();
      d[0] = rows;
      const out = await infer(model, variant, input.slice(from * row, (from + rows) * row), d);
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
      started.then(() => {
        // Half precision only where the GPU has 16-bit float shaders.
        Atomics.store(job, HALF, gpuHalf ? 1 : 0);
        Atomics.store(job, GPU, 1);
      },
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
