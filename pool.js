// Starting the game engine: the Python worker and its pool of network workers.
//
// Shared by the game page (bridge.js) and the benchmark (bench.html).  The
// engine worker (engine-worker.js) runs the project's Python; every network
// call it makes is answered by the network workers (ort-worker.js), one per
// core the device offers, through one shared buffer.  How many of them a
// batch is actually shared out over is learned on the device from the bots'
// own calls (see the tuning in engine-worker.js) and kept by the page.
(function () {
  "use strict";

  const CTL_INTS = 64;
  const INPUT_FLOATS = 4 * 1024 * 1024;   // 16 MB: the biggest batch any search asks for
  const OUTPUT_FLOATS = 4 * 1024 * 1024;  // 16 MB: its answers
  const MAX_WORKERS = 8;                  // past this, a phone runs short of memory first

  // The pool: one per core, up to eight -- and fewer on a device short of
  // memory: each worker costs about 50 MB once it has loaded a network.  (Only Chrome says
  // how much memory there is; elsewhere the core count alone decides.)
  function defaultWorkers() {
    const cores = navigator.hardwareConcurrency || 2;
    const gb = navigator.deviceMemory;
    const room = gb === undefined ? MAX_WORKERS : gb <= 1 ? 2 : gb <= 2 ? 4 : MAX_WORKERS;
    return Math.max(1, Math.min(cores, room));
  }

  // Returns {engine, networks}: the workers are started and sent their `init`;
  // the caller listens on `engine` for "ready".  `extra` is merged into the
  // engine's init message (profiles, the saved tuning, ...).
  //
  // The first network worker is the old setup: alone, on a few threads of
  // its own, with every network loaded up front.  Beside it are `workers`
  // more on one thread each -- the pool -- which load a network when first
  // asked for it.  `workers: 0` is the site as it was before the pool.
  //
  // Where the browser has WebGPU, one more worker runs the networks on the
  // graphics card; it is last, and `layout.gpu` says so.  `gpu: false` leaves
  // it out.
  function spawn(models, { workers = defaultWorkers(), gpu = true, extra = {} } = {}) {
    const withGpu = gpu && workers > 1 && !!navigator.gpu;
    const count = 1 + (workers > 1 ? workers : 0) + (withGpu ? 1 : 0);
    // The shared buffer (see ort-worker.js): the job block, a control block
    // per worker, then the input and the output.
    const layout = { job: 0, ctl: [], gpu: withGpu ? count - 1 : -1 };
    let bytes = 64;
    for (let i = 0; i < count; i++) { layout.ctl.push(bytes); bytes += CTL_INTS * 4; }
    Object.assign(layout, { input: bytes, inputFloats: INPUT_FLOATS });
    bytes += INPUT_FLOATS * 4;
    Object.assign(layout, { output: bytes, outputFloats: OUTPUT_FLOATS });
    bytes += OUTPUT_FLOATS * 4;
    const sab = new SharedArrayBuffer(bytes);
    const engine = new Worker("engine-worker.js");
    const networks = [], ports = [];
    for (let i = 0; i < count; i++) {
      const channel = new MessageChannel();
      const worker = new Worker("ort-worker.js", { type: "module" });
      worker.onmessage = ({ data }) => {
        if (data.type === "error") console.warn(`network worker ${i}:`, data.message);
      };
      worker.postMessage({ type: "init", sab, layout: Object.assign({}, layout, { ctl: layout.ctl[i] }),
                           models, port: channel.port1, lazy: i > 0,
                           threads: i > 0 ? 1 : undefined,
                           backend: i === layout.gpu ? "webgpu" : "wasm",
                           // ?gpu=force: use even a software GPU (for testing).
                           forceGpu: /[?&]gpu=force\b/.test(location.search) }, [channel.port1]);
      networks.push(worker);
      ports.push(channel.port2);
    }
    engine.postMessage(Object.assign({}, extra, { type: "init", sab, layout, ports, models }),
                       ports);
    return { engine, networks };
  }

  self.EnginePool = { spawn, defaultWorkers };
})();
