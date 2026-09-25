// Starting the game engine: the Python worker and its pool of network workers.
//
// Shared by the game page (bridge.js) and the benchmark (bench.html).  The
// engine worker (engine-worker.js) runs the project's Python; every network
// call it makes is answered by the network workers (ort-worker.js), one per
// core the device offers, each with its own slot of one shared buffer.  How
// many of them a batch is actually spread over is learned on the device from
// the bots' own calls (see the tuning in engine-worker.js) and kept by the page.
(function () {
  "use strict";

  const CTL_INTS = 64;
  const FIRST_FLOATS = 4 * 1024 * 1024;   // 16 MB: the first worker can take any batch whole
  const OTHER_FLOATS = 1024 * 1024;       // 4 MB each: the others only ever take a slice
  const MAX_WORKERS = 8;                  // past this, a phone runs short of memory first

  // One per core, up to eight -- and fewer on a device short of memory: each
  // worker costs about 50 MB once it has loaded a network.  (Only Chrome says
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
  function spawn(models, { workers = defaultWorkers(), extra = {} } = {}) {
    const slots = [];
    let bytes = 0;
    for (let i = 0; i < workers; i++) {
      const size = CTL_INTS * 4 + (i === 0 ? FIRST_FLOATS : OTHER_FLOATS) * 4;
      slots.push({ offset: bytes, bytes: size });
      bytes += size;
    }
    const sab = new SharedArrayBuffer(bytes);
    const engine = new Worker("engine-worker.js");
    const networks = [], ports = [];
    slots.forEach((slot, i) => {
      const channel = new MessageChannel();
      const worker = new Worker("ort-worker.js", { type: "module" });
      worker.onmessage = ({ data }) => {
        if (data.type === "error") console.warn(`network worker ${i}:`, data.message);
      };
      // Alone, a worker splits each call over a few threads of its own; in a
      // pool each runs on one, and the pool splits the batch instead.
      worker.postMessage({ type: "init", sab, offset: slot.offset, bytes: slot.bytes, models,
                           port: channel.port1, lazy: i > 0,
                           threads: workers > 1 ? 1 : undefined }, [channel.port1]);
      networks.push(worker);
      ports.push(channel.port2);
    });
    engine.postMessage(Object.assign({}, extra, { type: "init", sab, slots, ports, models }),
                       ports);
    return { engine, networks };
  }

  self.EnginePool = { spawn, defaultWorkers };
})();
