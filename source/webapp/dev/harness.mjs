// Node stand-in for the browser: Pyodide on this thread, onnxruntime on a
// worker thread, and a synchronous inference call between them over shared
// memory -- the same protocol as webapp/static/{engine,ort}-worker.js.
import { loadPyodide } from "pyodide";
import { Worker } from "node:worker_threads";
import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const HERE = path.dirname(fileURLToPath(import.meta.url));
// webapp/site in the main project; the repo root in its source/ copy.
const SITE = existsSync(path.resolve(HERE, "../site")) ? path.resolve(HERE, "../site")
                                                      : path.resolve(HERE, "../../..");

export const CTL_INTS = 64;
export const DATA_FLOATS = 8 * 1024 * 1024;

export async function start({ verbose = false } = {}) {
  const sab = new SharedArrayBuffer(CTL_INTS * 4 + DATA_FLOATS * 4);
  const ctl = new Int32Array(sab, 0, CTL_INTS);
  const data = new Float32Array(sab, CTL_INTS * 4, DATA_FLOATS);
  const models = JSON.parse(readFileSync(path.join(SITE, "models/models.json"), "utf8"));
  const worker = new Worker(path.join(HERE, "ort-node-worker.mjs"),
                            { workerData: { sab, site: SITE, models } });
  await new Promise((ok, bad) => { worker.once("message", ok); worker.once("error", bad); });

  let calls = 0, ms = 0;
  function inferSync(model, flat, dims) {
    const t = performance.now();
    data.set(flat, 0);
    Atomics.store(ctl, 0, 1);
    worker.postMessage({ model, dims: Array.from(dims), n: flat.length });
    while (Atomics.load(ctl, 0) === 1) Atomics.wait(ctl, 0, 1, 1000);
    if (Atomics.load(ctl, 0) === 3) throw new Error("inference failed");
    const outs = [];
    let at = 0;
    const count = ctl[1];
    for (let i = 0; i < count; i++) {
      const rank = ctl[2 + i * 5];
      const dimsOut = [];
      let size = 1;
      for (let k = 0; k < rank; k++) { dimsOut.push(ctl[3 + i * 5 + k]); size *= ctl[3 + i * 5 + k]; }
      outs.push({ data: data.slice(at, at + size), dims: dimsOut });
      at += size;
    }
    Atomics.store(ctl, 0, 0);
    calls++; ms += performance.now() - t;
    return outs;
  }

  // The native UTTT core as WASM, with the WASI calls it makes (see engine-worker.js).
  {
    let memory = null;
    const view = () => new DataView(memory.buffer);
    const wasi = {
      clock_time_get(_i, _p, out) { view().setBigUint64(out, BigInt(Math.round((performance.timeOrigin + performance.now()) * 1e6)), true); return 0; },
      random_get(ptr, len) { crypto.getRandomValues(new Uint8Array(memory.buffer, ptr, len)); return 0; },
      environ_sizes_get(c, s) { view().setUint32(c, 0, true); view().setUint32(s, 0, true); return 0; },
      environ_get() { return 0; },
      fd_write(_fd, iovs, n, written) { let t = 0; for (let i = 0; i < n; i++) t += view().getUint32(iovs + i * 8 + 4, true); view().setUint32(written, t, true); return 0; },
      proc_exit(code) { throw new Error("exit " + code); },
    };
    const { instance } = await WebAssembly.instantiate(readFileSync(path.join(SITE, "uttt_wasm.wasm")),
                                                       { wasi_snapshot_preview1: wasi });
    memory = instance.exports.memory;
    globalThis.utttWasm = instance.exports;
  }
  const py = await loadPyodide();
  await py.loadPackage(["numpy"]);
  const zip = new Uint8Array(readFileSync(path.join(SITE, "engine.zip")));
  py.unpackArchive(zip, "zip", { extractDir: "/proj" });
  py.runPython("import sys; sys.path.insert(0, '/proj')");
  const web = py.pyimport("webgui");
  web.set_infer(inferSync);
  web.boot("/proj", JSON.stringify(models),
           readFileSync(path.join(SITE, "engine.json"), "utf8"));

  const api = (p, body) => {
    const r = JSON.parse(web.dispatch(p, body === undefined ? null : JSON.stringify(body)));
    if (verbose) console.log(p, r.status);
    return r.body;
  };
  // Run the background work until there is none, the way the worker does.
  function drain(limit = 400) {
    for (let i = 0; i < limit; i++) {
      const kind = web.work();
      if (!kind) return i;
      const wait = web.step(kind);
      if (kind === "bot" && wait >= 0) web.step("commit");
    }
    return limit;
  }
  return { py, web, api, drain, stats: () => ({ calls, ms }), close: () => worker.terminate() };
}
