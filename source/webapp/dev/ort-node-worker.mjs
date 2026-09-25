// The inference thread for the Node harness: onnxruntime-web on the WASM backend.
import { parentPort, workerData } from "node:worker_threads";
import { readFileSync } from "node:fs";
import path from "node:path";
import * as ort from "onnxruntime-web";

const { sab, site, models } = workerData;
const CTL_INTS = 64;
const ctl = new Int32Array(sab, 0, CTL_INTS);
const data = new Float32Array(sab, CTL_INTS * 4);
ort.env.wasm.numThreads = 1;
const sessions = {};

async function session(name) {
  if (!sessions[name]) {
    const bytes = readFileSync(path.join(site, models[name].file));
    sessions[name] = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  }
  return sessions[name];
}

parentPort.on("message", async ({ model, dims, n }) => {
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
    console.error(err);
    Atomics.store(ctl, 0, 3);
  }
  Atomics.notify(ctl, 0);
});
parentPort.postMessage("ready");
