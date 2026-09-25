// The engine worker: the project's Python game server, in Pyodide.
//
// The page sends the same requests it would send the desktop server
// ({id, path, body}); this worker answers them from `webgui.dispatch` and, in
// between, does the work the desktop server did on its threads -- the bots'
// moves, the live analysis and the match review -- one step at a time, so a
// request never waits behind more than one step.
importScripts("pyodide/pyodide.js");

const CTL_INTS = 64;
let slots = [];            // one per network worker: {ctl, data, port}
let py = null, web = null;
const queue = [];
let busy = false;
let holding = false;       // a bot's move is chosen and waiting out the pace
let pumpTimer = null;

// ---------------------------------------------------------- the network call
// Synchronous on purpose: see ort-worker.js.
//
// A search asks for a batch of positions at a time.  With a pool of network
// workers the batch is cut into slices, one per worker, all run at once, and
// put back together in order -- the caller cannot tell.  How many workers a
// batch is spread over is learned on this device: see the tuning below.

// Send `rows` positions, starting at row `from`, to one worker's slot.
function post(slot, model, flat, dims, from, rows) {
  const row = flat.length / dims[0];
  if (rows * row > slot.data.length) throw new Error("input too large for the shared buffer");
  slot.data.set(flat.subarray(from * row, (from + rows) * row), 0);
  Atomics.store(slot.ctl, 0, 1);
  const d = Array.from(dims);
  d[0] = rows;
  slot.port.postMessage({ model, dims: d, n: rows * row });
}

// Wait for one worker's answer and read its outputs out of the slot.  The
// extra workers get a deadline: one that died (a phone out of memory, say)
// would otherwise be waited for forever.
const DEADLINE = 30000;
function collect(slot, patient) {
  const { ctl, data } = slot;
  const t0 = performance.now();
  while (Atomics.load(ctl, 0) === 1) {
    Atomics.wait(ctl, 0, 1, 2000);
    if (!patient && performance.now() - t0 > DEADLINE) return null;
  }
  const failed = Atomics.load(ctl, 0) === 3;
  const outs = [];
  let at = 0;
  for (let i = 0; !failed && i < ctl[1]; i++) {
    const rank = ctl[2 + i * 5];
    const shape = [];
    let size = 1;
    for (let k = 0; k < rank; k++) { shape.push(ctl[3 + i * 5 + k]); size *= ctl[3 + i * 5 + k]; }
    outs.push({ data: data.slice(at, at + size), dims: shape });
    at += size;
  }
  Atomics.store(ctl, 0, 0);
  return failed ? null : outs;
}

const whole = new Set();   // models whose outputs cannot be cut by row

function inferOn(workers, model, flat, dims) {
  const n = dims[0];
  let k = whole.has(model) ? 1 : Math.max(1, Math.min(workers, slots.length, n));
  // A slice too big for the smaller slots goes whole to the first, the big one.
  if (k > 1 && Math.ceil(n / k) * (flat.length / n) > slots[k - 1].data.length) k = 1;
  const parts = [];
  for (let i = 0, from = 0; i < k; i++) {
    const rows = Math.floor(n / k) + (i < n % k ? 1 : 0);
    post(slots[i], model, flat, dims, from, rows);
    parts.push(rows);
    from += rows;
  }
  // Collect every slot, even after a failure, so each is idle again.
  const answers = parts.map((_, i) => collect(slots[i], i === 0));
  const lost = answers.findIndex(a => a === null);
  if (lost === 0) throw new Error("the network failed");
  if (lost > 0) {
    // An extra worker failed: carry on without it and every one after it.
    console.warn(`network worker ${lost} failed; using ${lost} from now on`);
    slots.length = lost;
    setTuning(null);
    for (const key of Object.keys(learned)) delete learned[key];
    return inferOn(workers, model, flat, dims);
  }
  if (k === 1) return answers[0];
  // Every output must lead with the batch, or its slices cannot be joined:
  // run such a model whole from now on.
  if (answers.some((outs, i) => outs.some(o => o.dims[0] !== parts[i]))) {
    whole.add(model);
    return inferOn(1, model, flat, dims);
  }
  return answers[0].map((first, j) => {
    const all = new Float32Array(answers.reduce((t, outs) => t + outs[j].data.length, 0));
    let at = 0;
    for (const outs of answers) { all.set(outs[j].data, at); at += outs[j].data.length; }
    const shape = first.dims.slice();
    shape[0] = n;
    return { data: all, dims: shape };
  });
}

// ------------------------------------------------------------ the tuning
// How many workers a batch is best spread over depends on the device -- how
// many cores it has and, on a phone, how many of them are fast ones: a batch
// cut into eight waits for its slowest slice, and that slice may be on a slow
// core -- and on the network and the size of the batch.  So it is learned
// from the bots' own calls.  Each network and batch size starts on every
// worker and steps down, one spread at a time, for as long as fewer workers
// are no slower -- past the cores there are, eight and six can tie, and four
// is still ahead -- timing the two spreads being compared in turns, so a
// device warming up or slowing down is fair to both.  After that, now and
// then a spread either side of the best is timed again, since a phone slows
// down as it heats up.  The page keeps what was learned for the next visit.
const TUNING_VERSION = 2;
const SAMPLES = 5;         // calls to time a spread on before judging it
const MARGIN = 0.97;       // how much faster a spread must be to take over the best
const RECHECK = 25;        // calls between tries of a neighbour of the best
const WARMUP = 2;          // a worker's first calls on a network load and settle it
let choices = [1];         // the spreads worth trying on this pool
const learned = {};        // "model|8" -> see `entry`

function entry(best) {
  return {
    best,                    // the spread in use
    trying: null,            // the spread being compared with it, if any
    settled: false,          // done stepping down
    saved: false,            // told the page
    calls: 0,
    times: Object.fromEntries(choices.map(k => [k, []])),   // recent ms per position
  };
}

function setTuning(saved) {
  const n = slots.length;
  choices = [...new Set([1, 2, 3, 4, 6, 8, n])].filter(k => k <= n).sort((a, b) => a - b);
  if (!saved || saved.version !== TUNING_VERSION || saved.workers !== n) return;
  for (const [key, { best, ms }] of Object.entries(saved.learned || {})) {
    if (!choices.includes(best)) continue;
    const L = learned[key] = entry(best);
    L.settled = L.saved = true;
    for (const k of choices) if (ms[k] != null) L.times[k] = Array(SAMPLES).fill(ms[k]);
  }
}

// Batches are compared only with batches of about the same size -- 8 to 15
// positions with 8 to 15, and so on -- since a small one costs more per
// position and may be best on fewer workers.  (A PUCT search asks for up to
// 24 at a time, the v3 bot for 48.)
const sizeOf = rows => 2 ** Math.min(6, Math.floor(Math.log2(rows)));

// The typical time of the last few calls: one slow call (the device busy
// with something else) moves a median less than it moves an average.
function typical(list) {
  const s = [...list].sort((a, b) => a - b);
  return s.length ? s[Math.floor(s.length / 2)] : Infinity;
}

const below = k => choices[choices.indexOf(k) - 1];
const above = k => choices[choices.indexOf(k) + 1];

function spreadFor(model, rows) {
  if (choices.length < 2 || rows < 2 || whole.has(model)) return { k: 1 };
  const key = `${model}|${sizeOf(rows)}`;
  const L = learned[key] || (learned[key] = entry(choices[choices.length - 1]));
  L.calls++;
  if (!L.settled) {
    if (L.times[L.best].length >= SAMPLES && L.trying === null) {
      L.trying = below(L.best) ?? null;
      if (L.trying === null) L.settled = true;
      else L.times[L.trying] = [];
    }
    if (L.trying !== null) return { k: L.calls % 2 ? L.trying : L.best, key };
    return { k: L.best, key };
  }
  if (L.trying === null && L.calls % RECHECK === 0) {
    const side = (L.calls / RECHECK) % 2 ? below(L.best) : above(L.best);
    const k = side ?? below(L.best) ?? above(L.best);
    if (k !== undefined) { L.trying = k; L.times[k] = []; }
  }
  if (L.trying !== null) return { k: L.calls % 2 ? L.trying : L.best, key };
  return { k: L.best, key };
}

function record(key, k, ms, rows) {
  const L = learned[key];
  if (!L) return;
  const list = L.times[k];
  list.push(ms / rows);
  if (list.length > SAMPLES) list.shift();
  let changed = false;
  if (L.trying !== null && k === L.trying && list.length >= SAMPLES) {
    const mine = typical(list), best = typical(L.times[L.best]);
    // Stepping down, fewer workers win a tie; after that, a spread has to
    // be clearly faster to take over.
    const better = L.settled ? mine < best * MARGIN : mine <= best;
    if (better) { L.best = L.trying; changed = true; }
    L.trying = null;
    if (!L.settled && !better) L.settled = true;
  } else if (!L.settled && L.trying === null && below(L.best) === undefined
             && list.length >= SAMPLES) {
    L.settled = true;
  }
  if (L.settled && (changed || !L.saved)) {
    L.saved = true;
    self.postMessage({ type: "tuning", tuning: tuningNow() });
  }
}

function tuningNow() {
  const out = {};
  for (const [key, L] of Object.entries(learned)) {
    if (!L.saved) continue;
    const timed = choices.filter(k => L.times[k].length);
    out[key] = {
      best: L.best,
      ms: Object.fromEntries(timed.map(k => [k, +typical(L.times[k]).toFixed(3)])),
    };
  }
  return { version: TUNING_VERSION, workers: slots.length, learned: out };
}

const counters = { calls: 0, rows: 0, ms: 0 };    // read by bench.html
self.inferStats = counters;

function inferSync(model, flat, dims) {
  const { k, key } = spreadFor(model, dims[0]);
  // A worker's first call on a network loads it: not a time to learn from.
  const used = slots.slice(0, Math.min(k, dims[0]));
  const cold = used.some(s => (s.warm[model] || 0) < WARMUP);
  const t0 = performance.now();
  const outs = inferOn(k, model, flat, dims);
  const ms = performance.now() - t0;
  used.forEach(s => { s.warm[model] = (s.warm[model] || 0) + 1; });
  if (key && !cold) record(key, k, ms, dims[0]);
  counters.calls++;
  counters.rows += dims[0];
  counters.ms += ms;
  return outs;
}

const progress = (stage, detail) => self.postMessage({ type: "progress", stage, detail });

// ------------------------------------------------- the native UTTT core
// `uttt_rs` -- the Rust search behind Ultimate Tic-Tac-Toe's v3 bot -- built
// for WASI (webapp/uttt_wasm).  It asks the host for six things: the time
// (its solver runs on a clock), random bytes (for its hash tables), and four
// it never really uses.  Python reaches the exports as `js.utttWasm`.
async function loadUttt() {
  let memory = null;
  const view = () => new DataView(memory.buffer);
  const wasi = {
    clock_time_get(_id, _precision, out) {
      const ns = BigInt(Math.round((performance.timeOrigin + performance.now()) * 1e6));
      view().setBigUint64(out, ns, true);
      return 0;
    },
    random_get(ptr, len) {
      crypto.getRandomValues(new Uint8Array(memory.buffer, ptr, len));
      return 0;
    },
    environ_sizes_get(count, size) {
      view().setUint32(count, 0, true);
      view().setUint32(size, 0, true);
      return 0;
    },
    environ_get() { return 0; },
    fd_write(_fd, iovs, n, written) {
      const dv = view();
      let text = "", total = 0;
      for (let i = 0; i < n; i++) {
        const ptr = dv.getUint32(iovs + i * 8, true), len = dv.getUint32(iovs + i * 8 + 4, true);
        text += new TextDecoder().decode(new Uint8Array(memory.buffer, ptr, len));
        total += len;
      }
      if (text.trim()) console.log("[uttt core]", text.trim());
      dv.setUint32(written, total, true);
      return 0;
    },
    proc_exit(code) { throw new Error(`the UTTT core stopped (${code})`); },
  };
  try {
    const bytes = await (await fetch("uttt_wasm.wasm")).arrayBuffer();
    const { instance } = await WebAssembly.instantiate(bytes, { wasi_snapshot_preview1: wasi });
    memory = instance.exports.memory;
    self.utttWasm = instance.exports;
  } catch (err) {
    // Only the v3 bot needs it; every other game and bot runs without.
    console.warn("UTTT core unavailable:", err);
  }
}

async function boot(msg) {
  slots = (msg.slots || [{ offset: 0, bytes: msg.sab.byteLength }]).map((s, i) => ({
    ctl: new Int32Array(msg.sab, s.offset, CTL_INTS),
    data: new Float32Array(msg.sab, s.offset + CTL_INTS * 4, s.bytes / 4 - CTL_INTS),
    port: msg.ports ? msg.ports[i] : msg.port,
    warm: {},                // network -> calls this worker has answered
  }));
  setTuning(msg.tuning);
  progress("python", "Starting Python");
  const core = loadUttt();
  py = await loadPyodide({ indexURL: new URL("pyodide/", self.location).href });
  progress("numpy", "Loading numpy");
  await py.loadPackage(["numpy"], { messageCallback: () => {} });
  progress("engine", "Unpacking the game engine");
  const zip = new Uint8Array(await (await fetch("engine.zip")).arrayBuffer());
  py.unpackArchive(zip, "zip", { extractDir: "/proj" });
  restoreProfiles();
  py.runPython("import sys; sys.path.insert(0, '/proj')");
  web = py.pyimport("webgui");
  web.set_infer(inferSync);
  await core;
  let placements = await (await fetch("engine.json")).text();
  if (!self.utttWasm) {
    // No native core: leave out the network only it can drive.
    const p = JSON.parse(placements);
    delete p.uttt_v3;
    placements = JSON.stringify(p);
  }
  web.boot("/proj", JSON.stringify(msg.models), placements);
  self.postMessage({ type: "ready" });
  pump();
}

// ----------------------------------------------------- player profiles
// Rated games write each game's users_*.json.  They live in Pyodide's memory,
// so the page keeps a copy in localStorage (it owns storage; a worker has
// none) and hands it back at the next start.
let profiles = {};
function restoreProfiles() {
  for (const [name, text] of Object.entries(profiles)) {
    try { py.FS.writeFile(`/proj/runs/${name}`, text); } catch (err) { /* ignore */ }
  }
}
function collectProfiles() {
  const out = {};
  try {
    for (const name of py.FS.readdir("/proj/runs")) {
      if (/^users.*\.json$/.test(name)) out[name] = py.FS.readFile(`/proj/runs/${name}`, { encoding: "utf8" });
    }
  } catch (err) { /* no runs dir yet */ }
  return out;
}

// ------------------------------------------------------------- the loop
function answer({ id, path, body }) {
  const raw = web.dispatch(path, body === undefined || body === null ? null : JSON.stringify(body));
  const reply = JSON.parse(raw);
  self.postMessage({ type: "reply", id, status: reply.status, body: reply.body });
  if (path === "/api/user" || path === "/api/move") {
    self.postMessage({ type: "profiles", files: collectProfiles() });
  }
}

function schedule(ms) {
  clearTimeout(pumpTimer);
  pumpTimer = setTimeout(pump, ms);
}

// Answer everything waiting, then do one step of background work, then yield
// so that new requests get in before the next step.
function pump() {
  if (!web || busy) return;
  busy = true;
  try {
    while (queue.length) answer(queue.shift());
    if (holding) return;
    const kind = web.work();
    if (kind === "bot") {
      const wait = web.step("bot");
      if (wait >= 0) {
        holding = true;
        busy = false;
        // The pace: a move chosen too fast waits, so it can be seen being made.
        setTimeout(() => {
          busy = true;
          try { while (queue.length) answer(queue.shift()); web.step("commit"); }
          catch (err) { report(err); }
          finally { busy = false; holding = false; }
          schedule(0);
        }, wait * 1000);
        return;
      }
      schedule(0);
    } else if (kind) {
      web.step(kind);
      schedule(kind === "analysis" ? 5 : 0);
    } else {
      schedule(100);
    }
  } catch (err) {
    report(err);
    schedule(250);
  } finally {
    busy = false;
  }
}

function report(err) {
  console.error(err);
  self.postMessage({ type: "error", message: String(err && err.message || err) });
}

self.onmessage = ({ data: msg }) => {
  if (msg.type === "init") {
    profiles = msg.profiles || {};
    boot(msg).catch(err => self.postMessage({ type: "fatal", message: String(err && err.message || err) }));
  } else if (msg.type === "bench") {
    // bench.html: time the bots on this device.  Never sent by the game page.
    let result;
    try { result = String(py.runPython(msg.code)); } catch (err) { result = "ERROR " + err; }
    self.postMessage({ type: "bench", id: msg.id, result, tuning: tuningNow() });
  } else if (msg.type === "request") {
    queue.push(msg);
    if (!busy) schedule(0);
  }
};
