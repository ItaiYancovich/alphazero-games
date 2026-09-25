// The engine worker: the project's Python game server, in Pyodide.
//
// The page sends the same requests it would send the desktop server
// ({id, path, body}); this worker answers them from `webgui.dispatch` and, in
// between, does the work the desktop server did on its threads -- the bots'
// moves, the live analysis and the match review -- one step at a time, so a
// request never waits behind more than one step.
importScripts("pyodide/pyodide.js");

const CTL_INTS = 64, ROWS = CTL_INTS - 1;
let jobBlock = null, input = null, output = null;
let slots = [];            // one per network worker: {ctl, port, warm}
let py = null, web = null;
const queue = [];
let busy = false;
let holding = false;       // a bot's move is chosen and waiting out the pace
let pumpTimer = null;

// ---------------------------------------------------------- the network call
// Synchronous on purpose: see ort-worker.js.
//
// A search asks for a batch of positions at a time.  The batch goes into the
// shared buffer once, and a job goes either to the first network worker alone
// or to some of the pool beside it, who share it out among themselves as they
// go (see ort-worker.js) and write the answers back in order -- the caller
// cannot tell.  Which is learned on this device: see the tuning below.

// Wait for one worker to finish its part.  The extra workers get a deadline:
// one that died (a phone out of memory, say) would otherwise be waited for
// forever.
const DEADLINE = 30000;
function finished(slot, patient) {
  const { ctl } = slot;
  const t0 = performance.now();
  while (Atomics.load(ctl, 0) === 1) {
    Atomics.wait(ctl, 0, 1, 2000);
    if (!patient && performance.now() - t0 > DEADLINE) return 3;
  }
  const state = Atomics.load(ctl, 0);
  Atomics.store(ctl, 0, 0);
  return state;
}

const whole = new Set();   // models whose outputs cannot be cut by position
let jobs = 0;

// `choice` is "old" -- the first worker alone, on its own threads, as the site
// always ran -- or "4x2": four of the pool, the batch cut into two pieces a
// worker (more pieces let a worker on a fast core take more of them).
function inferOn(choice, model, flat, dims) {
  const n = dims[0];
  if (whole.has(model)) choice = "old";
  const [k, per] = choice === "old" ? [0, 1] : choice.split("x").map(Number);
  const team = k === 0 ? [slots[0]] : slots.slice(1, 1 + Math.min(k, n));
  if (flat.length > input.length) throw new Error("input too large for the shared buffer");
  input.set(flat, 0);
  const chunk = k === 0 ? n : Math.max(1, Math.ceil(n / (team.length * per)));
  const id = ++jobs;
  Atomics.store(jobBlock, 1, 0);
  Atomics.store(jobBlock, 0, id);
  for (const slot of team) {
    Atomics.store(slot.ctl, 0, 1);
    slot.ctl[ROWS] = 0;
    slot.port.postMessage({ id, model, dims: Array.from(dims), n, chunk });
  }
  // Hear from every worker, even after a failure, so each is idle again.
  const states = team.map(slot => finished(slot, slot === slots[0]));
  Atomics.store(jobBlock, 0, 0);            // anyone still at it: stop
  const lost = states.indexOf(3);
  if (lost >= 0 && team[lost] === slots[0]) throw new Error("the network failed");
  if (lost >= 0) {
    // A worker of the pool failed: carry on without it and every one after it.
    const at = slots.indexOf(team[lost]);
    console.warn(`network worker ${at} failed; the pool is ${at - 1} from now on`);
    slots.length = at;
    setTuning(null);
    for (const key of Object.keys(learned)) delete learned[key];
    return inferOn("old", model, flat, dims);
  }
  if (states.includes(4)) {
    // Its outputs do not lead with the batch: run it in one piece from now on.
    whole.add(model);
    return inferOn("old", model, flat, dims);
  }
  // The shapes, from any worker that answered; the batch is all of them.
  const from = team.find(s => s.ctl[ROWS] > 0);
  const answered = team.reduce((t, s) => t + s.ctl[ROWS], 0);
  if (!from || answered !== n) throw new Error("the network answered part of the batch");
  const { ctl } = from;
  const outs = [];
  let at = 0;
  for (let i = 0; i < ctl[1]; i++) {
    const shape = [];
    for (let j = 0; j < ctl[2 + i * 5]; j++) shape.push(ctl[3 + i * 5 + j]);
    if (k > 0) shape[0] = n;
    const size = shape.reduce((a, b) => a * b, 1);
    outs.push({ data: output.slice(at, at + size), dims: shape });
    at += size;
  }
  return outs;
}

// ------------------------------------------------------------ the tuning
// Which way is fastest depends on the device -- how many cores it has, how
// many of them are fast ones, how hot it is -- and on the network and the
// size of the batch.  So it is learned from the bots' own calls: for each
// network and batch size, every choice is timed on a few real batches, taking
// turns so a device warming up or slowing down is fair to all, and the
// fastest is used from then on.  "old" is one of the choices, so the result
// is never slower than the site was.  Now and then the runners-up are timed
// again, since a phone slows down as it heats up.  The page keeps what was
// learned for the next visit.
const TUNING_VERSION = 3;
const SAMPLES = 5;         // calls to time a choice on before judging it
const MARGIN = 0.97;       // how much faster a choice must be to take over the best
const RECHECK = 25;        // calls between second looks at a runner-up
const WARMUP = 2;          // a worker's first calls on a network load and settle it
let choices = ["old"];
const learned = {};        // "model|8" -> see `entry`

function entry() {
  return {
    best: null,              // the choice in use, once every one has been timed
    trying: null,            // a runner-up being timed again, if any
    saved: false,            // told the page
    calls: 0,
    rechecks: 0,
    times: Object.fromEntries(choices.map(c => [c, []])),   // recent ms per position
  };
}

function setTuning(saved) {
  const pool = slots.length - 1;
  const sizes = [...new Set([2, 3, 4, 6, 8, pool])].filter(k => k <= pool).sort((a, b) => a - b);
  // Two pieces a worker only where there are enough workers for some to be
  // on slow cores.
  choices = ["old", ...sizes.map(k => `${k}x1`), ...sizes.filter(k => k >= 4).map(k => `${k}x2`)];
  if (!saved || saved.version !== TUNING_VERSION || saved.workers !== pool) return;
  for (const [key, { best, ms }] of Object.entries(saved.learned || {})) {
    if (!choices.includes(best)) continue;
    const L = learned[key] = entry();
    L.best = best;
    L.saved = true;
    for (const c of choices) {
      L.times[c] = Array(SAMPLES).fill(ms[c] != null ? ms[c] : c === best ? 0 : Infinity);
    }
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

function choiceFor(model, rows) {
  if (choices.length < 2 || rows < 2 || whole.has(model)) return { choice: "old" };
  const key = `${model}|${sizeOf(rows)}`;
  const L = learned[key] || (learned[key] = entry());
  L.calls++;
  if (L.best === null) {
    // Every choice in turn, the least timed first.
    const fewest = Math.min(...choices.map(c => L.times[c].length));
    const behind = choices.filter(c => L.times[c].length === fewest);
    return { choice: behind[L.calls % behind.length], key };
  }
  if (L.trying === null && L.calls % RECHECK === 0) {
    // The best two of the rest, in turn.
    const rest = choices.filter(c => c !== L.best)
      .sort((a, b) => typical(L.times[a]) - typical(L.times[b]));
    L.trying = rest[L.rechecks++ % Math.min(2, rest.length)];
    L.times[L.trying] = [];
  }
  // A runner-up is timed in turns with the best, so both see the same device.
  if (L.trying !== null) return { choice: L.calls % 2 ? L.trying : L.best, key };
  return { choice: L.best, key };
}

function record(key, choice, ms, rows) {
  const L = learned[key];
  if (!L || !L.times[choice]) return;
  const list = L.times[choice];
  list.push(ms / rows);
  if (list.length > SAMPLES) list.shift();
  let changed = false;
  if (L.best === null) {
    if (choices.some(c => L.times[c].length < SAMPLES)) return;
    L.best = choices.reduce((a, b) => (typical(L.times[b]) < typical(L.times[a]) ? b : a));
    changed = true;
  } else if (L.trying !== null && choice === L.trying && list.length >= SAMPLES) {
    if (typical(list) < typical(L.times[L.best]) * MARGIN) { L.best = L.trying; changed = true; }
    L.trying = null;
  }
  if (changed || !L.saved) {
    L.saved = true;
    self.postMessage({ type: "tuning", tuning: tuningNow() });
  }
}

function tuningNow() {
  const out = {};
  for (const [key, L] of Object.entries(learned)) {
    if (!L.saved) continue;
    const timed = choices.filter(c => L.times[c].length && isFinite(typical(L.times[c])));
    out[key] = {
      best: L.best,
      ms: Object.fromEntries(timed.map(c => [c, +typical(L.times[c]).toFixed(3)])),
    };
  }
  return { version: TUNING_VERSION, workers: slots.length - 1, learned: out };
}

const counters = { calls: 0, rows: 0, ms: 0 };    // read by bench.html
self.inferStats = counters;

function inferSync(model, flat, dims) {
  const { choice, key } = choiceFor(model, dims[0]);
  // A worker's first calls on a network load it: not a time to learn from.
  const k = choice === "old" ? 0 : parseInt(choice, 10);
  const used = k === 0 ? [slots[0]] : slots.slice(1, 1 + Math.min(k, dims[0]));
  const cold = used.some(s => (s.warm[model] || 0) < WARMUP);
  const t0 = performance.now();
  const outs = inferOn(choice, model, flat, dims);
  const ms = performance.now() - t0;
  used.forEach(s => { s.warm[model] = (s.warm[model] || 0) + 1; });
  if (key && !cold) record(key, choice, ms, dims[0]);
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
  const { sab, layout } = msg;
  jobBlock = new Int32Array(sab, layout.job, 16);
  input = new Float32Array(sab, layout.input, layout.inputFloats);
  output = new Float32Array(sab, layout.output, layout.outputFloats);
  slots = msg.ports.map((port, i) => ({
    ctl: new Int32Array(sab, layout.ctl[i], CTL_INTS),
    port,
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
