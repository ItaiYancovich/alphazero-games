// The engine worker: the project's Python game server, in Pyodide.
//
// The page sends the same requests it would send the desktop server
// ({id, path, body}); this worker answers them from `webgui.dispatch` and, in
// between, does the work the desktop server did on its threads -- the bots'
// moves, the live analysis and the match review -- one step at a time, so a
// request never waits behind more than one step.
importScripts("pyodide/pyodide.js");

const CTL_INTS = 64;
let ctl = null, data = null, port = null;
let py = null, web = null;
const queue = [];
let busy = false;
let holding = false;       // a bot's move is chosen and waiting out the pace
let pumpTimer = null;

// ---------------------------------------------------------- the network call
// Synchronous on purpose: see ort-worker.js.
function inferSync(model, flat, dims) {
  if (flat.length > data.length) throw new Error("input too large for the shared buffer");
  data.set(flat, 0);
  Atomics.store(ctl, 0, 1);
  port.postMessage({ model, dims: Array.from(dims), n: flat.length });
  while (Atomics.load(ctl, 0) === 1) Atomics.wait(ctl, 0, 1, 2000);
  if (Atomics.load(ctl, 0) === 3) { Atomics.store(ctl, 0, 0); throw new Error("the network failed"); }
  const outs = [];
  let at = 0;
  for (let i = 0; i < ctl[1]; i++) {
    const rank = ctl[2 + i * 5];
    const shape = [];
    let size = 1;
    for (let k = 0; k < rank; k++) { shape.push(ctl[3 + i * 5 + k]); size *= ctl[3 + i * 5 + k]; }
    outs.push({ data: data.slice(at, at + size), dims: shape });
    at += size;
  }
  Atomics.store(ctl, 0, 0);
  return outs;
}

const progress = (stage, detail) => self.postMessage({ type: "progress", stage, detail });

async function boot(msg) {
  ctl = new Int32Array(msg.sab, 0, CTL_INTS);
  data = new Float32Array(msg.sab, CTL_INTS * 4);
  port = msg.port;
  progress("python", "Starting Python");
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
  const placements = await (await fetch("engine.json")).text();
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
  } else if (msg.type === "request") {
    queue.push(msg);
    if (!busy) schedule(0);
  }
};
