// The exact solver of Ultimate Tic-Tac-Toe's v3 bot, on a worker of its own.
//
// On the desktop the v3 bot proves positions on a thread beside its search:
// from about 38 open cells it tries to solve the position outright while the
// search runs, and it keeps solving the position you have to answer while
// you think.  A browser has no threads in Python, so this worker is that
// thread: its own copy of the native core (uttt-core.js) holding the solver
// (`uttt_rs.Searcher(1)`), and the engine asks it for work (webgui.py's
// RemoteSolver).
//
// Requests come on a port from the engine; answers go back through shared
// memory, since the engine may be busy searching and not reading messages:
//   Int32 control block: [0] state (0 idle, 1 working, 2 done, 3 failed),
//     [1] request id, [2] answer length in bytes, [3] cancel: an id to stop
//   then the answer, as JSON text.
//
// A request is the position (x, o, active, stm) and what to do with it:
//   "empties"   -> the open cells left
//   "canonical" -> the canonical board, 81 bytes
//   "solve"     -> root_solve(budget, stop_on_win, seconds): [[move, value]...]
//   "job"       -> keep solving, in slices of 0.1 s, until every root move
//                  is known, `seconds` pass or the job is cancelled; skipped
//                  (no answer) when more than `empties` cells are open.
importScripts("uttt-core.js");

let W = null, handle = 0, ctl = null, bytes = null;
let ready = null;

function view(ptr, n) { return new Uint8Array(W.memory.buffer, ptr, n); }

function out() {
  const n = W.out_len();
  return n ? view(W.out_ptr(), n).slice() : new Uint8Array(0);
}

function setState({ x, o, active, stm }) {
  const px = W.alloc(18), po = W.alloc(18);
  const dv = new DataView(W.memory.buffer);
  for (let i = 0; i < 9; i++) {
    dv.setUint16(px + 2 * i, x[i], true);
    dv.setUint16(po + 2 * i, o[i], true);
  }
  const bad = W.set_root_state(handle, 0, px, po, active, stm);
  W.dealloc(px, 18);
  W.dealloc(po, 18);
  if (bad !== 0) throw new Error("bad state");
}

function solve(budget, stopOnWin, seconds) {
  const n = W.root_solve(handle, 0, budget, stopOnWin ? 1 : 0, seconds);
  const raw = out();
  const pairs = [];
  for (let i = 0; i < n; i++) pairs.push([raw[2 * i], (raw[2 * i + 1] << 24) >> 24]);
  return pairs;
}

// rs_agent._classify: every root move known.
const settled = pairs => pairs.length > 0 && pairs.every(([, v]) => v !== 2)
  && pairs.some(([, v]) => v === 1 || v === 0 || v === -1);

function answer(id, value) {
  const text = new TextEncoder().encode(JSON.stringify(value));
  if (text.length > bytes.length) throw new Error("answer too long");
  bytes.set(text, 0);
  ctl[2] = text.length;
  if (Atomics.load(ctl, 1) === id) {
    Atomics.store(ctl, 0, 2);
    Atomics.notify(ctl, 0);
  }
}

async function handleRequest({ id, req }) {
  try {
    await ready;
    setState(req);
    if (req.op === "empties") return answer(id, W.root_open_empties(handle, 0));
    if (req.op === "canonical") {
      W.root_canonical(handle, 0);
      return answer(id, Array.from(out()));
    }
    if (req.op === "solve") return answer(id, solve(req.budget, req.stop, req.seconds));
    if (req.op === "job") {
      const t0 = performance.now();
      if (W.root_open_empties(handle, 0) > req.empties) return answer(id, { pairs: null, ms: 0 });
      let pairs = [];
      const deadline = t0 + req.seconds * 1000;
      // Until cancelled -- or until a newer request comes: it takes over.
      while (Atomics.load(ctl, 3) !== id && Atomics.load(ctl, 1) === id) {
        const left = (deadline - performance.now()) / 1000;
        if (left <= 0) break;
        pairs = solve(2 ** 62, false, Math.min(0.1, left));
        if (settled(pairs)) break;
        // Let a cancel land between slices.
        await new Promise(r => setTimeout(r, 0));
      }
      return answer(id, { pairs, ms: performance.now() - t0 });
    }
    throw new Error("unknown request " + req.op);
  } catch (err) {
    console.error("solver:", err);
    if (Atomics.load(ctl, 1) === id) {
      Atomics.store(ctl, 0, 3);
      Atomics.notify(ctl, 0);
    }
  }
}

self.onmessage = ({ data: msg }) => {
  if (msg.type !== "init") return;
  ctl = new Int32Array(msg.sab, 0, 16);
  bytes = new Uint8Array(msg.sab, 64);
  ready = instantiateUttt().then(exports => {
    W = exports;
    // The defaults of uttt_rs.Searcher(1), as rs_agent makes its solver.
    handle = W.searcher_new(1, 1.6, 0.25, 18, 20000, 300000, 0, 0, 0, 0, 0);
  });
  // Requests are handled one after another.
  let chain = Promise.resolve();
  msg.port.onmessage = ({ data }) => { chain = chain.then(() => handleRequest(data)); };
};
