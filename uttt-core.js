// `uttt_rs` -- the Rust search behind Ultimate Tic-Tac-Toe's v3 bot -- built
// for WASI (webapp/uttt_wasm).  It asks the host for six things: the time
// (its solver runs on a clock), random bytes (for its hash tables), and four
// it never really uses.  Loaded by the engine worker (where Python reaches
// the exports as `js.utttWasm`) and by the solver worker, each its own copy.
async function instantiateUttt() {
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
  const bytes = await (await fetch("uttt_wasm.wasm")).arrayBuffer();
  const { instance } = await WebAssembly.instantiate(bytes, { wasi_snapshot_preview1: wasi });
  memory = instance.exports.memory;
  return instance.exports;
}
