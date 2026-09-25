"""``uttt_rs`` for the web build: the native core, compiled to WebAssembly.

On the desktop, ``alphazero_uttt/uttt_rs.pyd`` is the Rust core wrapped in
PyO3.  Pyodide cannot load a native extension, so the same Rust source is
compiled to WASM (``webapp/uttt_wasm``) and the engine worker exposes its
exports as ``utttWasm``.  This module puts the ``uttt_rs`` interface back on
top of them -- same class, same methods, same return types -- so
``alphazero_uttt.rs_agent`` runs unchanged.

Everything here is synchronous: a WASM call is an ordinary function call.
Inputs are copied into the module's memory, results read back out of its
shared output buffer (see ``webapp/uttt_wasm/src/lib.rs``).
"""

from __future__ import annotations

import struct

import js

_W = js.utttWasm

U32_MAX = 2**32 - 1


def _view(ptr: int, n: int):
    # Re-read ``memory.buffer`` every time: it is replaced when memory grows.
    return js.Uint8Array.new(_W.memory.buffer, ptr, n)


def _out() -> bytes:
    n = int(_W.out_len())
    if n == 0:
        return b""
    return bytes(_view(int(_W.out_ptr()), n).to_py())


class _Buf:
    """Bytes copied into WASM memory for the length of a ``with`` block."""

    def __init__(self, data: bytes):
        self.data = bytes(data)
        self.size = max(1, len(self.data))

    def __enter__(self) -> int:
        self.ptr = int(_W.alloc(self.size))
        if self.data:
            _view(self.ptr, len(self.data)).assign(self.data)
        return self.ptr

    def __exit__(self, *exc):
        _W.dealloc(self.ptr, self.size)
        return False


def _u16s(values) -> bytes:
    values = [int(v) for v in values]
    if len(values) != 9:
        raise ValueError("bad state")
    return struct.pack("<9H", *values)


class Searcher:
    def __init__(self, n_trees: int = 1, c_puct: float = 1.6, fpu_reduction: float = 0.25,
                 solver_empties: int = 18, solver_budget: int = 20000, cache_size: int = 300000,
                 seed: int = 0, variance_c_puct: bool = False, q_init_weight: float = 0.0,
                 draw_value: float = 0.0, forced_playouts: float = 0.0):
        self._h = int(_W.searcher_new(int(n_trees), float(c_puct), float(fpu_reduction),
                                      int(solver_empties), float(solver_budget),
                                      int(cache_size), float(seed), int(bool(variance_c_puct)),
                                      float(q_init_weight), float(draw_value),
                                      float(forced_playouts)))

    def __del__(self):
        try:
            _W.searcher_free(self._h)
        except Exception:  # noqa: BLE001 -- interpreter shutting down
            pass

    def n_trees(self) -> int:
        return int(_W.n_trees(self._h))

    def set_root_moves(self, t: int, moves) -> None:
        data = bytes(int(m) for m in moves)
        with _Buf(data) as p:
            if int(_W.set_root_moves(self._h, int(t), p, len(data))) != 0:
                raise ValueError("illegal move sequence")

    def set_root_state(self, t: int, x, o, active: int, stm: int) -> None:
        with _Buf(_u16s(x)) as px, _Buf(_u16s(o)) as po:
            if int(_W.set_root_state(self._h, int(t), px, po, int(active), int(stm))) != 0:
                raise ValueError("bad state")

    def advance(self, t: int, mv: int) -> None:
        if int(_W.advance(self._h, int(t), int(mv))) != 0:
            raise ValueError("illegal move")

    def collect(self, max_leaves: int, per_tree: int = 1, sims_limit: int = U32_MAX,
                max_collisions: int = 4):
        n = int(_W.collect(self._h, int(max_leaves), int(per_tree),
                           int(min(int(sims_limit), U32_MAX)), int(max_collisions)))
        raw = _out()
        boards = raw[:n * 81]
        ids = list(struct.unpack(f"<{n}I", raw[n * 81:n * 81 + 4 * n])) if n else []
        return boards, ids

    def apply(self, priors: bytes, values: bytes, qs: bytes | None = None) -> None:
        n = int(_W.pending_len(self._h))
        if len(priors) != n * 81 * 4 or len(values) != n * 4:
            raise ValueError(f"expected {n * 81} priors and {n} values for {n} leaves")
        if qs is not None and len(qs) != n * 81 * 4:
            raise ValueError("qs must hold 81 floats per leaf")
        with _Buf(priors) as pp, _Buf(values) as pv:
            if qs is None:
                _W.apply(self._h, pp, pv, 0)
            else:
                with _Buf(qs) as pq:
                    _W.apply(self._h, pp, pv, pq)

    def set_limit(self, t: int, limit: int) -> None:
        _W.set_limit(self._h, int(t), int(min(int(limit), U32_MAX)))

    def root_expanded(self, t: int) -> bool:
        return bool(_W.root_expanded(self._h, int(t)))

    def mix_root_noise(self, t: int, noise, eps: float) -> bool:
        data = struct.pack(f"<{len(noise)}f", *[float(v) for v in noise])
        with _Buf(data) as p:
            return bool(_W.mix_root_noise(self._h, int(t), p, len(noise), float(eps)))

    def root_prior_temperature(self, t: int, temperature: float) -> bool:
        return bool(_W.root_prior_temperature(self._h, int(t), float(temperature)))

    def root_n(self, t: int) -> int:
        return int(_W.root_n(self._h, int(t)))

    def root_value(self, t: int) -> float:
        return float(_W.root_value(self._h, int(t)))

    def root_solved(self, t: int) -> int:
        return int(_W.root_solved(self._h, int(t)))

    def root_children(self, t: int):
        n = int(_W.root_children(self._h, int(t)))
        vals = struct.unpack(f"<{5 * n}d", _out()) if n else ()
        return [(int(vals[i]), int(vals[i + 1]), float(vals[i + 2]), float(vals[i + 3]),
                 int(vals[i + 4])) for i in range(0, 5 * n, 5)]

    def root_canonical(self, t: int) -> bytes:
        _W.root_canonical(self._h, int(t))
        return _out()

    def root_open_empties(self, t: int) -> int:
        return int(_W.root_open_empties(self._h, int(t)))

    def root_solve(self, t: int, budget: int, stop_on_win: bool = False, seconds: float = 0.0):
        n = int(_W.root_solve(self._h, int(t), float(budget), int(bool(stop_on_win)),
                              float(seconds)))
        raw = _out()
        return [(raw[2 * i], struct.unpack("b", raw[2 * i + 1:2 * i + 2])[0]) for i in range(n)]

    def pv(self, t: int, max_len: int):
        n = int(_W.pv(self._h, int(t), int(max_len)))
        return _out()[:n]              # bytes, as PyO3 returns a Vec<u8>

    def node_count(self, t: int) -> int:
        return int(_W.node_count(self._h, int(t)))

    def stats(self):
        _W.stats(self._h)
        return tuple(int(v) for v in struct.unpack("<3d", _out()))

    def clear_cache(self) -> None:
        _W.clear_cache(self._h)

    def set_variance_c_puct(self, on: bool) -> None:
        _W.set_variance_c_puct(self._h, int(bool(on)))

    def set_q_init_weight(self, w: float) -> None:
        _W.set_q_init_weight(self._h, float(w))

    def root_policy_target(self, t: int, k: float):
        n = int(_W.root_policy_target(self._h, int(t), float(k)))
        vals = struct.unpack(f"<{2 * n}d", _out()) if n else ()
        return [(int(vals[i]), float(vals[i + 1])) for i in range(0, 2 * n, 2)]

    def set_draw_value(self, v: float) -> None:
        _W.set_draw_value(self._h, float(v))

    def set_c_puct(self, c: float) -> None:
        _W.set_c_puct(self._h, float(c))

    def set_fpu_reduction(self, f: float) -> None:
        _W.set_fpu_reduction(self._h, float(f))


def legal_moves(moves):
    data = bytes(int(m) for m in moves)
    with _Buf(data) as p:
        n = int(_W.legal_moves(p, len(data)))
    if n < 0:
        return None
    raw = _out()
    return raw[:n], bool(raw[n]), int(raw[n + 1])


def _result(code: int):
    if code == -100:
        raise ValueError("illegal move sequence")
    return None if code == 2 else code


def solve(moves, budget: int):
    data = bytes(int(m) for m in moves)
    with _Buf(data) as p:
        return _result(int(_W.solve(p, len(data), float(budget))))


def solve_state(x, o, active: int, stm: int, budget: int):
    with _Buf(_u16s(x)) as px, _Buf(_u16s(o)) as po:
        return _result(int(_W.solve_state(px, po, int(active), int(stm), float(budget))))


def brute_force_state(x, o, active: int, stm: int) -> int:
    with _Buf(_u16s(x)) as px, _Buf(_u16s(o)) as po:
        code = int(_W.brute_force_state(px, po, int(active), int(stm)))
    if code == -100:
        raise ValueError("bad state")
    return code


def canonical(moves) -> bytes:
    data = bytes(int(m) for m in moves)
    with _Buf(data) as p:
        if int(_W.canonical(p, len(data))) < 0:
            raise ValueError("illegal move sequence")
    return _out()


def bench_playouts(seconds: float, seed: int):
    raise NotImplementedError("not part of the web build")
