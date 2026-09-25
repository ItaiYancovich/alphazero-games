"""A time-controlled search agent: the native core searches, a network evaluates.

The Rust core (:mod:`alphazero_uttt.uttt_rs`) walks the tree, applies the rules,
proves wins and losses and solves endgames exactly; this module only turns a
batch of canonical boards into priors and values and decides when to stop.

Evaluators take ``(N, 81) uint8`` canonical boards (0 playable, 1 own, 2 opp,
3 unreachable) and return ``(priors [N, 81] float32, values [N] float32)``, with
priors masked to the playable cells and values for the side to move.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import torch

from . import uttt_rs
from .agents.base import Agent
from .features import planes_from_boards
from .uttt_game import O, UltimateBoard, X

_PAD_SIZES = (8, 16, 32, 64, 128, 256, 512)


def _masked_softmax(logits: np.ndarray, boards: np.ndarray) -> np.ndarray:
    logits = np.where(boards == 0, logits, -1e9).astype(np.float64)
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


class NetBoardEvaluator:
    """Any network with the ``(policy logits, value)`` forward over feature planes.

    ``backend``: ``torch`` (CPU, traced), ``ov-gpu`` or ``ov-cpu`` (OpenVINO).
    OpenVINO batches are padded to a few fixed sizes so the GPU keeps compiled
    kernels instead of re-planning for every batch size.
    """

    def __init__(self, net: torch.nn.Module, in_planes: int, backend: str = "torch",
                 encode=planes_from_boards, precision: str = "f32", with_q: bool = True):
        self.encode = encode
        self.in_planes = in_planes
        self.backend = backend
        # One evaluator is shared by every agent that wants the same network --
        # compiling it again per game leaked a thread pool and ~60 MB each time
        # (see PLAN_uttt_v3.md).  The OpenVINO infer request below is a single
        # piece of mutable state, so calls through it are serialised; two agents
        # pondering at once would otherwise overwrite each other's input.
        self._infer_lock = threading.Lock()
        net.eval()
        self.net = net
        # A v3 net also predicts each move's value; search can start unvisited
        # moves there instead of at a constant.
        self.has_q = False
        if with_q and hasattr(net, "forward_all"):
            from .v3net import V3Export

            net = V3Export(net).eval()
            self.has_q = True
        if backend == "ov-gpu-f16":
            # Half-precision inference on the GPU: ~35% faster for the v3 hybrid,
            # value error ~0.001 against FP32 (PLAN_uttt_v3.md).
            backend, precision = "ov-gpu", "f16"
            self.backend = backend
        if backend.startswith("ov"):
            import openvino as ov

            example = torch.zeros(8, in_planes, 9, 9)
            model = ov.convert_model(net, example_input=example)
            model.reshape({model.inputs[0]: ov.PartialShape([-1, in_planes, 9, 9])})
            dev = "GPU" if backend == "ov-gpu" else "CPU"
            props = {"PERFORMANCE_HINT": "LATENCY"}
            if dev == "GPU":
                props["INFERENCE_PRECISION_HINT"] = precision
            self.compiled = ov.Core().compile_model(model, dev, props)
            self.request = self.compiled.create_infer_request()
        else:
            with torch.inference_mode():
                traced = torch.jit.trace(net, torch.zeros(8, in_planes, 9, 9))
                self.module = torch.jit.optimize_for_inference(traced)

    @torch.inference_mode()
    def evaluate_boards(self, boards: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = len(boards)
        planes = self.encode(boards.reshape(n, 9, 9), self.in_planes)
        if self.backend.startswith("ov"):
            size = next((s for s in _PAD_SIZES if s >= n), n)
            if size != n:
                planes = np.concatenate([planes, np.zeros((size - n, *planes.shape[1:]), np.float32)])
            with self._infer_lock:
                res = self.request.infer({0: planes})
                logits = np.asarray(res[self.compiled.outputs[0]],
                                    dtype=np.float32).reshape(size, 81)[:n]
                values = np.asarray(res[self.compiled.outputs[1]],
                                    dtype=np.float32).reshape(size)[:n]
                qs = (np.asarray(res[self.compiled.outputs[2]],
                                 dtype=np.float32).reshape(size, 81)[:n]
                      if self.has_q else None)
        else:
            out = self.module(torch.from_numpy(planes))
            logits = out[0].numpy().reshape(n, 81)
            values = out[1].numpy().reshape(n)
            qs = out[2].numpy().reshape(n, 81) if self.has_q else None
        priors = _masked_softmax(logits, boards)
        if qs is None:
            return priors, values.astype(np.float32)
        return priors, values.astype(np.float32), qs


class RsSearchAgent(Agent):
    """PUCT in the native core, stopped by time or by simulations.

    Keeps its tree between moves: after the opponent replies, the subtree under
    that reply becomes the new root, so a move's search starts from everything
    the previous one learned about it.
    """

    def __init__(self, evaluator, time_budget: float | None = 5.0, max_sims: int | None = None,
                 batch: int = 64, c_puct: float = 1.6, fpu_reduction: float = 0.25,
                 solver_empties: int = 18, solver_budget: int = 20_000, reuse_tree: bool = True,
                 name: str | None = None, seed: int = 0, variance_c_puct: bool = False,
                 q_init_weight: float = 0.0, draw_value: float = 0.0, book: str | None = None,
                 extend: float = 1.6, settled_share: float = 0.85, unsettled_share: float = 0.55,
                 root_solve_empties: int = 38, root_solve_share: float = 0.5,
                 practical_eps: float = 0.0):
        self.evaluator = evaluator
        self.time_budget = time_budget
        self.max_sims = max_sims
        self.batch = batch
        self.reuse_tree = reuse_tree
        self.search = uttt_rs.Searcher(1, c_puct, fpu_reduction, solver_empties, solver_budget,
                                       400_000, seed, variance_c_puct, q_init_weight, draw_value)
        self.draw_value = draw_value
        self.book = {}
        if book:
            import json
            from pathlib import Path

            path = Path(book)
            if path.is_file():
                self.book = json.loads(path.read_text())
        # Time management: stop early when one move already owns the search, and
        # keep going (up to ``extend`` x the budget) while the top two are close.
        self.extend = extend
        self.settled_share = settled_share
        self.unsettled_share = unsettled_share
        self.name = name or f"rs-search({time_budget}s)"
        self._after_our_move: UltimateBoard | None = None
        # Exact play from the middle game on (PLAN_uttt_v3.md, rematch).  The
        # in-tree solver only settles leaves with <= 18 open cells; real games
        # become exactly solvable at ~32-36, about fifteen plies earlier.  A
        # separate searcher does the solving so it can never disturb the search
        # tree's reuse, and its table lives for the whole game, so each solve
        # starts from everything the last one proved.  0 turns it off.
        self.root_solve_empties = root_solve_empties
        self.root_solve_share = root_solve_share
        self._solver = uttt_rs.Searcher(1) if root_solve_empties > 0 else None
        # Middle-game practical play: among moves the search scores within
        # ``practical_eps`` of the best, prefer the one leaving the opponent the
        # most confusing position.  0 = off (experimental; see _confusing).
        self.practical_eps = practical_eps
        self.last_info: dict = {}

    def reset(self) -> None:
        self._after_our_move = None
        self.search.clear_cache()

    @staticmethod
    def _masks(b: UltimateBoard):
        return ([b.small(X, i) for i in range(9)], [b.small(O, i) for i in range(9)],
                9 if b.active is None else b.active, 0 if b.to_move == X else 1)

    def _set_root(self, board: UltimateBoard) -> bool:
        prev = self._after_our_move
        if self.reuse_tree and prev is not None and board.move_count == prev.move_count + 1:
            diff = np.flatnonzero(board.array().reshape(-1) != prev.array().reshape(-1))
            if len(diff) == 1 and prev.is_legal(int(diff[0])):
                probe = prev.copy()
                probe.play(int(diff[0]))
                if probe.key() == board.key():
                    self.search.advance(0, int(diff[0]))
                    return True
        self.search.set_root_state(0, *self._masks(board))
        return False

    def _best_share(self) -> float:
        """Share of root visits held by the most-visited move (0 if none yet)."""
        kids = self.search.root_children(0)
        if not kids:
            return 0.0
        visits = sorted((k[1] for k in kids), reverse=True)
        total = sum(visits)
        return visits[0] / total if total else 0.0

    def run(self, board: UltimateBoard, time_budget: float | None, max_sims: int | None,
            extend: float | None = None, stop: threading.Event | None = None) -> dict:
        extend = self.extend if extend is None else extend
        reused = self._set_root(board)
        start_n = self.search.root_n(0)
        t0 = time.perf_counter()
        limit = 2**31 if max_sims is None else start_n + max_sims if not reused else max_sims
        evals = 0
        while True:
            expanded = self.search.node_count(0) > 1
            if expanded and stop is not None and stop.is_set():
                break               # the solver has settled the position
            if expanded and time_budget is not None:
                elapsed = time.perf_counter() - t0
                share = self._best_share()
                if elapsed >= time_budget * extend:
                    break
                if elapsed >= time_budget and share >= self.unsettled_share:
                    break
                # One move already owns the search: the rest of the budget is
                # better spent on the next move (or on pondering).
                if elapsed >= 0.4 * time_budget and share >= self.settled_share:
                    break
            if expanded and (self.search.root_n(0) >= limit or self.search.root_solved(0) != 2):
                break
            raw, _ = self.search.collect(self.batch, self.batch, limit, max(4, self.batch // 4))
            n = len(raw) // 81
            if n == 0:
                # Everything this pass reached was solved, cached or collided:
                # the collect already backed those up, so just go again.
                if self.search.root_solved(0) != 2:
                    break
                continue
            boards = np.frombuffer(raw, dtype=np.uint8).reshape(n, 81)
            out = self.evaluator.evaluate_boards(boards)
            priors, values = out[0], out[1]
            qs = np.ascontiguousarray(out[2], dtype="<f4").tobytes() if len(out) > 2 else None
            self.search.apply(np.ascontiguousarray(priors, dtype="<f4").tobytes(),
                              np.ascontiguousarray(values, dtype="<f4").tobytes(), qs)
            evals += n
        dt = time.perf_counter() - t0
        return dict(sims=self.search.root_n(0), new_sims=self.search.root_n(0) - start_n,
                    evals=evals, seconds=dt, reused=reused, value=self.search.root_value(0),
                    solved=self.search.root_solved(0), nodes=self.search.node_count(0),
                    best_share=round(self._best_share(), 3))

    def choose(self, allowed: set[int] | None = None) -> int:
        kids = self.search.root_children(0)
        if allowed:
            # The solver has settled the position: only moves reaching the
            # proven best value are candidates, and the search picks among them.
            kept = [k for k in kids if k[0] in allowed]
            if not kept:
                return int(min(allowed))
            kids = kept
        won = [k for k in kids if k[4] == 1]
        if won:
            return int(max(won, key=lambda k: k[1])[0])
        alive = [k for k in kids if k[4] != -1] or kids
        best = max(alive, key=lambda k: (k[1], -1e9 if np.isnan(k[2]) else k[2]))
        return int(best[0])

    def exact(self, board: UltimateBoard, seconds: float) -> tuple[int, set[int]] | None:
        """Solve this position outright, within ``seconds``.

        Returns the proven value for the side to move (1, 0, -1) and the moves
        that reach it, or None when the position is not settled in time.
        """
        if self._solver is None or seconds <= 0:
            return None
        self._solver.set_root_state(0, *self._masks(board))
        if self._solver.root_open_empties(0) > self.root_solve_empties:
            return None
        return self._classify(self._solver.root_solve(0, 2**62, False, seconds))

    @staticmethod
    def _classify(res) -> tuple[int, set[int]] | None:
        known = [v for _, v in res if v in (1, 0, -1)]
        if not known or any(v == 2 for _, v in res):
            return None
        best = max(known)
        return best, {int(m) for m, v in res if v == best}

    def _solve_beside(self, board: UltimateBoard, seconds: float, settled: threading.Event,
                      cancel: threading.Event, out: dict) -> None:
        """Keep solving ``board`` while the search runs, in half-second slices.

        The solver is native code with the GIL released and uses the CPU; the
        search mostly waits on the GPU, so the two run side by side and the
        search loses nothing.  (Solving *before* searching, on a share of the
        clock, cost a game: in the band of ~32-38 open cells a solve usually
        cannot finish, and the search was left with half its time -- the new
        bot lost to the old one exactly there.)  Slices so ``cancel`` is
        honoured within half a second; the table keeps what each slice proved.
        """
        t0 = time.perf_counter()
        self._solver.set_root_state(0, *self._masks(board))
        if self._solver.root_open_empties(0) <= self.root_solve_empties:
            deadline = t0 + seconds
            while not cancel.is_set():
                left = deadline - time.perf_counter()
                if left <= 0:
                    break
                got = self._classify(self._solver.root_solve(0, 2**62, False, min(0.5, left)))
                if got is not None:
                    out["result"] = got
                    settled.set()
                    break
        out["seconds"] = time.perf_counter() - t0

    def _practical(self, board: UltimateBoard, solved: tuple[int, set[int]],
                   seconds: float | None) -> tuple[int, dict]:
        """Among moves proven equal, the one hardest for the opponent to answer.

        For each candidate, the solver classifies the opponent's replies into
        those that keep his best result and those that throw it away; the
        network's policy says how natural each reply looks.  The score is the
        policy mass on the replies that throw it away -- roughly, the chance a
        strong player who thinks like the network goes wrong here.  Our friend
        predicted almost every move the bot made, so the network's policy is a
        fair model of him.  Ties, and any candidate the solver cannot classify
        in time, fall back to the search's own preference.
        """
        value, moves = solved
        visits = {int(k[0]): int(k[1]) for k in self.search.root_children(0)}
        cands = sorted(moves, key=lambda m: -visits.get(m, 0))
        deadline = None if seconds is None else time.perf_counter() + seconds
        classes, boards, order = {}, [], []
        for mv in cands:
            left = None if deadline is None else deadline - time.perf_counter()
            if left is not None and left <= 0.05:
                break
            after = board.copy()
            after.play(mv)
            if after.is_terminal():
                continue
            self._solver.set_root_state(0, *self._masks(after))
            can = self._solver.root_canonical(0)
            res = self._solver.root_solve(0, 2**62, False, min(1.0, left or 1.0))
            if any(v == 2 for _, v in res):
                continue                    # not settled in time: no claim either way
            his_best = max(v for _, v in res)
            classes[mv] = {int(r): (v != his_best) for r, v in res}
            boards.append(np.frombuffer(can, dtype=np.uint8))
            order.append(mv)
        traps: dict[int, float] = {}
        if order:
            priors = self.evaluator.evaluate_boards(np.stack(boards))[0]
            for mv, pr in zip(order, priors):
                traps[mv] = float(sum(pr[r] for r, bad in classes[mv].items() if bad))
        best = max(cands, key=lambda m: (round(traps.get(m, 0.0), 2), visits.get(m, 0)))
        from .uttt_game import move_to_str
        return best, {move_to_str(m): round(v, 3) for m, v in sorted(traps.items(),
                                                                      key=lambda kv: -kv[1])}

    def _confusing(self, board: UltimateBoard, chosen: int) -> tuple[int, dict]:
        """Among near-equal moves, the one whose reply position is hardest.

        "Hard" is measured by the network against itself: after our move, its
        policy says which replies look natural and its Q head what each is
        worth.  Where the two disagree -- natural-looking replies that are
        worse than the best one -- a player who thinks like the network is
        likely to go wrong.  The score is the value the opponent is expected to
        give away if he picks as the policy does::

            confusion = max_r Q(r) - sum_r P(r) Q(r)

        Only moves within ``practical_eps`` of the search's best, and with a
        real share of its visits, are candidates, so the cost in our own
        estimate is bounded by ``practical_eps``.
        """
        kids = [k for k in self.search.root_children(0) if not np.isnan(k[2])]
        if not kids:
            return chosen, {}
        top = max(kids, key=lambda k: k[1])
        cands = [k for k in kids
                 if k[1] >= 0.2 * top[1] and k[2] >= top[2] - self.practical_eps]
        if len(cands) < 2 or self._solver is None:
            return chosen, {}
        boards, order = [], []
        for k in cands:
            after = board.copy()
            after.play(int(k[0]))
            if after.is_terminal():
                return chosen, {}
            self._solver.set_root_state(0, *self._masks(after))
            boards.append(np.frombuffer(self._solver.root_canonical(0), dtype=np.uint8))
            order.append(int(k[0]))
        out = self.evaluator.evaluate_boards(np.stack(boards))
        if len(out) < 3:
            return chosen, {}
        priors, qs = out[0], out[2]
        conf = {}
        for i, mv in enumerate(order):
            legal = boards[i] == 0
            q = np.where(legal, qs[i], -np.inf)
            conf[mv] = float(q.max() - float((priors[i] * np.where(legal, qs[i], 0.0)).sum()))
        best = max(order, key=lambda m: conf[m])
        from .uttt_game import move_to_str
        return best, {move_to_str(m): round(v, 3) for m, v in conf.items()}

    def _pick_win(self, board: UltimateBoard, wins: set[int]) -> int:
        """Among proven wins, one that ends the game now if there is one."""
        for mv in sorted(wins):
            probe = board.copy()
            probe.play(mv)
            if probe.is_terminal():
                return mv
        kids = {k[0]: k[1] for k in self.search.root_children(0)}
        return max(sorted(wins), key=lambda m: kids.get(m, 0))

    def book_entry(self, board: UltimateBoard) -> dict | None:
        """The book's entry for this position, moves on the live board; mirror
        images of a stored position count (``alphazero_uttt.book_sym``)."""
        if not self.book:
            return None
        from .book_sym import lookup

        return lookup(self.book, board)

    def book_move(self, board: UltimateBoard) -> int | None:
        """The move a long offline search chose for this position, if there is one."""
        entry = self.book_entry(board)
        if entry is None:
            return None
        move = int(entry["move"])
        return move if board.is_legal(move) else None

    def select_move(self, board: UltimateBoard, last_move: int | None = None) -> int:
        booked = self.book_move(board)
        if booked is not None:
            self.last_info = dict(move=booked, book=True, value=self.book_entry(board)["value"])
            after = board.copy()
            after.play(booked)
            if self.reuse_tree:
                self.search.set_root_state(0, *self._masks(after))
                self._after_our_move = after
            return booked
        t0 = time.perf_counter()
        budget = self.time_budget
        # Most late positions settle in milliseconds: try briefly before any
        # search, so a proven result needs no network at all.
        solved = self.exact(board, min(0.2, 0.05 * budget) if budget else 0.2)
        info = None
        if solved is None and self._solver is not None and budget:
            settled, cancel, out = threading.Event(), threading.Event(), {}
            worker = threading.Thread(target=self._solve_beside, daemon=True,
                                      args=(board.copy(), budget * self.extend, settled,
                                            cancel, out))
            worker.start()
            info = self.run(board, budget, self.max_sims, stop=settled)
            cancel.set()
            worker.join()
            solved = out.get("result")
        spent = time.perf_counter() - t0
        if solved is not None and (solved[0] == 1 or len(solved[1]) == 1):
            # A proven win, or the only move that holds: nothing a search could
            # add.  Keep the tree in step so the next move still reuses it.
            if info is None:
                self._set_root(board)
            move = (self._pick_win(board, solved[1]) if solved[0] == 1
                    else next(iter(solved[1])))
            info = dict(info or {}, sims=self.search.root_n(0), seconds=spent,
                        value=float(solved[0]), solved=solved[0])
        elif solved is not None:
            # Proven, and several moves reach the same result.  A draw is a draw
            # to a solver but not to a person: prefer the one that leaves him the
            # most ways to go wrong.
            left = None if budget is None else max(budget - spent, 0.3 * budget)
            if info is None:
                info = self.run(board, None if left is None else 0.5 * left, self.max_sims,
                                extend=1.0)
            move, traps = self._practical(board, solved,
                                          None if left is None else 0.5 * left)
            info["traps"] = traps
        else:
            if info is None:
                info = self.run(board, budget, self.max_sims)
            move = self.choose()
            if self.practical_eps > 0 and info.get("solved", 2) == 2:
                move, conf = self._confusing(board, move)
                if conf:
                    info["confusion"] = conf
        info["exact"] = None if solved is None else solved[0]
        info["exact_moves"] = None if solved is None else len(solved[1])
        info["exact_seconds"] = round(spent, 3) if solved is not None else None
        info["move"] = move
        info["pv"] = list(self.search.pv(0, 12))
        self.last_info = info
        after = board.copy()
        after.play(move)
        if self.reuse_tree:
            self.search.advance(0, move)
            self._after_our_move = after
        return move


class PonderingAgent(RsSearchAgent):
    """A search agent that keeps thinking while the opponent thinks.

    After it moves, its tree is already rooted at the position the opponent has
    to answer, so a background thread can go on searching exactly the tree the
    next move will use.  The thread is stopped before any other call touches the
    tree, so the native searcher is never used from two threads at once.
    """

    def __init__(self, *args, ponder: bool = True, ponder_limit: float = 120.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.ponder = ponder
        # Thinking on the opponent's time stops by itself after this long, so a
        # game abandoned mid-way does not keep the CPU and GPU busy forever.
        self.ponder_limit = ponder_limit
        self._lock = __import__("threading").Lock()
        self._stop = __import__("threading").Event()
        self._thread = None
        self._solve_thread = None
        self.ponder_sims = 0
        self.ponder_solve_seconds = 0.0

    # How far past the solver's own threshold it is worth solving on the
    # opponent's clock.  An unfinished solve is not wasted: what it proved stays
    # in the table, so the solve on our own turn starts from there.
    PONDER_SOLVE_MARGIN = 4
    PONDER_SOLVE_SLICE = 0.5

    def _ponder_solve_loop(self, board: UltimateBoard) -> None:
        """Solve the opponent's position while they think, in short slices.

        Slices, not one long call, so that ``stop_ponder`` gets the solver back
        within half a second: the solver is one native object and must never be
        entered from two threads.  The position is the one after our move, so
        the table fills with exactly the positions our next move will ask about.
        """
        self._solver.set_root_state(0, *self._masks(board))
        if self._solver.root_open_empties(0) > self.root_solve_empties + self.PONDER_SOLVE_MARGIN:
            return
        t0 = time.perf_counter()
        deadline = t0 + self.ponder_limit
        while not self._stop.is_set() and time.perf_counter() < deadline:
            res = self._solver.root_solve(0, 2**62, False, self.PONDER_SOLVE_SLICE)
            if res and all(v != 2 for _, v in res):
                break                       # settled; nothing more to learn here
        self.ponder_solve_seconds += time.perf_counter() - t0

    def _ponder_loop(self) -> None:
        deadline = time.perf_counter() + self.ponder_limit
        while not self._stop.is_set() and time.perf_counter() < deadline:
            with self._lock:
                if self.search.root_solved(0) != 2 and self.search.root_expanded(0):
                    break
                raw, _ = self.search.collect(self.batch, self.batch, 2**31, max(4, self.batch // 4))
                n = len(raw) // 81
                if n == 0:
                    continue
                boards = np.frombuffer(raw, dtype=np.uint8).reshape(n, 81)
                out = self.evaluator.evaluate_boards(boards)
                qs = np.ascontiguousarray(out[2], dtype="<f4").tobytes() if len(out) > 2 else None
                self.search.apply(np.ascontiguousarray(out[0], dtype="<f4").tobytes(),
                                  np.ascontiguousarray(out[1], dtype="<f4").tobytes(), qs)
                self.ponder_sims += n

    def start_ponder(self) -> None:
        if not self.ponder or self._thread is not None:
            return
        import threading

        self._stop.clear()
        self._thread = threading.Thread(target=self._ponder_loop, daemon=True)
        self._thread.start()
        after = self._after_our_move
        if self._solver is not None and after is not None and not after.is_terminal():
            # The solver runs on the CPU in native code with the GIL released,
            # beside the search thread that mostly waits on the GPU.
            self._solve_thread = threading.Thread(target=self._ponder_solve_loop,
                                                  args=(after.copy(),), daemon=True)
            self._solve_thread.start()

    def stop_ponder(self) -> None:
        if self._thread is None and self._solve_thread is None:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None
        if self._solve_thread is not None:
            # At most one slice to finish; the solver must be free before the
            # caller touches it.
            self._solve_thread.join(timeout=10)
            self._solve_thread = None

    def select_move(self, board: UltimateBoard, last_move: int | None = None) -> int:
        self.stop_ponder()
        with self._lock:
            pondered = self.ponder_sims
            self.ponder_sims = 0
            solved_for = self.ponder_solve_seconds
            self.ponder_solve_seconds = 0.0
            move = super().select_move(board, last_move)
            self.last_info["pondered_evals"] = pondered
            self.last_info["pondered_solve_seconds"] = round(solved_for, 2)
        self.start_ponder()
        return move

    def reset(self) -> None:
        self.stop_ponder()
        super().reset()
