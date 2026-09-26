"""The desktop GUI's game server, run inside the player's own browser.

The page in ``webapp/site`` is the same page ``game_gui.py`` serves; what it
talks to is this module, running in Pyodide in a Web Worker, instead of an
HTTP server.  Everything that decides anything -- the rules, the bots, the
search, the ratings table -- is the project's own Python, imported unchanged.
Three things are different, and all three are here:

* **No PyTorch.**  A small ``torch`` shim (``webapp/py/torch``) covers what the
  evaluators do around a network, and every ``load_checkpoint`` in the project
  is pointed at :class:`WebNet`, which runs the exported ONNX model through
  onnxruntime in a second worker.  The call is synchronous from Python's point
  of view -- the worker waits on shared memory for the answer -- which is what
  lets the existing agents, searches and evaluators run as they are.
* **No threads.**  Pyodide cannot start one.  The session's bot loop, the live
  analysis and the match review each become a *step* the page's worker calls
  when it is idle: :meth:`WebSession.bot_think`, :meth:`WebAnalysis.tick` and
  :meth:`WebReview.tick`.
* **No HTTP.**  :func:`dispatch` answers the same paths with the same JSON the
  desktop server's handler does, so the page's ``api()`` only has to change
  where it sends a request, not what it sends.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np

# --------------------------------------------------------------- the network
_infer = None           # set by the worker: (model, planes) -> list of arrays
_infer_start = None     # (model, planes) -> wait() -> list of arrays, where the
                        # network can answer while Python goes on (see WebNet.start)
# Whether to use that.  Off: it halves the batches (see ``run_search``), and
# on a phone, where a network call costs about the same for 5 positions as for
# 12, twice the calls cost more than hiding the search behind the network
# saves -- measured on an 8-core Android phone, where the bots got slower.  On
# a desktop it made Connect Four 1.2x faster and the rest no faster.
BACKGROUND = False
_MODELS: dict = {}      # models.json, keyed by model name
_BY_PATH: dict = {}     # checkpoint path (as the adapters spell it) -> model name


def set_infer(js_fn, js_start=None, js_finish=None) -> None:
    """Install the worker's synchronous ``(model, data, dims) -> [{data, dims}]``
    and, when given, its two halves: ``js_start`` posts the batch to the network
    and returns at once, ``js_finish`` waits for that batch's answer."""
    global _infer, _infer_start
    from pyodide.ffi import to_js

    def unpack(result) -> list[np.ndarray]:
        outs = []
        for item in result:
            data = np.frombuffer(item.data.to_py(), dtype=np.float32).copy()
            outs.append(data.reshape([int(d) for d in item.dims.to_py()]))
        return outs

    def call(name: str, planes: np.ndarray) -> list[np.ndarray]:
        return unpack(js_fn(name, to_js(planes.reshape(-1)), to_js([int(d) for d in planes.shape])))

    def begin(name: str, planes: np.ndarray):
        js_start(name, to_js(planes.reshape(-1)), to_js([int(d) for d in planes.shape]))
        return lambda: unpack(js_finish())

    def run(name: str, planes: np.ndarray) -> list[np.ndarray]:
        planes = np.ascontiguousarray(planes, dtype=np.float32)
        return CACHE.run(name, planes, lambda some: call(name, some))

    def run_start(name: str, planes: np.ndarray):
        planes = np.ascontiguousarray(planes, dtype=np.float32)
        return CACHE.start(name, planes, lambda some: begin(name, some),
                           lambda some: call(name, some))

    _infer = run
    _infer_start = run_start if js_start is not None else None


class EvalCache:
    """The network's answers, by position, for positions the search meets again.

    A search reaches the same position by different move orders, and a bot's
    next search walks much of the tree its last one did: in Connect Four
    against a person, over half the positions a bot asks about are ones it has
    asked about before.  A network's answer for a position depends on that
    position alone, so answers are kept, keyed by a digest of the input planes
    -- which works for every network without knowing its game -- and those
    positions skip the network.  The oldest answers go first once the cache is
    full.
    """

    def __init__(self, floats: int = 4_000_000):
        from collections import OrderedDict

        self.limit = floats          # numbers kept, 4 bytes each: 16 MB
        self.held = 0
        self.entries: OrderedDict = OrderedDict()
        self.whole: set[str] = set()  # networks whose outputs are not one row a position
        self.enabled = True
        self.rows = self.hits = 0

    def run(self, name: str, planes: np.ndarray, infer) -> list[np.ndarray]:
        """``infer(planes)``, from the cache where it can be."""
        def now(some):
            out = infer(some)
            return lambda: out
        return self.start(name, planes, now, infer)()

    def start(self, name: str, planes: np.ndarray, begin, infer):
        """The same, for a network that answers in the background: ``begin``
        hands it the positions not held and returns a wait; so does this."""
        if not self.enabled or name in self.whole or len(planes) == 0:
            return begin(planes)
        import hashlib

        n = len(planes)
        rows = planes.reshape(n, -1)
        keys = [(name, hashlib.blake2b(rows[i], digest_size=16).digest()) for i in range(n)]
        found = [self.entries.get(k) for k in keys]
        self.rows += n
        # Each position not held goes to the network once, even if the batch
        # has it twice (two move orders reaching it in the same batch).
        wanted: dict = {}
        for i, got in enumerate(found):
            if got is None:
                wanted.setdefault(keys[i], i)
            else:
                self.hits += 1
                self.entries.move_to_end(keys[i])
        wait = begin(planes[list(wanted.values())] if len(wanted) < n else planes) if wanted else None

        def finish() -> list[np.ndarray]:
            if wait is not None:
                outs = wait()
                if any(o.ndim == 0 or o.shape[0] != len(wanted) for o in outs):
                    self.whole.add(name)
                    return outs if len(wanted) == n else infer(planes)
                for j, key in enumerate(wanted):
                    answer = tuple(o[j].copy() for o in outs)
                    self._keep(key, answer)
                    for i in range(n):
                        if found[i] is None and keys[i] == key:
                            found[i] = answer
            return [np.stack([row[j] for row in found]) for j in range(len(found[0]))]

        return finish

    def _keep(self, key, answer) -> None:
        self.entries[key] = answer
        self.held += sum(a.size for a in answer)
        while self.held > self.limit and self.entries:
            _, old = self.entries.popitem(last=False)
            self.held -= sum(a.size for a in old)


CACHE = EvalCache()


class WebNet:
    """A trained network, as far as any caller in the project can tell.

    ``cfg`` answers the same fields the checkpoint's config did, calling it
    returns the same outputs ``forward`` did (as shim tensors), and backgammon's
    ``equities`` is its own output of the exported graph.
    """

    def __init__(self, name: str):
        self.name = name
        meta = _MODELS[name]
        cfg = dict(meta.get("cfg") or {})
        self.cfg = SimpleNamespace(**cfg, to_dict=lambda: dict(cfg))
        self.outputs = list(meta["outputs"])
        self.training = False

    def _run(self, x):
        return self._named(_infer(self.name, self._planes(x)))

    @staticmethod
    def _planes(x) -> np.ndarray:
        return np.ascontiguousarray(np.asarray(x.a if hasattr(x, "a") else x), dtype=np.float32)

    def _named(self, outs) -> dict:
        import torch

        return {n: torch.Tensor(o) for n, o in zip(self.outputs, outs)}

    def _heads(self, out: dict):
        heads = [out[n] for n in self.outputs if n != "equities"]
        return heads[0] if len(heads) == 1 else tuple(heads)

    def __call__(self, x):
        return self._heads(self._run(x))

    @property
    def background(self) -> bool:
        """Can this network answer while the search goes on?  See ``start``."""
        return BACKGROUND and _infer_start is not None

    def start(self, x):
        """``__call__`` in two halves: the batch goes to the network workers
        now, and the returned function waits for what ``__call__`` would have
        returned -- so the search can collect its next batch meanwhile."""
        wait = _infer_start(self.name, self._planes(x))
        return lambda: self._heads(self._named(wait()))

    forward = __call__

    def equities(self, x):
        return self._run(x)["equities"]

    # The module protocol, for callers that move or freeze a network.
    def eval(self):
        return self

    def train(self, *_a):
        return self

    def to(self, *_a, **_k):
        return self

    def parameters(self):
        return iter(())


def _web_load(path, *_a, **_k):
    """Every ``load_checkpoint`` in the project, pointed at the ONNX models."""
    key = os.path.normpath(str(path))
    name = _BY_PATH.get(key)
    if name is None:
        raise FileNotFoundError(f"{Path(path).name} is not part of the web build")
    return WebNet(name), {}


def _patch_loaders() -> None:
    for modname, mod in list(sys.modules.items()):
        if not modname.startswith("alphazero_") or mod is None:
            continue
        for attr in ("load_checkpoint", "load_any"):
            if hasattr(mod, attr):
                setattr(mod, attr, _web_load)


# ---------------------------------------------------------------- start-up
g = None                 # game_gui, once imported
SESSION = None


def boot(root: str, models_json: str, placements_json: str) -> str:
    """Import the engine and wire it to the browser.  Returns the game list.

    ``placements`` maps each model to the checkpoint path it stands in for,
    relative to the project root; an empty file is created there so that the
    adapters' own checkpoint discovery finds exactly the shipped networks.
    """
    global g, SESSION
    _MODELS.update(json.loads(models_json))
    for name, rel in json.loads(placements_json).items():
        full = Path(root) / rel
        full.parent.mkdir(parents=True, exist_ok=True)
        if not full.exists():
            full.write_bytes(b"")
        _BY_PATH[os.path.normpath(str(full))] = name

    # Import every network module first, then point their loaders at ONNX, so
    # that a registry imported later binds the web loader, not the real one.
    import alphazero_core.net  # noqa: F401
    import alphazero_hex.net_v3  # noqa: F401
    import alphazero_bg.net  # noqa: F401
    import alphazero_splendor3.net  # noqa: F401
    for pkg in ("alphazero_c4", "alphazero_uttt", "alphazero_rps2", "alphazero_hex"):
        try:
            __import__(f"{pkg}.net")
        except Exception:
            traceback.print_exc()
    _patch_loaders()

    import game_gui
    g = game_gui
    g.DEVICE = "cpu"
    _patch_loaders()
    _adapt_games()
    SESSION = WebSession(g.DEFAULT_GAME)
    g.SESSION = SESSION
    return json.dumps(sorted(g.GAMES))


def _adapt_games() -> None:
    """The few places a game's menu must not offer what the browser lacks."""
    sp = g.GAMES["splendor"]
    cls = type(sp)
    base_catalog = cls.catalog
    # v1 and v2 of the Splendor network are not shipped; v3 beats both.
    cls.catalog = lambda self: [b for b in base_catalog(self) if b["id"] not in ("az", "az2")]
    cls.default_opponent = property(lambda self: "az3" if self.v3_checkpoint() else "planner")
    cls.v2_checkpoint = lambda self: None
    cls.checkpoints = lambda self: (["v3"] if self.v3_checkpoint() else [])
    cls.analysis_checkpoint = lambda self: ("v3" if self.v3_checkpoint() else "")
    # Its live evaluation is built on the v1/v2 networks.
    cls.analysable = False

    # Ultimate Tic-Tac-Toe's v3 match bot: the native core is the WASM build
    # (``alphazero_uttt/uttt_rs.py`` here), the network is ONNX.
    if "uttt_v3" in _BY_PATH.values():
        g._v3_evaluator = _v3_evaluator
        _install_v3_agent()
        _judge_uttt_with_v3()


# --------------------------------------------------------- the v3 match bot
_V3_EVALUATORS: dict = {}


class WebBoardEvaluator:
    """``rs_agent.NetBoardEvaluator`` for the browser: same inputs, same outputs.

    Canonical ``(N, 81)`` boards in; priors masked to the playable cells,
    values for the side to move, and the per-move Q the v3 network predicts.
    """

    has_q = True

    def __init__(self, name: str):
        from alphazero_uttt.features import planes_from_boards

        self.name = name
        self.encode = planes_from_boards
        self.in_planes = int(_MODELS[name]["cfg"].get("in_planes", 23))

    def evaluate_boards(self, boards: np.ndarray):
        return self._answer(boards, _infer(self.name, self._planes(boards)))

    def evaluate_boards_start(self, boards: np.ndarray):
        """``evaluate_boards`` in two halves, as ``WebNet.start``."""
        wait = _infer_start(self.name, self._planes(boards))
        return lambda: self._answer(boards, wait())

    def _planes(self, boards: np.ndarray) -> np.ndarray:
        n = len(boards)
        planes = self.encode(np.asarray(boards).reshape(n, 9, 9), self.in_planes)
        return np.asarray(planes, dtype=np.float32)

    def _answer(self, boards: np.ndarray, outs):
        from alphazero_uttt.rs_agent import _masked_softmax

        n = len(boards)
        logits, values, qs = outs
        priors = _masked_softmax(logits.reshape(n, 81), np.asarray(boards).reshape(n, 81))
        return priors, values.reshape(n).astype(np.float32), qs.reshape(n, 81).astype(np.float32)


class V3Judge:
    """The v3 network as the PUCT search's evaluator, for analysis and review.

    The v3 match bot's network is the strongest Ultimate network there is, so
    it judges positions -- through the same PUCT search as before, which only
    needs ``evaluate(states) -> (priors, values)``.  Positions go in as the
    canonical boards both sides share (0 playable, 1 own, 2 opponent, 3 out of
    reach); values come back for the side to move, as the search wants them.
    """

    def __init__(self, boards: WebBoardEvaluator):
        self.boards = boards

    @staticmethod
    def _canonical(states) -> np.ndarray:
        return np.stack([s.canonical_board().reshape(-1) for s in states])

    def evaluate(self, states):
        priors, values, _ = self.boards.evaluate_boards(self._canonical(states))
        return priors, values

    @property
    def background(self) -> bool:
        return BACKGROUND and _infer_start is not None

    def evaluate_start(self, states):
        wait = self.boards.evaluate_boards_start(self._canonical(states))
        return lambda: wait()[:2]


def _judge_uttt_with_v3() -> None:
    """Ultimate's analysis and match review on the v3 network.

    ``evaluator`` is what both call, and nothing else: the bots build their
    own, and which network the ratings belong to (``analysis_checkpoint``) is
    left as it is.  The page names the judge from ``judge`` in the game's blob.
    """
    cls = type(g.GAMES["uttt"])
    cls.evaluator = lambda self, ckpt, board: V3Judge(_v3_evaluator("", ""))
    blob = g.game_blob

    def game_blob(adapter):
        out = blob(adapter)
        if adapter.key == "uttt":
            out["judge"] = "the v3 network"
        return out

    g.game_blob = game_blob


def _v3_evaluator(ckpt: str, backend: str):
    """``game_gui._v3_evaluator``, answered by the shipped ONNX network."""
    if "uttt_v3" not in _V3_EVALUATORS:
        _V3_EVALUATORS["uttt_v3"] = WebBoardEvaluator("uttt_v3")
    return _V3_EVALUATORS["uttt_v3"]


# How long each slice of the exact solver runs between search batches.  The
# desktop runs the solver on its own thread beside the search; one thread has
# to take turns, and short turns keep both moving.
SOLVE_SLICE = 0.08


def _install_v3_agent() -> None:
    """Swap ``rs_agent.PonderingAgent`` for a version that needs no threads.

    Two things in the agent start threads: solving the position beside the
    search on the agent's own move, and searching on the opponent's time.
    Both keep their logic -- what is searched, what is solved, when to stop --
    and change only *how* they share the one thread there is:

    * The solver runs in short slices between the search's batches instead of
      alongside them (``run`` below).  ``select_move`` itself is unchanged: the
      thread it asks for is recorded rather than started, and ``run`` serves it.
    * Pondering is a step the page's worker takes while the human thinks
      (``ponder_tick``, driven by :func:`work`), rather than a loop on a thread.
    """
    import threading as real

    from alphazero_uttt import rs_agent

    class _Deferred:
        """A thread that is not started: its work is taken up by ``run``."""

        def __init__(self, target=None, args=(), kwargs=None, daemon=None):
            self.target, self.args = target, args

        def start(self):
            agent = getattr(self.target, "__self__", None)
            if agent is not None and getattr(self.target, "__name__", "") == "_solve_beside":
                board, seconds, settled, cancel, out = self.args
                agent._beside = dict(board=board, seconds=seconds, settled=settled,
                                     cancel=cancel, out=out, init=False, done=False)
            else:
                self.target(*self.args)

        def join(self, timeout=None):
            agent = getattr(self.target, "__self__", None)
            beside = getattr(agent, "_beside", None)
            if beside is not None:
                beside["out"]["seconds"] = beside.get("spent", 0.0)
                agent._beside = None

    rs_agent.threading = SimpleNamespace(Thread=_Deferred, Event=real.Event, Lock=real.Lock)

    base = rs_agent.PonderingAgent

    class WebPonderingAgent(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Smaller batches than the desktop's GPU wants when the network
            # runs on the CPU, so a request never waits long; the desktop's
            # own once the device's tuning has put it on the GPU.
            self._gpu_batch = self.batch
            self.batch = min(self.batch, 48)
            self._beside = None
            self._pondering = False
            self._ponder_until = 0.0
            self._ponder_solve = None

        # ----------------------------------------- the solver, in slices
        def _serve_beside(self) -> None:
            beside = self._beside
            if beside is None or beside["done"]:
                return
            if not beside["init"]:
                beside["init"] = True
                beside["spent"] = 0.0
                beside["deadline"] = time.perf_counter() + beside["seconds"]
                self._solver.set_root_state(0, *self._masks(beside["board"]))
                if self._solver.root_open_empties(0) > self.root_solve_empties:
                    beside["done"] = True
                    return
            left = beside["deadline"] - time.perf_counter()
            if left <= 0 or beside["cancel"].is_set():
                beside["done"] = True
                return
            t0 = time.perf_counter()
            got = self._classify(self._solver.root_solve(0, 2**62, False, min(SOLVE_SLICE, left)))
            beside["spent"] += time.perf_counter() - t0
            if got is not None:
                beside["out"]["result"] = got
                beside["settled"].set()
                beside["done"] = True

        def _choose_batch(self) -> None:
            """48 positions a batch on the CPU; the desktop's 128 on a GPU."""
            import js

            on_gpu = getattr(js, "gpuBest", None)
            self.batch = self._gpu_batch if on_gpu is not None and on_gpu("uttt_v3") \
                else min(self._gpu_batch, 48)

        def run(self, board, time_budget, max_sims, extend=None, stop=None):
            """``RsSearchAgent.run``, taking turns with the solver between batches."""
            self._choose_batch()
            extend = self.extend if extend is None else extend
            reused = self._set_root(board)
            start_n = self.search.root_n(0)
            t0 = time.perf_counter()
            limit = 2**31 if max_sims is None else start_n + max_sims if not reused else max_sims
            evals = 0
            while True:
                self._serve_beside()
                expanded = self.search.node_count(0) > 1
                if expanded and stop is not None and stop.is_set():
                    break
                if expanded and time_budget is not None:
                    elapsed = time.perf_counter() - t0
                    share = self._best_share()
                    if elapsed >= time_budget * extend:
                        break
                    if elapsed >= time_budget and share >= self.unsettled_share:
                        break
                    if elapsed >= 0.4 * time_budget and share >= self.settled_share:
                        break
                if expanded and (self.search.root_n(0) >= limit or self.search.root_solved(0) != 2):
                    break
                raw, _ = self.search.collect(self.batch, self.batch, limit, max(4, self.batch // 4))
                n = len(raw) // 81
                if n == 0:
                    if self.search.root_solved(0) != 2:
                        break
                    continue
                boards = np.frombuffer(raw, dtype=np.uint8).reshape(n, 81)
                out = self.evaluator.evaluate_boards(boards)
                qs = np.ascontiguousarray(out[2], dtype="<f4").tobytes() if len(out) > 2 else None
                self.search.apply(np.ascontiguousarray(out[0], dtype="<f4").tobytes(),
                                  np.ascontiguousarray(out[1], dtype="<f4").tobytes(), qs)
                evals += n
            dt = time.perf_counter() - t0
            return dict(sims=self.search.root_n(0), new_sims=self.search.root_n(0) - start_n,
                        evals=evals, seconds=dt, reused=reused, value=self.search.root_value(0),
                        solved=self.search.root_solved(0), nodes=self.search.node_count(0),
                        best_share=round(self._best_share(), 3))

        # ------------------------------------ thinking on the human's time
        def start_ponder(self) -> None:
            if not self.ponder:
                return
            self._pondering = True
            self._ponder_until = time.perf_counter() + self.ponder_limit
            after = self._after_our_move
            self._ponder_solve = (after.copy() if self._solver is not None and after is not None
                                  and not after.is_terminal() else None)
            self._ponder_solve_ready = False

        def stop_ponder(self) -> None:
            self._pondering = False
            self._ponder_solve = None

        def pondering(self) -> bool:
            if self._pondering and time.perf_counter() > self._ponder_until:
                self._pondering = False
            return self._pondering

        def ponder_tick(self) -> None:
            """One batch of search on the tree the next move will use, then a
            slice of the solver on the position the human has to answer."""
            self._choose_batch()
            if not self.pondering():
                return
            with self._lock:
                if self.search.root_solved(0) != 2 and self.search.root_expanded(0):
                    self._pondering = False
                    return
                raw, _ = self.search.collect(self.batch, self.batch, 2**31, max(4, self.batch // 4))
                n = len(raw) // 81
                if n:
                    boards = np.frombuffer(raw, dtype=np.uint8).reshape(n, 81)
                    out = self.evaluator.evaluate_boards(boards)
                    qs = np.ascontiguousarray(out[2], dtype="<f4").tobytes() if len(out) > 2 else None
                    self.search.apply(np.ascontiguousarray(out[0], dtype="<f4").tobytes(),
                                      np.ascontiguousarray(out[1], dtype="<f4").tobytes(), qs)
                    self.ponder_sims += n
                board = self._ponder_solve
                if board is not None:
                    if not self._ponder_solve_ready:
                        self._ponder_solve_ready = True
                        self._solver.set_root_state(0, *self._masks(board))
                        if (self._solver.root_open_empties(0)
                                > self.root_solve_empties + self.PONDER_SOLVE_MARGIN):
                            self._ponder_solve = None
                            return
                    t0 = time.perf_counter()
                    res = self._solver.root_solve(0, 2**62, False, SOLVE_SLICE)
                    self.ponder_solve_seconds += time.perf_counter() - t0
                    if res and all(v != 2 for _, v in res):
                        self._ponder_solve = None

    rs_agent.PonderingAgent = WebPonderingAgent


# ----------------------------------------------------------------- analysis
class WebAnalysis:
    """The desktop ``Analysis``, advanced one step at a time by the page."""

    def __new__(cls):
        base = g.Analysis

        class _Web(base):
            def configure(self, enabled: bool, max_sims: int) -> None:
                with self.lock:
                    was_off = not self.enabled
                    self.enabled = enabled
                    self.max_sims = max(1, int(max_sims))
                    if was_off and enabled:
                        self._result = {}

            def busy(self) -> bool:
                """Is there anything for :meth:`tick` to do?"""
                with self.lock:
                    enabled, wanted = self.enabled, self._wanted
                if not enabled or wanted is None:
                    return False
                board, key, ckpt, game = wanted
                if not game.analysable:
                    return self._key is not None
                if not game.uses_search:
                    return key != self._key or getattr(self, "_stream", None) is not None
                if key != self._key or self._search is None:
                    return True
                return self._search.sims_done < self.max_sims

            def tick(self) -> None:
                """One pass of the desktop worker loop, without its sleeps."""
                with self.lock:
                    enabled, ceiling, wanted = self.enabled, self.max_sims, self._wanted
                if not enabled or wanted is None:
                    return
                board, key, ckpt, game = wanted
                try:
                    if not game.analysable:
                        if self._key is not None:
                            self._key = None
                            self._search = None
                            self._store({})
                        return
                    if not game.uses_search:
                        if key != self._key:
                            self._key = key
                            self._board = board.copy()
                            self._game = game
                            self._search = None
                            self._stream = game.analysis_stream(board, ckpt)
                        stream = getattr(self, "_stream", None)
                        if stream is None:
                            return
                        try:
                            self._store(dict(next(stream), error=None))
                        except StopIteration:
                            self._stream = None
                        return
                    if key != self._key:
                        self._start(game, board, key, ceiling)
                    if self._search is None:
                        return
                    self._search.cfg.simulations = ceiling
                    if self._search.sims_done >= ceiling:
                        return
                    self._step(game, board, key, ckpt)
                except Exception as exc:  # noqa: BLE001 -- shown on the panel
                    traceback.print_exc()
                    self._publish_error(f"analysis stopped: {exc}")
                    with self.lock:
                        self.enabled = False

        return _Web()


# ------------------------------------------------------------------- review
class WebReview:
    """The desktop ``MatchReview``, one position per :meth:`tick`."""

    def __new__(cls):
        base = g.MatchReview

        class _Web(base):
            def __init__(self):
                super().__init__()
                self._job = None

            def start(self, game, moves, size_id, ckpt, sims):
                if not game.reviewable:
                    return f"match review is not available for {game.label}"
                if len(moves) < 2:
                    return "nothing to analyse yet"
                with self.lock:
                    if self._state["running"]:
                        return "already analysing this game"
                    self._token += 1
                    token = self._token
                    self._state = dict(self._blank(), running=True, total=len(moves) + 1,
                                       sims=int(sims), ckpt=ckpt, workers=1)
                self._job = (token, self._serial(token, game, list(moves), size_id,
                                                 ckpt, int(sims)), list(moves))
                return None

            def clear(self):
                super().clear()
                self._job = None

            def busy(self) -> bool:
                return self._job is not None

            def _serial(self, token, game, moves, size_id, ckpt, sims):
                """``_run_serial`` from the desktop, pausing after each position."""
                from alphazero_core.mcts import run_search

                board = game.new_board(size_id)
                engine = game.evaluator(ckpt, board)
                cfg = g.MCTSConfig(simulations=sims, c_puct=1.6, add_noise=False)
                search = None
                previous_board = None
                plies = []
                for index in range(len(moves) + 1):
                    if board.is_terminal():
                        entry = game.terminal_readout(board)
                        search = None
                    else:
                        search = g.reuse_or_new(search, previous_board, board, cfg, sims)
                        search.cfg.simulations = sims
                        run_search(search, engine, sims, self.BATCH)
                        entry = game.readout(search, board)
                        previous_board = board.copy()
                    entry["ply"] = index
                    plies.append(entry)
                    self._advance(token)
                    if index < len(moves):
                        board.play(moves[index])
                    yield None
                return plies

            def tick(self) -> None:
                if self._job is None:
                    return
                token, job, moves = self._job
                if not self._alive(token):
                    self._job = None
                    return
                try:
                    next(job)
                except StopIteration as done:
                    self._job = None
                    plies = done.value
                    for index, entry in enumerate(plies):
                        entry["played"] = int(moves[index]) if index < len(moves) else None
                    with self.lock:
                        if token != self._token:
                            return
                        self._state["plies"] = plies
                    self._finish(token)
                except Exception as exc:  # noqa: BLE001 -- shown in the banner
                    traceback.print_exc()
                    self._job = None
                    self._fail(token, f"analysis failed: {exc}")

        return _Web()


# ------------------------------------------------------------------ session
def WebSession(game: str):  # noqa: N802 -- a class, built once game_gui exists
    base = g.Session

    class _Web(base):
        """The desktop session, with the bot loop turned inside out.

        ``_wake_bot`` only raises the flag; the page's worker then calls
        :meth:`bot_think` (search, which may take seconds) and, after waiting
        out the pace, :meth:`bot_commit` (play it).  Everything between those
        calls -- a new game, a take-back, a changed opponent -- bumps
        ``generation`` exactly as it does on the desktop, and a stale move is
        dropped exactly as it is there.
        """

        def __init__(self, game_key):
            super().__init__(game_key)
            self.analysis = WebAnalysis()
            self.review = WebReview()
            self._bot_gen = None
            self._planned = None
            # Online play: seats played from another browser.  They count as
            # humans here; the page decides whose clicks may move them.
            self.remote_seats: set[int] = set()

        def _wake_bot(self) -> None:
            with self.lock:
                if self.thinking or self.board.is_terminal():
                    return
                if self.agents[self.board.to_move] is None:
                    return
                self.thinking = True
                self._bot_gen = self.generation
                self._planned = None
                self._touch()

        def bot_pending(self) -> bool:
            with self.lock:
                return (self.thinking and self._bot_gen == self.generation
                        and not self.board.is_terminal()
                        and self.agents.get(self.board.to_move) is not None)

        def bot_think(self) -> float:
            """Choose the bot's move.  Returns how long to hold it before playing."""
            with self.lock:
                gen = self.generation
                if not self.bot_pending():
                    self._settle(gen)
                    return -1.0
                agent = self.agents[self.board.to_move]
                position = self.board.copy()
                last = self.moves[-1].get("cell") if self.moves else None
                game = self.game
            started = time.time()
            try:
                cell = game.select(agent, position, last)
            except Exception:
                traceback.print_exc()
                with self.lock:
                    if gen == self.generation:
                        self.error = "the bot crashed while thinking"
                        self.thinking = False
                        self._touch()
                return -1.0
            self._planned = (gen, cell, agent)
            return max(0.0, game.bot_pause * self.pace - (time.time() - started))

        def bot_commit(self) -> None:
            planned, self._planned = self._planned, None
            if planned is None:
                return
            gen, cell, agent = planned
            with self.lock:
                if gen != self.generation:
                    return
                self._record(cell, agent.name)
                self.info = self._agent_info(agent)
                # Still a bot on turn: stay "thinking" for the next one.
                if self.board.is_terminal() or self.agents.get(self.board.to_move) is None:
                    self.thinking = False
                self._bot_gen = self.generation
                self._touch()

        def _settle(self, gen) -> None:
            if gen == self.generation and self.thinking:
                self.thinking = False
                self._touch()

        def set_remote(self, seats) -> None:
            with self.lock:
                self.remote_seats = {int(s) for s in seats}
                self._touch()

        def _agent_info(self, agent) -> dict:
            """The desktop's readout, plus the v3 bot's search in numbers."""
            last = getattr(agent, "last_info", None)
            if not isinstance(last, dict) or not last:
                return super()._agent_info(agent)
            return v3_readout(last, self.game)

        def snapshot(self) -> dict:
            out = super().snapshot()
            out["remote_seats"] = sorted(self.remote_seats)
            return out

    return _Web(game)


def v3_readout(info: dict, game) -> dict[str, str]:
    """What a native-search move cost, from the agent's ``last_info``.

    The tree is kept between moves, so the root it decided at already held
    the visits under the move the human actually played -- from the bot's
    earlier searches and from thinking while the human thought.  "Total" is
    all of it; "carried over" is that part; "on its turn" is what it added
    after the human moved, and the speed is measured over that alone.
    """
    out: dict[str, str] = {}
    if info.get("book"):
        out["move from"] = "opening book (no search)"
        if info.get("value") is not None:
            out["book's estimate"] = game.format_value(float(info["value"]))
        return out
    total = info.get("sims")
    new = info.get("new_sims")
    seconds = info.get("seconds")
    if total is not None:
        out["simulations, total"] = f"{int(total):,}"
    if total is not None and new is not None:
        out["carried over from your turn"] = f"{max(0, int(total) - int(new)):,}"
    if new is not None:
        spent = f" in {float(seconds):.1f}s" if seconds else ""
        out["searched on its turn"] = f"{int(new):,}{spent}"
        if seconds and float(seconds) > 0:
            out["speed on its turn"] = f"{int(new) / float(seconds):,.0f} sims/s"
    exact = info.get("exact")
    if exact is not None:
        out["exact solver"] = {1: "proven win", 0: "proven draw", -1: "proven loss"}.get(
            int(exact), str(exact))
    value = info.get("value")
    if value is not None and exact is None:
        out["its own estimate"] = game.format_value(float(value))
    return out


# ---------------------------------------------------------------- dispatch
def _bots_payload() -> dict:
    current = SESSION.game
    return dict(
        games=[g.game_blob(g.GAMES[k])
               for k in ("hex", "c4", "uttt", "uxx", "rps2", "bg", "splendor")],
        game=current.key,
        bots=current.catalog(), checkpoints=current.checkpoints(),
        sizes=current.sizes,
        sim_stops=g.SIM_STOPS, eval_stops=g.EVAL_STOPS[:6],
        review_stops=[s for s in g.REVIEW_STOPS if s <= 800] or g.REVIEW_STOPS,
        default_review_sims=min(g.DEFAULT_REVIEW_SIMS, 400),
        analysis_ckpt=current.analysis_checkpoint(),
        default_sims=g.DEFAULT_SIMS, default_eval_sims=g.DEFAULT_EVAL_SIMS,
        users=sorted(current.R.load_users()),
        provisional_games=current.R.PROVISIONAL_GAMES,
        web=True)


def dispatch(path: str, body_json: str | None) -> str:
    """One request, answered as the desktop handler answers it."""
    try:
        status, payload = _dispatch(path, json.loads(body_json) if body_json else None)
    except Exception as exc:  # noqa: BLE001 -- the page shows it in the banner
        traceback.print_exc()
        status, payload = 500, dict(error=f"the game engine failed -- "
                                           f"{type(exc).__name__}: {exc}")
    return json.dumps(dict(status=status, body=payload))


def _with(payload: dict, problem: str | None) -> dict:
    return dict(payload, problem=problem) if problem else payload


def _dispatch(path: str, body: dict | None):
    S = SESSION
    if body is None:
        if path == "/api/bots":
            return 200, _bots_payload()
        if path == "/api/state":
            return 200, S.snapshot()
        if path.startswith("/api/training"):
            return 200, {}
        return 404, dict(error="no such endpoint")

    if path == "/api/new":
        problem = S.new_game(
            game=body.get("game") or None,
            black=str(body.get("black", "human")),
            white=str(body.get("white", "rule")),
            size=str(body.get("size") or ""),
            ckpt=str(body.get("ckpt") or ""),
            seed=int(body.get("seed", 0)),
            black_sims=int(body.get("black_sims", g.DEFAULT_SIMS)),
            white_sims=int(body.get("white_sims", g.DEFAULT_SIMS)),
            black_vary=bool(body.get("black_vary", False)),
            white_vary=bool(body.get("white_vary", False)),
            black_ckpt=body.get("black_ckpt") or None,
            white_ckpt=body.get("white_ckpt") or None,
            players=body.get("players") or None,
            rated=bool(body.get("rated", False)))
        return 200, _with(S.snapshot(), problem)
    if path == "/api/opponent":
        if body.get("seat") is not None:
            index = max(0, min(int(body["seat"]), len(S.seats) - 1))
            seat = S.seats[index]
        else:
            seat = S.seats[0 if str(body.get("colour")) == "black" else 1]
        problem = S.set_opponent(seat, str(body.get("choice", "human")),
                                 int(body.get("sims", g.DEFAULT_SIMS)),
                                 bool(body.get("vary", False)), body.get("ckpt") or None)
        return 200, _with(S.snapshot(), problem)
    if path == "/api/review":
        with S.lock:
            game, moves, size = S.game, [m["cell"] for m in S.moves], S.size
        problem = S.review.start(game, moves, size, game.analysis_checkpoint(),
                                 int(body.get("sims", g.DEFAULT_REVIEW_SIMS)))
        return 200, _with(S.snapshot(), problem)
    if path == "/api/pace":
        S.set_pace(float(body.get("pace", 1.0)))
        return 200, S.snapshot()
    if path == "/api/analysis":
        S.analysis.configure(bool(body.get("enabled", False)),
                             int(body.get("max_sims", g.DEFAULT_EVAL_SIMS)))
        S.analysis.set_position(S.game, S.board, S.game.analysis_checkpoint())
        return 200, S.snapshot()
    if path == "/api/user":
        R = S.game.R
        problem = None
        if body.get("action") == "create":
            _, problem = R.create_user(str(body.get("name", "")))
        if problem is None:
            S.select_user(str(body.get("name", "")).strip() or None)
        return 200, _with(dict(S.snapshot(), users=sorted(R.load_users())), problem)
    if path == "/api/move":
        problem = S.play_human(S.game.parse_move(body, S.board))
        return (200 if problem is None else 409), _with(S.snapshot(), problem)
    if path == "/api/undo":
        problem = S.undo()
        return (200 if problem is None else 409), _with(S.snapshot(), problem)
    if path == "/api/remote":
        S.set_remote(body.get("seats") or [])
        return 200, S.snapshot()
    return 404, dict(error="no such endpoint")


# ---------------------------------------------------------- background work
def work() -> str:
    """What the worker should do next when it is otherwise idle.

    ``"bot"`` -- a bot is on turn; ``"review"`` / ``"analysis"`` -- a panel has
    work; ``""`` -- nothing.  The bot comes first: it is the game.
    """
    S = SESSION
    if S is None:
        return ""
    if S.bot_pending():
        return "bot"
    if S.review.busy():
        return "review"
    if S.analysis.busy():
        return "analysis"
    if _ponderer() is not None:
        return "ponder"
    return ""


def _ponderer():
    """A bot that wants to think on the human's time, while the human is on turn."""
    S = SESSION
    if S.thinking or S.board.is_terminal() or S.agents.get(S.board.to_move) is not None:
        return None
    for agent in S.agents.values():
        tick = getattr(agent, "ponder_tick", None)
        if tick is not None and agent.pondering():
            return agent
    return None


def step(kind: str) -> float:
    """Do one unit of ``kind``.  For a bot, returns the pause before playing."""
    S = SESSION
    if kind == "bot":
        return S.bot_think()
    if kind == "commit":
        S.bot_commit()
        return 0.0
    if kind == "review":
        S.review.tick()
    elif kind == "analysis":
        S.analysis.tick()
    elif kind == "ponder":
        agent = _ponderer()
        if agent is not None:
            agent.ponder_tick()
    return 0.0
