"""Max^n PUCT over an open loop, for the v2 game.

The algorithm is :mod:`alphazero_core.vecmcts` -- a vector of values per edge,
one per seat, no sign flips, and an open tree in which every simulation replays
*action indices* against a freshly sampled deck rather than descending a fixed
position.  What differs is what it runs on, and what that lets it skip.

**Chance is sampled where the rules put it.**  In v1 a simulation began by
re-permuting all three decks at the root, which cost more than the descent that
followed.  Here a deck is a bag and a card is drawn when a slot refills, so a
simulation begins with a copy; a determinization is then only about the cards an
*opponent* reserved off the top of a deck -- usually none -- and hidden
information costs essentially nothing per simulation.

**The tree is plain Python, not numpy.**  This is not a stylistic choice: a
Splendor node has around thirty legal actions, and at that size a numpy
operation is almost entirely call overhead.  The vectorised version of ``_select``
below measured 32 microseconds against 8 for the loop that replaced it, and it
was forty per cent of the search.  Numpy earns its place in the *encoder*, where
a batch is thousands of numbers wide; it does not earn it here.

Two properties of the open loop are worth restating because they look like bugs:
legality varies between simulations at the same node, so the stored priors are
renormalised over whatever is legal on this pass; and there are no proven wins,
because nothing can be proven about a shuffle nobody has seen.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from math import sqrt

import numpy as np

from .game import Game, NACTIONS

BATCH = 32


@dataclass
class SearchConfig:
    simulations: int = 200
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25
    # An action nobody has tried inherits the node's own value, reduced.  With
    # thirty-odd legal moves and a few hundred simulations, too small a
    # reduction spends the whole budget trying everything once.
    fpu_reduction: float = 0.3
    virtual_loss: float = 1.0
    add_noise: bool = True


def rotate_to_absolute(values, to_move: int, players: int) -> list[float]:
    """Seat-relative network output -> a list indexed by seat, 0 unused."""
    out = [0.0] * (players + 1)
    for k in range(players):
        out[(to_move - 1 + k) % players + 1] = float(values[k])
    return out


def rotate_to_relative(values, to_move: int, players: int, seats: int) -> np.ndarray:
    out = np.zeros(seats, dtype=np.float32)
    for k in range(min(players, seats)):
        out[k] = values[(to_move - 1 + k) % players + 1]
    return out


class Search:
    """One max^n tree rooted at a position, searched open-loop.

    ``root`` is the position as the searching seat *knows* it; the search never
    looks at anything :meth:`Game.determinize` would have hidden, so handing it
    the server's own copy is safe.
    """

    __slots__ = ("root", "cfg", "rng", "seat", "players", "N", "W", "P",
                 "child", "expanded", "total", "wsum", "sims_done", "_batch",
                 "root_value_sum", "root_visits", "n_nodes")

    def __init__(self, root: Game, cfg: SearchConfig, rng: random.Random):
        self.root = root
        self.cfg = cfg
        self.rng = rng
        self.seat = int(root.to_move)
        self.players = int(root.players)
        self.n_nodes = 0
        # Per node: visits per action, summed value per seat per action, priors,
        # child node index.  Lists of floats, indexed [action] and [seat][action].
        self.N: list[list[float]] = []
        self.W: list[list[list[float]]] = []
        self.P: list[list[float]] = []
        self.child: list[list[int]] = []
        self.expanded: list[bool] = []
        # Maintained alongside, so selection never has to sum a whole node:
        # total visits, and the summed value per seat across every action.
        self.total: list[float] = []
        self.wsum: list[list[float]] = []
        self.sims_done = 0
        self._batch: list[tuple[list, int, Game]] = []
        self.root_value_sum = [0.0] * (self.players + 1)
        self.root_visits = 0
        self._new_node()

    # ----------------------------------------------------------------- nodes
    def _new_node(self) -> int:
        idx = self.n_nodes
        self.n_nodes += 1
        self.N.append([0.0] * NACTIONS)
        self.W.append([[0.0] * NACTIONS for _ in range(self.players + 1)])
        self.P.append([0.0] * NACTIONS)
        self.child.append([-1] * NACTIONS)
        self.expanded.append(False)
        self.total.append(0.0)
        self.wsum.append([0.0] * (self.players + 1))
        return idx

    # ------------------------------------------------------------- selection
    def _select(self, node: int, state: Game) -> int:
        """PUCT among the actions legal in *this* determinization."""
        legal = state.legal_actions()
        if len(legal) == 1:
            return legal[0]
        seat = state.to_move
        Nn = self.N[node]
        Wn = self.W[node][seat]
        Pn = self.P[node]
        total = self.total[node]

        # The priors were fitted over whatever was legal when this node was
        # first reached.  Renormalise over what is legal now, so a
        # determinization that opens a new action does not hand it a prior of
        # zero for ever.
        psum = 0.0
        for a in legal:
            psum += Pn[a]
        if psum > 1e-12:
            scale = self.cfg.c_puct * (sqrt(total) if total > 0.0 else 1.0) / psum
            flat = 0.0
        else:
            scale = 0.0
            flat = (self.cfg.c_puct * (sqrt(total) if total > 0.0 else 1.0)
                    / len(legal))

        if total > 0.0:
            fpu = self.wsum[node][seat] / total - self.cfg.fpu_reduction
        else:
            fpu = -self.cfg.fpu_reduction

        best = -1e30
        chosen = legal[0]
        for a in legal:
            n = Nn[a]
            q = Wn[a] / n if n > 0.0 else fpu
            score = q + (Pn[a] * scale + flat) / (1.0 + n)
            if score > best:
                best = score
                chosen = a
        return chosen

    # --------------------------------------------------------------- descent
    def next_leaf_batch(self, max_leaves: int) -> list[Game]:
        """Collect up to ``max_leaves`` leaves needing a network evaluation.

        Each path carries a virtual loss on the acting seat's component, so the
        next descent is pushed elsewhere rather than piling onto the same leaf.
        """
        self._batch = []
        pending: set[int] = set()
        vloss = self.cfg.virtual_loss
        while (len(self._batch) < max_leaves
               and self.sims_done + len(self._batch) < self.cfg.simulations):
            state = self.root.determinize(self.seat, self.rng)
            path: list[tuple[int, int, int]] = []
            node = 0
            while True:
                if state.to_move == 0:
                    self._undo_virtual_loss(path)
                    result = state.result_vector()
                    self._backup(path, [0.0] + [float(v) for v in result[1:]])
                    self.sims_done += 1
                    node = -1
                    break
                if not self.expanded[node]:
                    break
                action = self._select(node, state)
                seat = state.to_move
                path.append((node, action, seat))
                self.N[node][action] += 1.0
                self.W[node][seat][action] -= vloss
                self.total[node] += 1.0
                self.wsum[node][seat] -= vloss
                state.play(action)
                nxt = self.child[node][action]
                if nxt < 0:
                    nxt = self._new_node()
                    self.child[node][action] = nxt
                node = nxt
            if node < 0:
                continue  # ended at a terminal position; nothing to evaluate
            if node in pending:
                # The same unexpanded node twice in one batch would waste the
                # slot, so stop collecting here.
                self._undo_virtual_loss(path)
                break
            pending.add(node)
            self._batch.append((path, node, state))
        return [b[2] for b in self._batch]

    def expand_batch(self, priors: np.ndarray, values: np.ndarray) -> None:
        """Attach network output to each pending leaf and back its value up."""
        for i, (path, node, state) in enumerate(self._batch):
            self._undo_virtual_loss(path)
            if not self.expanded[node]:
                self.P[node] = priors[i].tolist()
                self.expanded[node] = True
                if node == 0:
                    self._apply_root_noise(state)
            self._backup(path, rotate_to_absolute(values[i], state.to_move,
                                                  self.players))
            self.sims_done += 1
        self._batch = []

    def _undo_virtual_loss(self, path) -> None:
        vloss = self.cfg.virtual_loss
        for node, action, seat in path:
            self.N[node][action] -= 1.0
            self.W[node][seat][action] += vloss
            self.total[node] -= 1.0
            self.wsum[node][seat] += vloss

    def _backup(self, path, value: list[float]) -> None:
        """Add one simulation's per-seat values to every edge on its path.

        No sign flips and no discount: the vector already says what the position
        is worth to each seat, so the seat on turn at each node reads off its own
        component when it selects.
        """
        rvs = self.root_value_sum
        for s in range(1, self.players + 1):
            rvs[s] += value[s]
        self.root_visits += 1
        for node, action, _seat in path:
            self.N[node][action] += 1.0
            self.total[node] += 1.0
            Wn = self.W[node]
            wsum = self.wsum[node]
            for s in range(1, self.players + 1):
                v = value[s]
                Wn[s][action] += v
                wsum[s] += v

    def _apply_root_noise(self, state: Game) -> None:
        if not self.cfg.add_noise:
            return
        legal = state.legal_actions()
        if len(legal) <= 1:
            return
        alpha = self.cfg.dirichlet_alpha
        # Dirichlet(alpha, ..., alpha) as normalised Gamma(alpha) draws -- the
        # standard construction, and it keeps the search on one RNG.
        draws = [self.rng.gammavariate(alpha, 1.0) for _ in legal]
        total = sum(draws) or 1.0
        eps = self.cfg.dirichlet_eps
        p = self.P[0]
        for a, d in zip(legal, draws):
            p[a] = (1 - eps) * p[a] + eps * (d / total)

    # -------------------------------------------------------------- readout
    def root_visit_counts(self) -> np.ndarray:
        return np.asarray(self.N[0], dtype=np.float32)

    def root_values(self) -> np.ndarray:
        """The root's value to each seat, ``(players + 1,)``, index 0 unused."""
        if self.root_visits == 0:
            return np.zeros(self.players + 1, dtype=np.float32)
        return (np.asarray(self.root_value_sum, dtype=np.float64)
                / self.root_visits).astype(np.float32)

    def root_score(self) -> float:
        """What the position is worth to the seat that is searching."""
        return float(self.root_values()[self.seat])

    def root_child_scores(self) -> np.ndarray:
        """Per-action value to the searching seat; zero where never visited."""
        N = np.asarray(self.N[0], dtype=np.float64)
        W = np.asarray(self.W[0][self.seat], dtype=np.float64)
        return np.divide(W, N, out=np.zeros_like(W), where=N > 0)

    def policy_target(self, temperature: float) -> np.ndarray:
        """Visit counts as a distribution over the whole action space."""
        N = np.asarray(self.N[0], dtype=np.float64)
        total = N.sum()
        if total <= 0:
            mask = self.root.legal_mask().astype(np.float64)
            return (mask / max(mask.sum(), 1.0)).astype(np.float32)
        if temperature <= 1e-3:
            out = np.zeros_like(N)
            out[int(np.argmax(N))] = 1.0
            return out.astype(np.float32)
        counts = np.power(N, 1.0 / temperature)
        return (counts / counts.sum()).astype(np.float32)


def run_search(search: Search, engine, target: int, batch: int = BATCH) -> None:
    """Grow a tree until ``target`` simulations are behind it."""
    while search.sims_done < target:
        states = search.next_leaf_batch(batch)
        if not states:
            break
        priors, values = engine.evaluate(states)
        search.expand_batch(priors, values)
