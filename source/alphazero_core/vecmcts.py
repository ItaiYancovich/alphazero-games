"""PUCT search for games with several players, chance, and hidden information.

:mod:`alphazero_core.mcts` is AlphaZero's search and assumes the game is a
two-player, perfect-information contest: it stores one number per edge and flips
its sign on the way up, which is only a meaningful thing to do between exactly
two people.  Three of those assumptions fail in Splendor, and each one has an
answer here.

**Several players: max^n instead of negamax.**  Every edge carries a *vector* of
values, one per seat.  A node maximises the component belonging to whoever is on
turn *there*, and a backup adds the whole vector to every edge on the path with
no sign flips at all.  With two seats and a zero-sum result this reduces exactly
to negamax; with four it is the standard max^n generalisation, which assumes
each player maximises their own score and nothing about what that costs anyone
else.

**Chance and hidden cards: an open-loop tree.**  There is no fixed position at a
node -- the deck could have turned up anything -- so nothing is cached there.
Each simulation re-samples a determinization of the root (via
``state.determinize``) and *replays the action indices* down the tree, so the
statistics at an edge are automatically averaged over the chance outcomes and
the hands that could have produced them.  This is what makes a fixed action
space essential rather than merely tidy: the action ``buy the second card of
tier 1`` means the same thing in every determinization, while *which card that
is* does not.

Two consequences of the open loop are worth naming, because they look like bugs:

* **Legality varies between simulations at the same node.**  The stored priors
  are therefore renormalised over whichever actions are legal on *this* pass,
  and an action seen legal only sometimes simply accumulates fewer visits.
* **There are no proven wins.**  :mod:`alphazero_core.mcts` runs an MCTS-solver
  and reports a forced win as exactly +1.  Proving anything about a shuffle you
  have not seen is not possible, so there is no solver here and the root value
  is always an average.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .vecstate import VecState, rotate_to_absolute


@dataclass
class VecMCTSConfig:
    simulations: int = 200
    c_puct: float = 1.6
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25
    # First-play urgency: an action nobody has tried inherits the node's own
    # value, reduced.  Matters more here than in the board games -- the action
    # space is small and most of it is legal, so an over-eager FPU would fan the
    # visits out over everything before deepening anywhere.
    fpu_reduction: float = 0.25
    add_noise: bool = True


BATCH = 16  # leaves per network call


def run_vec_search(search: "VecSearch", engine, target: int,
                   batch: int = BATCH) -> None:
    """Grow a tree until ``target`` simulations are behind it."""
    while search.sims_done < target:
        states = search.next_leaf_batch(batch)
        if not states:
            break
        priors, values = engine.evaluate(states)
        search.expand_batch(priors, values)


class VecSearch:
    """One max^n tree rooted at a position, searched open-loop.

    ``root_state`` is the position as the searching seat *knows* it; the search
    never looks at anything ``determinize`` would have hidden, so handing it the
    server's own copy is safe.
    """

    __slots__ = ("root_state", "cfg", "rng", "seat", "players", "nactions",
                 "n_nodes", "N", "W", "P", "child", "expanded",
                 "sims_done", "root_noise_applied", "_batch", "root_value_sum",
                 "root_visits")

    def __init__(self, root_state: VecState, cfg: VecMCTSConfig,
                 rng: np.random.Generator, nactions: int):
        self.root_state = root_state
        self.cfg = cfg
        self.rng = rng
        self.seat = int(root_state.to_move)
        self.players = int(root_state.players)
        self.nactions = int(nactions)
        self.n_nodes = 0
        self.N: list[np.ndarray] = []          # (A,)      visits per action
        self.W: list[np.ndarray] = []          # (A, P+1)  summed value per seat
        self.P: list[np.ndarray] = []          # (A,)      priors
        self.child: list[np.ndarray] = []      # (A,)      node index or -1
        self.expanded: list[bool] = []
        self.sims_done = 0
        self.root_noise_applied = False
        self._batch: list[tuple[list, int, VecState]] = []
        # The root's own value, accumulated over every simulation including the
        # ones that ended at a terminal position before reaching an edge.
        self.root_value_sum = np.zeros(self.players + 1, dtype=np.float64)
        self.root_visits = 0
        self._new_node()

    # ----------------------------------------------------------------- nodes
    def _new_node(self) -> int:
        idx = self.n_nodes
        self.n_nodes += 1
        self.N.append(np.zeros(self.nactions, dtype=np.float32))
        self.W.append(np.zeros((self.nactions, self.players + 1), dtype=np.float32))
        self.P.append(np.zeros(self.nactions, dtype=np.float32))
        self.child.append(np.full(self.nactions, -1, dtype=np.int32))
        self.expanded.append(False)
        return idx

    def is_expanded(self, node: int) -> bool:
        return self.expanded[node]

    # ------------------------------------------------------------- selection
    def _select(self, node: int, state: VecState) -> int:
        """PUCT among the actions legal in *this* determinization."""
        mask = state.legal_mask()
        seat = state.to_move
        N = self.N[node]
        W = self.W[node][:, seat]
        total = float(N.sum())
        sqrt_total = np.sqrt(total) if total > 0 else 1.0

        # Priors were fitted over whatever was legal when this node was first
        # reached.  Renormalise over what is legal now, so a determinization
        # that opens a new action does not hand it a prior of zero for ever.
        p = self.P[node] * mask
        s = float(p.sum())
        if s > 1e-12:
            p = p / s
        else:
            p = mask / max(float(mask.sum()), 1.0)

        parent_q = (float(W.sum()) / total) if total > 0 else 0.0
        q = np.where(N > 0, W / np.maximum(N, 1.0),
                     parent_q - self.cfg.fpu_reduction)
        u = self.cfg.c_puct * p * sqrt_total / (1.0 + N)
        score = np.where(mask, q + u, -np.inf)
        return int(np.argmax(score))

    # --------------------------------------------------------------- descent
    def next_leaf_batch(self, max_leaves: int) -> list[VecState]:
        """Collect up to ``max_leaves`` leaves needing evaluation.

        Each path carries a virtual loss on the acting seat's component, so the
        next descent is pushed elsewhere rather than piling onto the same leaf.
        """
        self._batch = []
        pending: set[int] = set()
        while (len(self._batch) < max_leaves
               and self.sims_done + len(self._batch) < self.cfg.simulations):
            state = self.root_state.determinize(self.seat, self.rng)
            path: list[tuple[int, int, int]] = []
            node = 0
            while True:
                if state.is_terminal():
                    self._undo_virtual_loss(path)
                    self._backup(path, state.result_vector())
                    self.sims_done += 1
                    node = -1
                    break
                if not self.expanded[node]:
                    break
                action = self._select(node, state)
                seat = state.to_move
                path.append((node, action, seat))
                self.N[node][action] += 1.0
                self.W[node][action, seat] -= 1.0
                state.play(action)
                nxt = int(self.child[node][action])
                if nxt < 0:
                    nxt = self._new_node()
                    self.child[node][action] = nxt
                node = nxt
            if node < 0:
                continue  # ended at a terminal position; nothing to evaluate
            if node in pending:
                # The same unexpanded node twice in one batch: evaluating it
                # twice would waste the slot, so stop collecting here.
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
                self.P[node] = np.asarray(priors[i], dtype=np.float32)
                self.expanded[node] = True
                if node == 0:
                    self._apply_root_noise(state)
            value = rotate_to_absolute(values[i], state.to_move, self.players)
            self._backup(path, value)
            self.sims_done += 1
        self._batch = []

    def _undo_virtual_loss(self, path) -> None:
        for node, action, seat in path:
            self.N[node][action] -= 1.0
            self.W[node][action, seat] += 1.0

    def _backup(self, path, value: np.ndarray) -> None:
        """Add one simulation's per-seat values to every edge on its path.

        No sign flips and no discount: the vector already says what the
        position is worth to each seat, so the seat on turn at each node reads
        off its own component when it selects.
        """
        self.root_value_sum += value
        self.root_visits += 1
        for node, action, _seat in path:
            self.N[node][action] += 1.0
            self.W[node][action] += value

    def _apply_root_noise(self, state: VecState) -> None:
        if not self.cfg.add_noise or self.root_noise_applied:
            self.root_noise_applied = True
            return
        legal = np.flatnonzero(state.legal_mask())
        self.root_noise_applied = True
        if len(legal) <= 1:
            return
        noise = self.rng.dirichlet(np.full(len(legal), self.cfg.dirichlet_alpha))
        eps = self.cfg.dirichlet_eps
        p = self.P[0].copy()
        p[legal] = (1 - eps) * p[legal] + eps * noise.astype(np.float32)
        self.P[0] = p

    # -------------------------------------------------------------- readout
    def root_visit_counts(self) -> np.ndarray:
        return self.N[0]

    def root_values(self) -> np.ndarray:
        """The root's value to each seat, ``(players + 1,)`` with index 0 unused."""
        if self.root_visits == 0:
            return np.zeros(self.players + 1, dtype=np.float32)
        return (self.root_value_sum / self.root_visits).astype(np.float32)

    def root_score(self) -> float:
        """What the position is worth to the seat that is searching."""
        return float(self.root_values()[self.seat])

    def root_child_scores(self) -> np.ndarray:
        """Per-action value to the searching seat; zero where never visited."""
        N = self.N[0]
        W = self.W[0][:, self.seat]
        return np.divide(W, N, out=np.zeros_like(W, dtype=np.float64), where=N > 0)

    def policy_target(self, temperature: float) -> np.ndarray:
        """Visit counts as a distribution over the whole action space."""
        N = self.N[0].astype(np.float64)
        total = N.sum()
        if total <= 0:
            mask = self.root_state.legal_mask().astype(np.float64)
            return (mask / max(mask.sum(), 1.0)).astype(np.float32)
        if temperature <= 1e-3:
            out = np.zeros_like(N)
            out[int(np.argmax(N))] = 1.0
            return out.astype(np.float32)
        counts = np.power(N, 1.0 / temperature)
        return (counts / counts.sum()).astype(np.float32)
