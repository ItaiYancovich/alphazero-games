"""Gumbel AlphaZero for the closed-loop board games.

The PUCT search in :mod:`alphazero_core.mcts` explores every root action a
little and builds its policy target from visit counts.  At the budgets a CPU
run can afford that target is mostly quantisation noise: at 400 simulations
over the eight or nine moves Ultimate allows it is passable, but at 100 it is a
handful of integers, and the improvement it encodes is not guaranteed to be an
improvement at all.

*Policy improvement by planning with Gumbel* (Danihelka, Guez, Schrittwieser
and Silver, ICLR 2022) replaces that with something that **is** guaranteed at
any budget, however small:

* the root **samples** ``max_considered`` actions by Gumbel top-k without
  replacement instead of spreading the budget thinly over all of them;
* the budget is divided among the survivors by **sequential halving**, so each
  phase gives twice the evidence the last one did;
* the move played is ``argmax(g + logits + sigma(q))``, which is a policy
  improvement over the prior in expectation for any number of simulations;
* and the policy target is built from **completed Q-values** rather than visit
  counts, so an action that was never visited still carries a real number
  instead of a zero.

There is no ``c_puct`` and no Dirichlet noise -- the Gumbel sample *is* the
exploration, and setting ``gumbel_scale`` to zero turns the same code into a
deterministic evaluation-strength search.

This is a sibling of :class:`alphazero_core.mcts.Search`, not a replacement.
It offers the same handful of methods the self-play driver and the agents use
(``next_leaf``, ``expand``, ``policy_target``, ``root_score``, ``child_search``)
so a caller can swap one for the other, and the two can be compared on equal
terms.  Splendor's :mod:`alphazero_splendor3.search` is the open-loop, many-seat
cousin of this file; the mathematics is the same and the bookkeeping is not,
because there the deck is reshuffled every simulation and nothing can be proven.

What is deliberately *not* carried over from the PUCT search is the solver.
Sequential halving spends its budget on a fixed candidate set, so a proven win
found deep in one branch cannot be used to abandon the others early; a proven
terminal value is still backed up as an ordinary value, which is correct, just
not exploited.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2

import numpy as np

from .state import GameState

BATCH = 24


@dataclass
class GumbelConfig:
    simulations: int = 128
    # How many root actions to consider.  Ultimate offers nine moves in a
    # forced board and up to 81 in a free one, so this only bites on the free
    # ones -- which is exactly where a thin spread of visits is least useful.
    max_considered: int = 16
    # The paper's two constants for the monotone transform of Q.  These are the
    # published defaults and there is no reason here to depart from them.
    c_visit: float = 50.0
    c_scale: float = 1.0
    # 1.0 while training, 0.0 when the agent should simply play its best move.
    gumbel_scale: float = 1.0
    value_discount: float = 1.0


def halving_schedule(m: int, simulations: int) -> list[tuple[int, int]]:
    """``(candidates kept, simulations each)`` for each sequential-halving phase.

    The total never exceeds the budget.  Considering more actions than there
    are simulations to spend on them is not a search, so ``m`` is capped first.
    """
    simulations = max(1, int(simulations))
    m = max(1, min(int(m), simulations))
    if m == 1:
        return [(1, simulations)]
    phases = max(1, int(ceil(log2(m))))
    out: list[tuple[int, int]] = []
    left = simulations
    k = m
    for p in range(phases):
        if left <= 0:
            break
        if p == phases - 1:
            per = max(1, left // k)
        else:
            per = max(1, (simulations // phases) // k)
        if per * k > left:
            per = max(1, left // k)
            k = min(k, max(1, left // per))
        out.append((k, per))
        left -= per * k
        k = max(1, k // 2)
    return out


class GumbelSearch:
    """One Gumbel tree over a deterministic, perfect-information position."""

    __slots__ = ("root_state", "cfg", "rng", "n_nodes", "moves", "P", "N", "W",
                 "child", "term_value", "V", "sims_done", "sim_budget",
                 "_pending_path", "_pending_state", "_pending_node",
                 "root_logits", "cand", "gumbel", "schedule", "phase", "queue",
                 "started")

    def __init__(self, root_state: GameState, cfg: GumbelConfig,
                 rng: np.random.Generator):
        self.root_state = root_state
        self.cfg = cfg
        self.rng = rng
        self.n_nodes = 0
        self.moves: list[np.ndarray | None] = []
        self.P: list[np.ndarray | None] = []      # priors, as probabilities
        self.N: list[np.ndarray | None] = []
        self.W: list[np.ndarray | None] = []
        self.child: list[np.ndarray | None] = []
        self.term_value: list[float | None] = []
        self.V: list[float] = []                  # the node's own network value
        self.sims_done = 0
        self.sim_budget: int | None = None
        self._pending_path: list[tuple[int, int]] = []
        self._pending_state: GameState | None = None
        self._pending_node: int = -1
        self.root_logits: np.ndarray | None = None
        self.cand: list[int] = []                 # indices into moves[0]
        self.gumbel: np.ndarray | None = None
        self.schedule: list[tuple[int, int]] = []
        self.phase = -1
        self.queue: list[int] = []
        self.started = False
        self._new_node()

    # ----------------------------------------------------------------- nodes
    @property
    def budget(self) -> int:
        return self.cfg.simulations if self.sim_budget is None else self.sim_budget

    def _new_node(self) -> int:
        idx = self.n_nodes
        self.n_nodes += 1
        for arr in (self.moves, self.P, self.N, self.W, self.child, self.term_value):
            arr.append(None)
        self.V.append(0.0)
        return idx

    def is_expanded(self, node: int) -> bool:
        return self.moves[node] is not None

    # ------------------------------------------------------------- the root
    def _start_root(self) -> None:
        """Draw the Gumbel sample and lay out the halving schedule."""
        k = len(self.moves[0])
        scale = self.cfg.gumbel_scale
        if scale > 0.0:
            # Gumbel(0, 1) is -log(-log U); adding it to the logits and taking
            # the top m is an exact sample of m actions without replacement.
            u = np.clip(self.rng.random(k), 1e-12, 1.0)
            self.gumbel = (-scale * np.log(-np.log(u))).astype(np.float64)
        else:
            self.gumbel = np.zeros(k, dtype=np.float64)
        m = min(self.cfg.max_considered, k, max(1, self.budget))
        order = np.argsort(-(self.gumbel + self.root_logits))
        self.cand = [int(a) for a in order[:m]]
        self.schedule = halving_schedule(len(self.cand), self.budget)
        self.phase = -1
        self.started = True
        self._advance_phase()

    def _advance_phase(self) -> bool:
        """Halve the candidate set and refill the queue.  False when finished."""
        if self.phase >= 0 and len(self.cand) > 1:
            score = self.root_scores()
            self.cand = sorted(self.cand, key=lambda a: score[a],
                               reverse=True)[:len(self.cand) // 2]
        self.phase += 1
        if self.phase >= len(self.schedule) or not self.cand:
            return False
        _, per = self.schedule[self.phase]
        # Round-robin rather than in blocks, so a phase cut short by the budget
        # is still a fair comparison between its candidates.
        self.queue = [a for _ in range(per) for a in self.cand]
        return True

    def _sigma(self, q: np.ndarray, max_n: float) -> np.ndarray:
        """The paper's monotone transform, on Q rescaled to ``[0, 1]``."""
        return (self.cfg.c_visit + max_n) * self.cfg.c_scale * (q + 1.0) * 0.5

    def root_scores(self) -> np.ndarray:
        """``g + logits + sigma(q)`` for every root action."""
        q = self.completed_q(0)
        n = self.N[0]
        return self.gumbel + self.root_logits + self._sigma(q, float(n.max()) if len(n) else 0.0)

    # ----------------------------------------------------- completed Q values
    def completed_q(self, node: int) -> np.ndarray:
        """Q for every action, unvisited ones completed with a value mix.

        The mix is the paper's: the node's own value and the visited children's
        Q, weighted by how much prior mass has actually been explored.  It is
        what lets an unvisited action carry a number rather than a zero, which
        is the difference between a usable target and a histogram of noise.
        """
        p = self.P[node]
        n = self.N[node]
        w = self.W[node]
        visited = n > 0
        q = np.where(visited, w / np.maximum(n, 1.0), 0.0).astype(np.float64)
        total = float(n.sum())
        if total > 0.0:
            pi_visited = float(p[visited].sum())
            if pi_visited > 1e-12:
                weighted = float((p[visited] * q[visited]).sum()) / pi_visited
                v_mix = (self.V[node] + total * weighted) / (1.0 + total)
            else:
                v_mix = self.V[node]
        else:
            v_mix = self.V[node]
        return np.where(visited, q, v_mix)

    def _select(self, node: int) -> int:
        """Gumbel MuZero's deterministic interior rule.

        The improved policy says how often each action *should* have been
        visited; take whichever is furthest behind that quota.  No exploration
        constant and nothing to tune.
        """
        n = self.N[node]
        if len(n) == 1:
            return 0
        q = self.completed_q(node)
        adjusted = np.log(np.maximum(self.P[node], 1e-30)) + \
            self._sigma(q, float(n.max()))
        adjusted -= adjusted.max()
        improved = np.exp(adjusted)
        improved /= improved.sum()
        return int(np.argmax(improved - n / (1.0 + float(n.sum()))))

    # ------------------------------------------------------------ traversal
    def next_leaf(self) -> tuple[GameState, int] | None:
        """Descend to a leaf needing evaluation, or ``None`` when finished."""
        while True:
            if not self.is_expanded(0):
                # The root itself has to be evaluated before anything else.
                self._pending_path = []
                self._pending_state = self.root_state.copy()
                self._pending_node = 0
                return self._pending_state, 0
            if not self.started:
                self._start_root()
            if not self.queue:
                if not self._advance_phase():
                    return None
                continue
            a = self.queue.pop()
            state = self.root_state.copy()
            state.play(int(self.moves[0][a]))
            path = [(0, a)]
            node = int(self.child[0][a])
            if node < 0:
                node = self._new_node()
                self.child[0][a] = node
                if state.is_terminal():
                    self.term_value[node] = state.terminal_value()
            leaf = self._descend(state, node, path)
            if leaf is not None:
                return leaf

    def _descend(self, state, node, path):
        """Walk down from ``node`` until something needs the network."""
        while True:
            if self.term_value[node] is not None:
                self._backup(path, self.term_value[node])
                self.sims_done += 1
                return None
            if not self.is_expanded(node):
                self._pending_path = path
                self._pending_state = state
                self._pending_node = node
                return state, node
            a = self._select(node)
            path.append((node, a))
            state.play(int(self.moves[node][a]))
            nxt = int(self.child[node][a])
            if nxt < 0:
                nxt = self._new_node()
                self.child[node][a] = nxt
                if state.is_terminal():
                    self.term_value[nxt] = state.terminal_value()
            node = nxt

    def expand(self, node: int, priors: np.ndarray, value: float) -> None:
        """Attach network output to the pending leaf and back its value up."""
        state = self._pending_state
        legal = state.legal_moves()
        if len(legal) == 0:
            self.term_value[node] = 0.0
            self._backup(self._pending_path, 0.0)
            self.sims_done += 1
            self._pending_state = None
            return
        canon = np.array([state.to_canonical_move(int(m)) for m in legal],
                         dtype=np.int64)
        p = priors[canon].astype(np.float64)
        s = p.sum()
        p = p / s if s > 1e-12 else np.full(len(legal), 1.0 / len(legal))

        self.moves[node] = legal
        self.P[node] = p
        self.N[node] = np.zeros(len(legal), dtype=np.float64)
        self.W[node] = np.zeros(len(legal), dtype=np.float64)
        self.child[node] = np.full(len(legal), -1, dtype=np.int32)
        self.V[node] = float(value)
        if node == 0:
            # Gumbel top-k needs logits; the evaluator hands back a masked
            # softmax, and log of that differs from the true logits only by a
            # constant, which top-k and the argmax below are invariant to.
            self.root_logits = np.log(np.maximum(p, 1e-30))

        self._backup(self._pending_path, float(value))
        self.sims_done += 1
        self._pending_state = None
        self._pending_node = -1

    def _backup(self, path, value: float) -> None:
        v = value
        g = self.cfg.value_discount
        for node, a in reversed(path):
            v = -v * g
            self.N[node][a] += 1.0
            self.W[node][a] += v

    # -------------------------------------------------------------- readouts
    def root_visit_distribution(self) -> tuple[np.ndarray, np.ndarray]:
        return self.moves[0], self.N[0]

    def root_value(self) -> float:
        n = self.N[0]
        total = float(n.sum())
        return float(self.W[0].sum() / total) if total > 0 else self.V[0]

    def root_score(self) -> float:
        """What the position is worth, for display and for the value target."""
        if not self.is_expanded(0):
            return 0.0
        return float(self.completed_q(0).max()) if float(self.N[0].sum()) > 0 \
            else self.V[0]

    def best_move(self) -> int:
        """``argmax(g + logits + sigma(q))`` over the surviving candidates."""
        scores = self.root_scores()
        pool = self.cand if self.cand else list(range(len(self.moves[0])))
        return int(self.moves[0][max(pool, key=lambda a: scores[a])])

    def policy_target(self, temperature: float = 1.0,
                      prune: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """The improved policy, from completed Q rather than visit counts.

        ``temperature`` is accepted so this can stand in for the PUCT search's
        method, and it sharpens the same way -- but the distribution it
        sharpens is already a policy improvement, which visit counts at these
        budgets are not.  ``prune`` is accepted for the same interchangeability
        and ignored: there are no forced playouts to take back out, because
        there was no Dirichlet noise to compensate for in the first place.
        """
        moves = self.moves[0]
        n = self.N[0]
        q = self.completed_q(0)
        adjusted = self.root_logits + self._sigma(q, float(n.max()) if len(n) else 0.0)
        adjusted -= adjusted.max()
        probs = np.exp(adjusted)
        probs /= probs.sum()
        if temperature <= 1e-3:
            out = np.zeros_like(probs)
            out[int(np.argmax(probs))] = 1.0
            return moves, out.astype(np.float32)
        if abs(temperature - 1.0) > 1e-6:
            probs = np.power(probs, 1.0 / temperature)
            probs /= probs.sum()
        return moves, probs.astype(np.float32)

    def child_search(self, move: int) -> "GumbelSearch | None":
        """No tree reuse: sequential halving owns its budget from the root.

        Carrying a subtree over would mean starting a phase schedule against
        visit counts collected under a different candidate set, which is not
        the algorithm.  The lost work is smaller than it looks -- most of a
        Gumbel budget goes to the few actions that survive halving.
        """
        return None
