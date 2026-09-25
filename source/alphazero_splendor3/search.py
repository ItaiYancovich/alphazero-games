"""Gumbel AlphaZero over an open loop, for two to four seats.

v2 searched with PUCT, Dirichlet noise at the root, and a policy target made of
visit counts.  At two hundred simulations over thirty-odd legal moves that
target is mostly quantisation noise -- most actions get nought, one or two
visits -- and the v2 run's policy loss duly flattened at 1.86 with top-1 at
0.50.  This is the regime *Policy improvement by planning with Gumbel*
(Danihelka, Guez, Schrittwieser and Silver, ICLR 2022) was written for, so v3
uses it instead of PUCT:

* the root **samples** ``max_considered`` actions with Gumbel top-k without
  replacement, rather than exploring all of them badly;
* the simulation budget is split among them by **sequential halving**, so the
  survivors of each phase get twice the evidence the last phase gave them;
* the move played is ``argmax(g + logits + sigma(q))``, which is a *guaranteed*
  policy improvement at any budget, however small -- PUCT has no such guarantee
  at two hundred simulations;
* and the policy target is built from **completed Q-values** rather than visit
  counts, so every considered action carries a real number and the unvisited
  ones are completed with a value mix instead of a zero.

There is no ``c_puct`` and no Dirichlet noise: the Gumbel sample *is* the
exploration, and it is the same mechanism at the root in training and at
evaluation (where ``gumbel_scale`` is set to zero and the search becomes
deterministic).

Two properties of the open loop look like bugs and are not.  Every simulation
replays action indices against a freshly sampled deck, so legality varies
between simulations at the same interior node and the improved policy is
renormalised over whatever is legal on this pass.  And nothing is ever proven,
because nothing can be proven about a shuffle nobody has seen.

Values are a vector with one component per seat and no sign flips: the seat on
turn at each node reads its own component.  That is max^n, which is what a
three- or four-player table needs and what a two-player table reduces to.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from math import ceil, log2

import numpy as np

from .game import Game, NACTIONS

BATCH = 16


@dataclass
class SearchConfig:
    simulations: int = 128
    # How many root actions the Gumbel sample considers.  Sixteen of thirty is
    # most of the plausible ones; the rest are what the prior is for.
    max_considered: int = 16
    c_visit: float = 50.0
    c_scale: float = 0.5
    # 1.0 during self-play, 0.0 when the agent should play its best move.
    gumbel_scale: float = 1.0
    virtual_loss: float = 1.0


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


def _halving_schedule(m: int, simulations: int) -> list[tuple[int, int]]:
    """``(candidates kept, simulations each)`` per sequential-halving phase.

    The total never exceeds the budget.  Considering more actions than there are
    simulations to spend on them is not a search, so ``m`` is capped first --
    which is also what makes the arithmetic below come out exact.
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


class Search:
    """One max^n Gumbel tree rooted at a position, searched open-loop."""

    __slots__ = ("root", "cfg", "rng", "seat", "players", "N", "W", "P",
                 "child", "expanded", "total", "wsum", "V", "n_nodes",
                 "root_legal", "root_logits", "root_value", "cand", "gumbel",
                 "schedule", "phase", "queue", "_batch", "_pending", "started",
                 "sims_done")

    def __init__(self, root: Game, cfg: SearchConfig, rng: random.Random):
        self.root = root
        self.cfg = cfg
        self.rng = rng
        self.seat = root.to_move
        self.players = root.players

        self.N: list[list[float]] = []
        self.W: list[list[list[float]]] = []
        self.P: list[np.ndarray | None] = []
        self.child: list[list[int]] = []
        self.expanded: list[bool] = []
        self.total: list[float] = []
        self.wsum: list[list[float]] = []
        self.V: list[list[float]] = []
        self.n_nodes = 0
        self._new_node()

        self.root_legal: list[int] = list(root.legal_actions())
        self.root_logits: np.ndarray | None = None
        self.root_value: list[float] = [0.0] * (self.players + 1)
        self.cand: list[int] = []
        self.gumbel: dict[int, float] = {}
        self.schedule: list[tuple[int, int]] = []
        self.phase = -1
        self.queue: list[int] = []
        self._batch: list = []
        self._pending = 0
        self.started = False
        self.sims_done = 0

    # ------------------------------------------------------------ tree store
    def _new_node(self) -> int:
        idx = self.n_nodes
        self.n_nodes += 1
        self.N.append([0.0] * NACTIONS)
        self.W.append([[0.0] * NACTIONS for _ in range(self.players + 1)])
        self.P.append(None)
        self.child.append([-1] * NACTIONS)
        self.expanded.append(False)
        self.total.append(0.0)
        self.wsum.append([0.0] * (self.players + 1))
        self.V.append([0.0] * (self.players + 1))
        return idx

    # ------------------------------------------------------------------ root
    def root_states(self) -> list[Game]:
        """What the root evaluation is run on: the position as this seat knows it."""
        return [self.root]

    def set_root(self, logits: np.ndarray, values: np.ndarray) -> None:
        """Seed the root with the network's opinion and draw the Gumbel sample."""
        self.root_logits = np.asarray(logits, dtype=np.float64)
        self.P[0] = self.root_logits
        self.expanded[0] = True
        self.root_value = rotate_to_absolute(values, self.seat, self.players)
        self.V[0] = self.root_value

        legal = self.root_legal
        m = min(self.cfg.max_considered, len(legal), max(1, self.cfg.simulations))
        scale = self.cfg.gumbel_scale
        if scale > 0.0:
            # Gumbel(0, 1) is -log(-log U); adding it to the logits and taking
            # the top m is an exact sample of m actions without replacement.
            g = {a: -scale * np.log(-np.log(max(self.rng.random(), 1e-12)))
                 for a in legal}
        else:
            g = {a: 0.0 for a in legal}
        self.gumbel = g
        self.cand = sorted(legal, key=lambda a: g[a] + self.root_logits[a],
                           reverse=True)[:m]
        self.schedule = _halving_schedule(len(self.cand), self.cfg.simulations)
        self.phase = -1
        self.started = True
        self._advance_phase()

    def _advance_phase(self) -> bool:
        """Halve the candidate set and refill the queue.  False when finished."""
        if self.phase >= 0 and len(self.cand) > 1:
            score = self.root_scores()
            self.cand = sorted(self.cand, key=score.__getitem__,
                               reverse=True)[:len(self.cand) // 2]
        self.phase += 1
        if self.phase >= len(self.schedule) or not self.cand:
            return False
        _, per = self.schedule[self.phase]
        # Round-robin rather than in blocks, so a partly spent phase is still
        # a fair comparison between its candidates.
        self.queue = [a for _ in range(per) for a in self.cand]
        return True

    def _sigma(self, q: np.ndarray, max_n: float) -> np.ndarray:
        """The paper's monotone transform, on Q rescaled to ``[0, 1]``."""
        return (self.cfg.c_visit + max_n) * self.cfg.c_scale * (q + 1.0) * 0.5

    def root_scores(self) -> dict:
        """``g + logits + sigma(q)`` for every root candidate, computed once."""
        q = self._completed_q(0, self.root_legal, self.seat)
        max_n = max(self.N[0]) if self.N[0] else 0.0
        s = self._sigma(q, max_n)
        return {a: float(self.gumbel[a] + self.root_logits[a] + s[i])
                for i, a in enumerate(self.root_legal)}

    # ----------------------------------------------------- completed Q values
    def _completed_q(self, node: int, state_legal: list[int], seat: int) -> np.ndarray:
        """Q for every legal action, unvisited ones completed with a value mix."""
        logits = self.P[node]
        Nn = self.N[node]
        Wn = self.W[node][seat]
        pi = np.exp(logits[state_legal])
        s = pi.sum()
        pi = pi / s if s > 1e-30 else np.full(len(state_legal), 1.0 / len(state_legal))

        visited = 0.0
        pi_visited = 0.0
        weighted = 0.0
        q = np.empty(len(state_legal), dtype=np.float64)
        for i, a in enumerate(state_legal):
            n = Nn[a]
            if n > 0.0:
                qa = Wn[a] / n
                q[i] = qa
                visited += n
                pi_visited += pi[i]
                weighted += pi[i] * qa
            else:
                q[i] = np.nan
        if visited > 0.0 and pi_visited > 1e-12:
            v_mix = (self.V[node][seat] + (visited / pi_visited) * weighted) / (1.0 + visited)
        else:
            v_mix = self.V[node][seat]
        np.nan_to_num(q, copy=False, nan=v_mix)
        return q

    # --------------------------------------------------------------- descent
    def _select(self, node: int, state: Game) -> int:
        """Gumbel MuZero's deterministic interior rule.

        The improved policy says how often each action *should* have been
        visited; take the one furthest behind that quota.  No exploration
        constant, and nothing to tune.
        """
        legal = state.legal_actions()
        if len(legal) == 1:
            return legal[0]
        seat = state.to_move
        q = self._completed_q(node, legal, seat)
        max_n = max(self.N[node]) if self.N[node] else 0.0
        adjusted = self.P[node][legal] + self._sigma(q, max_n)
        adjusted -= adjusted.max()
        improved = np.exp(adjusted)
        improved /= improved.sum()

        Nn = self.N[node]
        total = 1.0 + self.total[node]
        best = -1e30
        chosen = legal[0]
        for i, a in enumerate(legal):
            score = improved[i] - Nn[a] / total
            if score > best:
                best = score
                chosen = a
        return chosen

    def next_leaf_batch(self, max_leaves: int) -> list[Game]:
        """Collect up to ``max_leaves`` leaves that need a network evaluation."""
        self._batch = []
        pending: set[int] = set()
        vloss = self.cfg.virtual_loss
        while len(self._batch) < max_leaves:
            if not self.queue:
                if self._pending or self._batch:
                    break
                if not self._advance_phase():
                    break
                continue
            root_action = self.queue.pop()
            state = self.root.determinize(self.seat, self.rng)
            if root_action not in state.legal_actions():
                continue                       # cannot happen; cheap to survive
            path: list[tuple[int, int, int]] = []
            node = 0
            action = root_action
            while True:
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
            if node < 0:
                continue
            if node in pending:
                self._undo_virtual_loss(path)
                self.queue.append(root_action)
                break
            pending.add(node)
            self._batch.append((path, node, state))
        self._pending = len(self._batch)
        return [b[2] for b in self._batch]

    def expand_batch(self, logits: np.ndarray, values: np.ndarray) -> None:
        for i, (path, node, state) in enumerate(self._batch):
            self._undo_virtual_loss(path)
            if not self.expanded[node]:
                self.P[node] = np.asarray(logits[i], dtype=np.float64)
                self.expanded[node] = True
            absolute = rotate_to_absolute(values[i], state.to_move, self.players)
            self.V[node] = absolute
            self._backup(path, absolute)
            self.sims_done += 1
        self._batch = []
        self._pending = 0

    def _undo_virtual_loss(self, path) -> None:
        vloss = self.cfg.virtual_loss
        for node, action, seat in path:
            self.N[node][action] -= 1.0
            self.W[node][seat][action] += vloss
            self.total[node] -= 1.0
            self.wsum[node][seat] += vloss

    def _backup(self, path, value: list[float]) -> None:
        for node, action, _seat in path:
            self.N[node][action] += 1.0
            self.total[node] += 1.0
            Wn = self.W[node]
            wsum = self.wsum[node]
            for s in range(1, self.players + 1):
                v = value[s]
                Wn[s][action] += v
                wsum[s] += v

    def done(self) -> bool:
        return (self.started and not self.queue and not self._batch
                and self.phase >= len(self.schedule))

    # ---------------------------------------------------------------- outputs
    def best_action(self) -> int:
        if len(self.root_legal) == 1:
            return self.root_legal[0]
        if not self.cand:
            return self.root_legal[0]
        score = self.root_scores()
        return max(self.cand, key=score.__getitem__)

    def policy_target(self) -> np.ndarray:
        """The improved policy over all 72 actions: softmax(logits + sigma(q)).

        Every legal action gets a real number, whether the search visited it or
        not, which is the whole point of completing the Q-values: a visit-count
        target at this budget would be mostly zeros.
        """
        out = np.zeros(NACTIONS, dtype=np.float32)
        legal = self.root_legal
        if not legal:
            return out
        if self.root_logits is None:
            out[legal] = 1.0 / len(legal)
            return out
        q = self._completed_q(0, legal, self.seat)
        max_n = max(self.N[0]) if self.N[0] else 0.0
        adjusted = self.root_logits[legal] + self._sigma(q, max_n)
        adjusted -= adjusted.max()
        p = np.exp(adjusted)
        out[legal] = (p / p.sum()).astype(np.float32)
        return out

    def root_values(self) -> np.ndarray:
        """The searched value of the position, per seat, index 0 unused."""
        out = np.zeros(self.players + 1, dtype=np.float32)
        n = self.total[0]
        if n <= 0:
            for s in range(1, self.players + 1):
                out[s] = self.root_value[s]
            return out
        for s in range(1, self.players + 1):
            out[s] = self.wsum[0][s] / n
        return out

    def root_score(self) -> float:
        return float((self.root_values()[self.seat] + 1.0) / 2.0)


def run_search(search: Search, engine, batch: int = BATCH) -> None:
    """Drive one search to completion against a batched evaluator."""
    if not search.started:
        logits, values = engine.evaluate(search.root_states())
        search.set_root(logits[0], values[0])
    # ``next_leaf_batch`` advances the halving phases itself and returns an
    # empty list only when the whole schedule is spent.
    for _ in range(10_000):
        leaves = search.next_leaf_batch(batch)
        if not leaves:
            return
        logits, values = engine.evaluate(leaves)
        search.expand_batch(logits, values)
