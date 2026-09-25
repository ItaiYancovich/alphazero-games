"""Classic (pre-AlphaGo) MCTS: UCT plus semi-random rollouts.

Included as the "what search alone gets you" control: the same family of
algorithm as AlphaZero, minus the learned policy and value.

A uniform playout is worse evidence here than in any other game in this
project, and for a reason specific to pieces that move.  Every other game adds
marks, so a random playout always terminates and usually terminates *somewhere*
-- the board fills. Here two random armies shuffle: they walk past captures,
walk past the enemy base, and a large share of playouts reach the 200-ply
no-capture draw having decided nothing at all.  Backing that up as 0 tells the
tree that every line is level, which is exactly the wrong answer.

So the playout carries three cheap pieces of knowledge, and a fourth guard:

* it **wins when it can** -- steps into the enemy base rather than past it;
* it **defends when it must** -- takes or blocks a piece standing next to its
  own base, which is the one threat that ends the game next move;
* it **prefers captures**, weighted by the matchup, because walking past a free
  piece is what a uniform player does most often and most expensively;
* and it is **cut off after a fixed number of plies**, at which point the
  static evaluation is backed up instead of a fabricated draw.  A truncated
  rollout with a real evaluation is a far better estimate than a full one that
  wandered into stagnation.
"""

from __future__ import annotations

import time

import numpy as np

from ..heuristics import (base_attackers, captures, evaluate, immediate_wins,
                          piece_value, _counts)
from ..rps2_game import (BASE_OF, NCELLS, STEP_LIST, RPS2Board, beats,
                         kind_of, other, split_action)
from .base import Agent

# Plies a playout runs before the static evaluation takes over.  Long enough to
# resolve the tactics around a threat, far short of the 200-ply draw.
ROLLOUT_DEPTH = 40


def random_rollout(board: RPS2Board, rng: np.random.Generator,
                   depth: int = ROLLOUT_DEPTH) -> float:
    """Play the position out semi-randomly; +1 if the side to move wins."""
    state = board.copy()
    me = state.to_move
    for _ in range(depth):
        if state.is_terminal():
            break
        mover = state.to_move
        wins = immediate_wins(state, mover)
        if wins:
            state.play(wins[0])
            continue
        move = _defence(state, mover)
        if move is None:
            move = _greedy_or_random(state, mover, rng)
        state.play(move)
    if not state.is_terminal():
        # Cut short: hand back what the position is worth rather than a draw
        # that nobody played for.
        return evaluate(state, me)
    if state.winner == 0:
        return 0.0
    return 1.0 if state.winner == me else -1.0


def _defence(state: RPS2Board, mover: int) -> int | None:
    """Take or block a piece that would enter our base next move."""
    intruders = base_attackers(state, other(mover))
    if not intruders:
        return None
    home = BASE_OF[mover]
    cells = state.cells
    legal = [int(m) for m in state.legal_moves()]
    if len(intruders) == 1:
        target = intruders[0]
        for m in legal:
            if STEP_LIST[m % NCELLS][m // NCELLS] == target:
                return m
    threats = [kind_of(cells[s]) for s in intruders]
    for m in legal:
        src, d = split_action(m)
        if STEP_LIST[src][d] != home:
            continue
        if all(not beats(t, kind_of(cells[src])) for t in threats):
            return m
    return None


def _greedy_or_random(state: RPS2Board, mover: int,
                      rng: np.random.Generator) -> int:
    """A capture two times in three, otherwise anything legal.

    Not always a capture: a playout that never declines one is a different game
    from the one being searched, and the tree would learn the wrong thing about
    positions where waiting is right.
    """
    takes = captures(state, mover)
    if takes and rng.random() < 0.67:
        ours = _counts(state, mover)
        values = np.array([
            piece_value(kind_of(state.cells[STEP_LIST[m % NCELLS][m // NCELLS]]),
                        ours)
            for m in takes
        ])
        return int(takes[int(np.argmax(values))])
    legal = state.legal_moves()
    return int(legal[rng.integers(len(legal))])


class _Node:
    __slots__ = ("moves", "N", "W", "child", "terminal")

    def __init__(self, moves: np.ndarray):
        self.moves = moves
        self.N = np.zeros(len(moves), dtype=np.float64)
        self.W = np.zeros(len(moves), dtype=np.float64)
        self.child: list[_Node | None] = [None] * len(moves)
        self.terminal = np.zeros(len(moves), dtype=bool)


class RolloutMCTSAgent(Agent):
    def __init__(
        self,
        simulations: int = 5000,
        time_budget: float | None = None,
        c_uct: float = 1.0,
        rollouts_per_leaf: int = 1,
        seed: int | None = None,
        name: str | None = None,
    ):
        self.simulations = simulations
        self.time_budget = time_budget
        self.c_uct = c_uct
        self.rollouts_per_leaf = rollouts_per_leaf
        self.rng = np.random.default_rng(seed)
        budget = f"{time_budget}s" if time_budget else f"{simulations}sims"
        self.name = name or f"mcts-rollout({budget})"

    def select_move(self, board: RPS2Board, last_move: int | None = None) -> int:
        me = board.to_move
        wins = immediate_wins(board, me)
        if wins:
            return wins[0]

        root = _Node(board.legal_moves())
        deadline = time.time() + self.time_budget if self.time_budget else None
        sims = 0
        while sims < self.simulations:
            if deadline is not None and time.time() >= deadline:
                break
            self._simulate(board, root)
            sims += 1
            if deadline is not None and sims % 32 == 0 and time.time() >= deadline:
                break
        return int(root.moves[int(np.argmax(root.N))])

    def _simulate(self, root_board: RPS2Board, root: _Node) -> None:
        state = root_board.copy()
        path: list[tuple[_Node, int]] = []
        node = root
        while True:
            a = self._uct_select(node)
            move = int(node.moves[a])
            path.append((node, a))
            state.play(move)
            if state.is_terminal():
                node.terminal[a] = True
                # Whoever is on turn has lost -- or nobody has, on a position
                # that ran out of captures.
                value = state.terminal_value()
                break
            nxt = node.child[a]
            if nxt is None:
                node.child[a] = _Node(state.legal_moves())
                value = float(
                    np.mean([random_rollout(state, self.rng)
                             for _ in range(self.rollouts_per_leaf)])
                )
                break
            node = nxt
        self._backup(path, value)

    def _uct_select(self, node: _Node) -> int:
        N = node.N
        total = N.sum()
        unvisited = np.flatnonzero(N == 0)
        if len(unvisited):
            return int(unvisited[self.rng.integers(len(unvisited))])
        q = node.W / N
        u = self.c_uct * np.sqrt(2.0 * np.log(total) / N)
        return int(np.argmax(q + u))

    @staticmethod
    def _backup(path, value: float) -> None:
        v = value
        for node, a in reversed(path):
            v = -v
            node.N[a] += 1.0
            node.W[a] += v
