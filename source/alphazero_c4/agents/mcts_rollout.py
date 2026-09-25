"""Classic (pre-AlphaGo) MCTS: UCT plus uniformly random rollouts.

Included as the "what search alone gets you" control: the same family of
algorithm as AlphaZero, minus the learned policy and value.

Random playouts are weaker evidence in Connect Four than in Hex.  A filled Hex
board always has exactly one winner, so a random fill *is* a game result; here a
random playout usually walks past several wins neither side takes, and a real
draw and a random draw are not the same thing.  Two cheap corrections keep it
honest: the playout takes a win when it sees one and blocks when it must, and
drawn playouts back up 0 instead of being discarded.
"""

from __future__ import annotations

import time

import numpy as np

from ..c4_game import Connect4Board, other
from ..heuristics import immediate_wins
from .base import Agent


def random_rollout(board: Connect4Board, rng: np.random.Generator) -> float:
    """Play the position out semi-randomly; +1 if the side to move wins.

    "Semi" is the win/block filter: a purely random playout in Connect Four
    steps over a four-in-a-row about as often as it takes one, and the noise
    that adds swamps the signal the tree is trying to collect.
    """
    state = board.copy()
    me = state.to_move
    while not state.is_terminal():
        mover = state.to_move
        wins = immediate_wins(state, mover)
        if wins:
            state.play(wins[0])
            continue
        threats = immediate_wins(state, other(mover))
        if threats:
            state.play(threats[0])
            continue
        legal = state.legal_moves()
        state.play(int(legal[rng.integers(len(legal))]))
    if state.winner == 0:
        return 0.0
    return 1.0 if state.winner == me else -1.0


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

    def select_move(self, board: Connect4Board, last_move: int | None = None) -> int:
        me = board.to_move
        wins = immediate_wins(board, me)
        if wins:
            return wins[0]
        threats = immediate_wins(board, other(me))
        if len(threats) == 1:
            return threats[0]

        root = _Node(board.legal_moves())
        deadline = time.time() + self.time_budget if self.time_budget else None
        sims = 0
        while sims < self.simulations:
            if deadline is not None and time.time() >= deadline:
                break
            self._simulate(board, root)
            sims += 1
            if deadline is not None and sims % 64 == 0 and time.time() >= deadline:
                break
        return int(root.moves[int(np.argmax(root.N))])

    def _simulate(self, root_board: Connect4Board, root: _Node) -> None:
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
                # Whoever is on turn has lost -- or nobody has, on a full board.
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
