"""Classic (pre-AlphaGo) MCTS: UCT plus uniformly random rollouts.

This is the strongest thing anyone had for Hex before neural networks, and it
is a genuinely awkward opponent -- random rollouts work unusually well in Hex
because a *filled* board always has exactly one winner, so a playout never
needs an evaluation function at all: fill the empty cells at random and read
off the result.

Included as the "what search alone gets you" control: same family of algorithm
as AlphaZero, minus the learned policy and value.
"""

from __future__ import annotations

import time

import numpy as np

from ..hex_game import BLACK, WHITE, HexBoard, winner_of_full_board
from ..heuristics import immediate_wins
from .base import Agent


def random_rollout(board: HexBoard, rng: np.random.Generator) -> float:
    """Fill the board at random; +1 if the side to move wins, else -1."""
    arr = board.array().reshape(-1).copy()
    empties = np.flatnonzero(arr == 0)
    if len(empties) == 0:
        w = winner_of_full_board(arr.reshape(board.n, board.n))
        return 1.0 if w == board.to_move else -1.0
    perm = rng.permutation(len(empties))
    me = board.to_move
    opp = WHITE if me == BLACK else BLACK
    arr[empties[perm[0::2]]] = me
    arr[empties[perm[1::2]]] = opp
    w = winner_of_full_board(arr.reshape(board.n, board.n))
    return 1.0 if w == me else -1.0


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

    def select_move(self, board: HexBoard, last_move: int | None = None) -> int:
        me = board.to_move
        opp = WHITE if me == BLACK else BLACK
        wins = immediate_wins(board, me)
        if wins:
            return wins[0]
        threats = immediate_wins(board, opp)
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

    def _simulate(self, root_board: HexBoard, root: _Node) -> None:
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
                value = -1.0  # side to move at `state` has lost
                break
            nxt = node.child[a]
            if nxt is None:
                node.child[a] = _Node(state.legal_moves())
                value = float(
                    np.mean([random_rollout(state, self.rng) for _ in range(self.rollouts_per_leaf)])
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
