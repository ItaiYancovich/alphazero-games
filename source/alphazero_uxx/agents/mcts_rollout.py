"""Classic (pre-AlphaGo) MCTS: UCT plus semi-random rollouts.

Included as the "what search alone gets you" control: the same family of
algorithm as AlphaZero, minus the learned policy and value.

A uniform playout is close to worthless here.  A random player steps on its own
losing square as readily as on any other, so a pure rollout measures very little
beyond who happened to be on turn when the board ran out of quiet squares.  One
correction fixes most of that and costs nothing: **the playout never plays a
move that loses at once unless every move does.**  That is two table lookups per
candidate and no board copy, and it turns the playout from noise into a rough
model of two players who can at least see the end of their own nose.

Deliberately *not* corrected: the playout does not look for the forced win --
sending the opponent somewhere every square loses.  Finding it costs a board
copy per candidate move per ply, which is an order of magnitude more work than
the rest of the playout put together, and the tree above finds it a ply later
anyway.  The agents' root decision does check for it, where it is cheap and
where getting it wrong throws away a won game.
"""

from __future__ import annotations

import time

import numpy as np

from ..heuristics import forced_wins
from ..uxx_game import UltimateXXBoard
from .base import Agent


def random_rollout(board: UltimateXXBoard, rng: np.random.Generator) -> float:
    """Play the position out semi-randomly; +1 if the side to move wins."""
    state = board.copy()
    me = state.to_move
    while not state.is_terminal():
        # ``safe_moves`` falls back to every legal move when they all lose,
        # which is a position that has to play *something*.
        options = state.safe_moves() or [int(m) for m in state.legal_moves()]
        state.play(options[rng.integers(len(options))])
    if state.loser == 0:
        return 0.0
    return -1.0 if state.loser == me else 1.0


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

    def select_move(self, board: UltimateXXBoard, last_move: int | None = None) -> int:
        wins = forced_wins(board, board.to_move)
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
            if deadline is not None and sims % 64 == 0 and time.time() >= deadline:
                break
        return int(root.moves[int(np.argmax(root.N))])

    def _simulate(self, root_board: UltimateXXBoard, root: _Node) -> None:
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
                # Whoever is on turn has *won* -- the mark just played completed
                # a line of small boards and lost it -- or nobody has, on a grid
                # where all nine boards are claimed and no three line up.
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
