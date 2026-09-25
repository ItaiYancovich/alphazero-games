"""What a backgammon agent looks like.

Deliberately *not* the shared :class:`alphazero_core.agents.Agent`: that one
returns an ``int`` because a move in Hex or Connect Four is a cell.  Here a move
is a tuple of hops chosen from a list that changes with every roll, so the
signature has to differ -- and pretending otherwise would push the difference
somewhere less obvious.
"""

from __future__ import annotations

import numpy as np

from ..bg_game import Backgammon

Move = tuple[tuple[int, int], ...]


class Agent:
    """Anything that can pick a backgammon move for the current roll."""

    name: str = "agent"

    def reset(self) -> None:
        """Called at the start of every game."""

    def select_move(self, board: Backgammon) -> Move:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{self.name}>"


class RandomAgent(Agent):
    """Uniform over legal moves -- the floor of the rating scale."""

    def __init__(self, seed: int | None = None, name: str = "random"):
        self.rng = np.random.default_rng(seed)
        self.name = name

    def select_move(self, board: Backgammon) -> Move:
        moves = board.legal_moves()
        return moves[int(self.rng.integers(len(moves)))]
