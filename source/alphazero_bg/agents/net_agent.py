"""The trained agent: afterstate evaluation with the value network.

One ply is the default and is what self-play uses -- it is a single batched
forward pass per turn, and the dice supply all the variety the training needs.
Two ply averages the opponent's reply over all 21 rolls and is markedly
stronger, at roughly twenty times the cost; it is the "with search" rung of the
ladder, the way ``az@400`` is for the other two games.

``explore`` mirrors what self-play does with temperature elsewhere: with a
positive value the agent sometimes takes a move that is not the best, which
keeps repeated games from following identical lines.  Left at 0 -- the default,
and what the ladder rates -- it is deterministic given the dice.
"""

from __future__ import annotations

import numpy as np

from ..bg_game import Backgammon
from ..evaluator import Evaluator
from .base import Agent, Move


class NetAgent(Agent):
    def __init__(
        self,
        evaluator: Evaluator,
        plies: int = 1,
        explore: float = 0.0,
        candidates: int = 6,
        seed: int | None = None,
        name: str | None = None,
    ):
        self.evaluator = evaluator
        self.plies = plies
        self.explore = explore
        self.candidates = candidates
        self.rng = np.random.default_rng(seed)
        self.name = name or f"net({plies}-ply)"
        self.last_value = 0.0

    def select_move(self, board: Backgammon) -> Move:
        if self.plies >= 2:
            moves, equities = self.evaluator.move_equities_two_ply(
                board, candidates=self.candidates)
        else:
            moves, equities = self.evaluator.move_equities(board)
        if not moves:
            return ()
        top = float(np.max(equities))
        tied = np.flatnonzero(equities >= top - 1e-6)
        best = int(tied[0] if len(tied) == 1 else self.rng.choice(tied))
        self.last_value = top
        if self.explore > 0 and len(moves) > 1 and self.rng.random() < self.explore:
            # Sample among the moves the network does not actively dislike,
            # rather than uniformly: a blunder teaches much less than a
            # plausible alternative.
            gap = top - equities
            plausible = np.flatnonzero(gap <= 0.08)
            pick = int(self.rng.choice(plausible)) if len(plausible) else best
            return moves[pick]
        return moves[best]
