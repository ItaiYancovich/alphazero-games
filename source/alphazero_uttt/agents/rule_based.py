"""Hand-written Ultimate Tic-Tac-Toe player (no search, no network).

Decision order -- each rule is a hard override of the ones below it:

1. Win now if a move takes the third small board of a line.
2. Block the opponent's winning move, if it happens to be one this turn's
   constraint lets us reach.
3. Otherwise play the best cell by :func:`~alphazero_uttt.heuristics.move_scores`,
   which already refuses to hand the opponent a win on the reply and charges a
   move for sending them somewhere they may choose freely.

Rules 1 and 2 are what a human works out in their first game.  Rule 3 is a
static two-level line count plus the free-choice penalty -- the one piece of
Ultimate-specific knowledge that separates a player who understands the game
from one who is playing nine independent games of noughts and crosses.  Together
they make an opponent that beats random essentially always and punishes anything
that has not learned to look one ply ahead, which is the rung this occupies on
the ladder.
"""

from __future__ import annotations

import numpy as np

from ..heuristics import immediate_wins, move_scores
from ..uttt_game import UltimateBoard, other
from .base import Agent


class RuleBasedAgent(Agent):
    def __init__(
        self,
        seed: int | None = None,
        defence_weight: float = 1.0,
        noise: float = 0.0,
        name: str = "rule-based",
    ):
        self.rng = np.random.default_rng(seed)
        self.defence_weight = defence_weight
        self.noise = noise  # small tie-breaking jitter, keeps games diverse
        self.name = name

    def select_move(self, board: UltimateBoard, last_move: int | None = None) -> int:
        me = board.to_move
        opp = other(me)

        wins = immediate_wins(board, me)
        if wins:
            return wins[0]

        # The opponent's winning cell is only blockable when it is inside the
        # board we have been sent to -- unlike Connect Four, where every threat
        # is always reachable.  ``immediate_wins`` already filters to our legal
        # moves, so asking it about the opponent asks exactly that question.
        threats = immediate_wins(board, opp)
        if threats:
            return threats[0]

        scores = move_scores(board, me, self.defence_weight)
        if self.noise > 0:
            finite = np.isfinite(scores)
            scores = scores.copy()
            scores[finite] += self.rng.normal(0.0, self.noise * 10.0, finite.sum())
        return int(np.argmax(scores))
