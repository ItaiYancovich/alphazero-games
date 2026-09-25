"""Hand-written, knowledge-driven Connect Four player (no search, no network).

Decision order -- each rule is a hard override of the ones below it:

1. Win now if a winning drop exists.
2. Block the opponent's winning drop if they have one.
3. Otherwise play the best cell by :func:`~alphazero_c4.heuristics.move_scores`,
   which already refuses to hand the opponent the square directly above.

Rules 1 and 2 are what every human learns in their first three games; rule 3 is
a static window count with a centre bias.  Together they make a player that
beats random essentially always and punishes anything that has not learned to
look one ply ahead -- which is exactly the rung this occupies on the ladder.
"""

from __future__ import annotations

import numpy as np

from ..c4_game import Connect4Board, other
from ..heuristics import immediate_wins, move_scores
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

    def select_move(self, board: Connect4Board, last_move: int | None = None) -> int:
        me = board.to_move
        opp = other(me)

        wins = immediate_wins(board, me)
        if wins:
            return wins[0]

        threats = immediate_wins(board, opp)
        if threats:
            # More than one and the game is lost anyway; block one and play on.
            return threats[0]

        scores = move_scores(board, me, self.defence_weight)
        if self.noise > 0:
            finite = np.isfinite(scores)
            scores = scores.copy()
            scores[finite] += self.rng.normal(0.0, self.noise * 10.0, finite.sum())
        return int(np.argmax(scores))
