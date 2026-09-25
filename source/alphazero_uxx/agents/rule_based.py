"""Hand-written Ultimate XX player (no search, no network).

Decision order -- each rule is a hard override of the ones below it:

1. If a move leaves the opponent with nothing safe to play, make it and win.
2. Never play a move that completes our own line of small boards.
3. Otherwise play the best cell by
   :func:`~alphazero_uxx.heuristics.move_scores`, which reads the position the
   move leaves behind: how near each side is to a third board, how many
   harmless squares the opponent will have, whether they end up free to choose,
   and how many live boards each side must now stay out of.

Rules 1 and 2 are what a player works out in their first game, and they are
already enough to beat random essentially always -- a uniform player walks into
its own losing square about as readily as anywhere else.  Rule 3 is the piece
that separates a player who has understood the game from one playing nine
independent games of noughts and crosses: the marks are shared, so a small board
is not something to win but something to avoid being cornered in.
"""

from __future__ import annotations

import numpy as np

from ..heuristics import forced_wins, move_scores
from ..uxx_game import UltimateXXBoard
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

    def select_move(self, board: UltimateXXBoard, last_move: int | None = None) -> int:
        wins = forced_wins(board, board.to_move)
        if wins:
            return wins[0]

        # ``move_scores`` gives a losing move -1e9, so rule 2 needs no separate
        # pass; the jitter below is far too small to reach across that gap.
        scores = move_scores(board, board.to_move, self.defence_weight)
        if self.noise > 0:
            finite = np.isfinite(scores)
            scores = scores.copy()
            scores[finite] += self.rng.normal(0.0, self.noise * 10.0, finite.sum())
        return int(np.argmax(scores))
