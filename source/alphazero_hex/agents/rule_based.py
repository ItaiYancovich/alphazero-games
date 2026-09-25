"""Hand-written, knowledge-driven Hex player (no search, no network).

Decision order -- each rule is a hard override of the ones below it:

1. Win now if a winning cell exists.
2. If the opponent has exactly one winning cell, take it.
3. If the opponent just intruded into one of my bridges, restore it.
4. Otherwise play the cell that most improves my two-distance connection
   while most damaging theirs.

Rules 1-3 are the standard Hex "must-play" responses; rule 4 is Anshelevich's
two-distance evaluation.  Together they make a player that beats random
essentially always and punishes any agent that has not learned bridges.
"""

from __future__ import annotations

import numpy as np

from ..hex_game import BLACK, WHITE, HexBoard
from ..heuristics import broken_bridge_response, immediate_wins, move_scores
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

    def select_move(self, board: HexBoard, last_move: int | None = None) -> int:
        me = board.to_move
        opp = WHITE if me == BLACK else BLACK

        wins = immediate_wins(board, me)
        if wins:
            return wins[0]

        threats = immediate_wins(board, opp)
        if len(threats) == 1:
            return threats[0]

        save = broken_bridge_response(board, me, last_move)
        if save is not None:
            return save

        scores = move_scores(board, me, self.defence_weight)
        if self.noise > 0:
            finite = np.isfinite(scores)
            scores = scores.copy()
            scores[finite] += self.rng.normal(0.0, self.noise, finite.sum())
        return int(np.argmax(scores))
