"""The knowledge-driven Splendor player: one ply of afterstate evaluation.

It looks at every legal move, plays it, scores the position it reaches with
:func:`alphazero_splendor.heuristics.position_value`, and takes the best.  No
tree: this is the rung of the ladder that says what pure game knowledge is
worth before any search or any network is involved, the way the rule-based Hex
agent does.

Like every agent here it evaluates a *determinization* rather than the server's
own position, so a card it has not been shown cannot influence its choice.  That
matters more than it sounds: a greedy player that could read the deck would
reserve exactly the right card every time, and the ladder would be measuring the
peek rather than the knowledge.
"""

from __future__ import annotations

import numpy as np

from alphazero_core.vecagents import VecAgent

from ..heuristics import position_value
from ..splendor_game import SplendorGame


class HeuristicAgent(VecAgent):
    """Greedy over one move, by a hand-written evaluation.

    ``noise`` is a standard deviation in points-equivalent added to each
    candidate's score, so the agent varies its play between games instead of
    replaying one line for ever.  The ladder rates it with the same noise it
    plays with, so the number it earns is the number you face.
    """

    def __init__(self, seed: int | None = None, noise: float = 0.15,
                 name: str = "heuristic"):
        self.rng = np.random.default_rng(seed)
        self.noise = float(noise)
        self.name = name
        self.last_value = 0.0

    def select_move(self, board: SplendorGame, last_move: int | None = None) -> int:
        seat = board.to_move
        view = board.determinize(seat, self.rng)
        legal = view.legal_moves()
        scores = np.empty(len(legal), dtype=np.float64)
        for i, move in enumerate(legal):
            after = view.copy()
            after.play(int(move))
            scores[i] = position_value(after, seat)
        if self.noise > 0:
            scores += self.rng.normal(0.0, self.noise, size=len(scores))
        best = int(np.argmax(scores))
        self.last_value = float(scores[best])
        return int(legal[best])
