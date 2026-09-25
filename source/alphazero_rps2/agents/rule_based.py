"""Hand-written Intransitive player (no search, no network).

Decision order -- each rule is a hard override of the ones below it:

1. Win now: step into the enemy base.
2. Stop them winning now.  Unlike every other game here that means one of two
   different moves, because there is no such thing as blocking a king move:
   either **take the piece** that is standing next to our base, or **stand on
   the base ourselves** with something it cannot capture.  The second is the
   move a first-time player never finds, and it is what makes the rule-based
   agent an opponent rather than a target.
3. Otherwise the best move by :func:`~alphazero_rps2.heuristics.move_scores`,
   which prices a capture by the matchup rather than by the piece, rewards
   progress towards the far corner, refuses squares where something that beats
   us is waiting, and discards any move after which the opponent walks in.

Rules 1 and 2 are what a human works out in their first game.  Rule 3 is the
static evaluation, and the matchup pricing in it is the one piece of
Intransitive-specific knowledge that separates a player who understands the
game from one playing chess with funny capture rules.
"""

from __future__ import annotations

import numpy as np

from ..heuristics import base_attackers, immediate_wins, move_scores
from ..rps2_game import (BASE_OF, NCELLS, STEP_LIST, RPS2Board, beats,
                         kind_of, other, split_action)
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

    def select_move(self, board: RPS2Board, last_move: int | None = None) -> int:
        me = board.to_move
        opp = other(me)

        wins = immediate_wins(board, me)
        if wins:
            return wins[0]

        intruders = base_attackers(board, opp)
        if intruders:
            answer = self._defend(board, me, intruders)
            if answer is not None:
                return answer

        scores = move_scores(board, me)
        if self.defence_weight != 1.0:
            # The dial the ladder uses to make a weaker version of this agent:
            # below 1 it cares less about the squares it is walking into.
            scores = np.where(scores < 0, scores * self.defence_weight, scores)
        if self.noise > 0:
            finite = np.isfinite(scores)
            scores = scores.copy()
            scores[finite] += self.rng.normal(0.0, self.noise * 10.0, finite.sum())
        return int(np.argmax(scores))

    def _defend(self, board: RPS2Board, me: int, intruders: list[int]) -> int | None:
        """Answer pieces standing next to our base, or give up on answering.

        With one intruder both cures work; with two, taking one leaves the
        other, so only occupying the base can save the game -- and only with a
        piece that beats or ties every one of them, since a tie is a wall.
        """
        cells = board.cells
        home = BASE_OF[me]
        legal = [int(m) for m in board.legal_moves()]

        if len(intruders) == 1:
            target = intruders[0]
            takes = [m for m in legal
                     if STEP_LIST[m % NCELLS][m // NCELLS] == target]
            if takes:
                return takes[0]

        # Stand on the base with something none of them can take.  A piece that
        # merely ties is enough: equal types cannot capture each other.
        threats = [kind_of(cells[s]) for s in intruders]
        guards = []
        for m in legal:
            src, d = split_action(m)
            if STEP_LIST[src][d] != home:
                continue
            mine = kind_of(cells[src])
            if all(not beats(t, mine) for t in threats):
                guards.append(m)
        if guards:
            return guards[0]
        return None
