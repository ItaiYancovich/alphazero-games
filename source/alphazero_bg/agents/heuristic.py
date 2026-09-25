"""A hand-written backgammon player: no network, no search past this roll.

It scores each position the roll can reach and takes the best.  The terms are
the ones a beginner is taught, in roughly the order they are taught:

* **Race** -- the pip count difference.  In a position with no contact left it
  is the whole story, and it is never irrelevant.
* **Blots** -- a lone checker can be hit, and the cost of that is roughly how
  far it would have to come back, weighted by how easy it is to reach.
* **Points made** -- two checkers on a point deny it to the opponent; points in
  your own home board and the ones just outside it are worth the most.
* **The bar** -- an opponent on the bar is not playing, and a checker of your
  own on the bar is 25 pips from home.
* **Bearing off** -- once the race is over, checkers off are all that count.

It is not strong.  It is here for the same reason the rule-based Hex and
Connect Four players are: a fixed, understandable reference that a network can
be measured against while it learns, and a floor above random for the rating
ladder.
"""

from __future__ import annotations

import numpy as np

from ..bg_game import BLACK, CHECKERS, POINTS, WHITE, Backgammon, other
from .base import Agent, Move

# All in pips, so the terms are commensurable with the race.
BLOT_COST = 0.55        # per pip of exposure, scaled by the chance of a hit
POINT_VALUE = 3.0       # for making any point
HOME_BONUS = 2.5        # extra, per point, inside the home board
BAR_VALUE = 8.0         # per enemy checker on the bar
OFF_VALUE = 4.0         # per checker borne off


def _hit_chance(distance: int) -> float:
    """Roughly how often a blot ``distance`` pips away is hit, on an open board.

    The exact numbers depend on what is in between; this is the standard shape
    -- direct shots (1-6) are common, and the odds fall away sharply past that.
    """
    table = {1: 11 / 36, 2: 12 / 36, 3: 14 / 36, 4: 15 / 36, 5: 15 / 36,
             6: 17 / 36, 7: 6 / 36, 8: 6 / 36, 9: 5 / 36, 10: 3 / 36,
             11: 2 / 36, 12: 3 / 36}
    return table.get(distance, 1 / 36 if distance <= 24 else 0.0)


def evaluate(board: Backgammon, player: int) -> float:
    """Score a position for ``player``; positive is good.  Units are pips."""
    opponent = other(player)
    score = float(board.pip_count(opponent) - board.pip_count(player))

    score += OFF_VALUE * (board.off[player] - board.off[opponent])
    score += BAR_VALUE * (board.bar[opponent] - board.bar[player])

    home = board.home_range(player)
    for point in range(POINTS):
        mine = board.count_on(point, player)
        if mine >= 2:
            score += POINT_VALUE + (HOME_BONUS if point in home else 0.0)
        elif mine == 1:
            # How far back this checker would be sent, and how likely that is.
            back = point + 1 if player == BLACK else POINTS - point
            distance = _nearest_attacker(board, point, player)
            if distance is not None:
                score -= BLOT_COST * back * _hit_chance(distance)
    return score


def _nearest_attacker(board: Backgammon, point: int, player: int) -> int | None:
    """Pips from the closest enemy checker that could hit this blot."""
    opponent = other(player)
    # The opponent moves towards this point from the direction it came from.
    candidates = (range(point + 1, POINTS) if opponent == BLACK
                  else range(point - 1, -1, -1))
    if board.bar[opponent]:
        entry = 24 if opponent == BLACK else -1
        return abs(entry - point)
    for other_point in candidates:
        if board.count_on(other_point, opponent):
            return abs(other_point - point)
    return None


class HeuristicAgent(Agent):
    """Greedy over this roll's moves, by the score above."""

    def __init__(self, seed: int | None = None, noise: float = 0.0,
                 name: str = "heuristic"):
        self.rng = np.random.default_rng(seed)
        self.noise = noise  # small jitter, so repeated games are not identical
        self.name = name

    def select_move(self, board: Backgammon) -> Move:
        moves = board.legal_moves()
        if len(moves) == 1:
            return moves[0]
        player = board.to_move
        scores = np.empty(len(moves), dtype=np.float64)
        for i, move in enumerate(moves):
            after = board.after(move)
            scores[i] = (1e6 if after.winner == player
                         else evaluate(after, player))
        if self.noise > 0:
            scores += self.rng.normal(0.0, self.noise, len(scores))
        return moves[int(np.argmax(scores))]
