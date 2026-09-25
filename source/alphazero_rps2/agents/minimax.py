"""Brute-force Intransitive: negamax with alpha-beta, on a clock.

The classical answer to the game, and a weaker one here than in Connect Four
for reasons that are worth stating, because they are the reasons this game is
hard for a search at all:

* **The branching factor is the problem, not the depth.**  Ten pieces with
  eight directions each is up to eighty moves and typically thirty-five or
  forty -- five times Connect Four's seven -- so a ply costs five times as much
  and the tree is nowhere near as deep for the same clock.
* **There is no quiet position.**  Every piece can move every turn and nothing
  is ever pinned in place, so the horizon is a real one: the evaluation at the
  leaf is a guess about a position where four things are half-captured.
* **Transpositions are everywhere and mostly useless.**  Pieces move, so the
  same arrangement recurs constantly by different routes -- but the stagnation
  counter is part of the position, so two arrivals at the same squares are only
  the same node if the same number of quiet plies got them there.

What earns its keep, roughly in order of how much it is worth:

* **Alpha-beta with a transposition table**, whose stored best move also orders
  the next iteration's search.
* **Move ordering by the static score.**  With forty moves to a node, ordering
  is most of what makes the cutoffs happen -- far more so than in a game with
  seven.  Captures and base runs come first because those are the moves that
  refute something.
* **Iterative deepening**, with the clock as the budget rather than a depth.
* **A one-ply blunder filter near the root**: moves after which the opponent
  simply walks into our base are searched last, because they are almost never
  the answer and they are cheap to spot.

Mate scores carry the ply they were found at, so the search prefers the
*quickest* win and the *slowest* loss instead of treating all of them alike.
"""

from __future__ import annotations

import time

import numpy as np

from ..heuristics import (evaluate, gives_opponent_win, immediate_wins,
                          move_scores)
from ..rps2_game import RPS2Board, other
from .base import Agent

WIN_SCORE = 1e6      # a win now; each ply of delay costs one point
EXACT, LOWER, UPPER = 0, 1, 2


class _TimeUp(Exception):
    """Raised deep in the search when the clock runs out."""


class MinimaxAgent(Agent):
    def __init__(
        self,
        time_budget: float = 1.0,
        max_depth: int = 40,
        seed: int | None = None,
        name: str | None = None,
    ):
        self.time_budget = time_budget
        self.max_depth = max_depth
        self.rng = np.random.default_rng(seed)
        self.name = name or f"minimax({time_budget:g}s)"
        self.table: dict[int, tuple[int, int, float, int]] = {}
        self.nodes = 0
        self.last_depth = 0
        self.last_value = 0.0
        self._deadline: float | None = None

    def reset(self) -> None:
        self.table.clear()

    # ------------------------------------------------------------- interface
    def select_move(self, board: RPS2Board, last_move: int | None = None) -> int:
        me = board.to_move
        wins = immediate_wins(board, me)
        if wins:
            self.last_value, self.last_depth = 1.0, 1
            return wins[0]

        self.nodes = 0
        self._deadline = time.time() + self.time_budget if self.time_budget else None
        legal = [int(m) for m in board.legal_moves()]
        if not legal:
            raise ValueError("no legal move in a position that is not terminal")
        best = self._ordered(board, legal, None, me, self.max_depth)[0]
        for depth in range(1, self.max_depth + 1):
            try:
                score, move = self._negamax(board, depth, -np.inf, np.inf)
            except _TimeUp:
                break
            if move is not None:
                best = move
            self.last_depth = depth
            self.last_value = _display_value(score)
            if abs(score) > WIN_SCORE / 2:
                break  # the result is settled; deeper cannot change it
        return best

    # --------------------------------------------------------------- search
    def _negamax(self, board: RPS2Board, depth: int, alpha: float, beta: float
                 ) -> tuple[float, int | None]:
        """Score of ``board`` for the side to move, plus its best move."""
        self.nodes += 1
        if self._deadline is not None and (self.nodes & 127) == 0:
            if time.time() > self._deadline:
                raise _TimeUp

        me = board.to_move
        wins = immediate_wins(board, me)
        if wins:
            # Win found: score it by *when*, so a faster win beats a slower one.
            return WIN_SCORE - board.move_count, wins[0]

        legal = [int(m) for m in board.legal_moves()]
        if not legal:
            # No move is a loss here, not a draw -- and it is scored by when,
            # so the search will prefer to be stuck later rather than sooner.
            return -(WIN_SCORE - board.move_count), None
        if depth <= 0:
            return evaluate(board, me), None

        alpha0 = alpha
        key = board.key()
        hit = self.table.get(key)
        tt_move = None
        if hit is not None:
            hit_depth, flag, score, tt_move = hit
            if hit_depth >= depth:
                if flag == EXACT:
                    return score, tt_move
                if flag == LOWER:
                    alpha = max(alpha, score)
                elif flag == UPPER:
                    beta = min(beta, score)
                if alpha >= beta:
                    return score, tt_move

        best_score = -np.inf
        best_move = legal[0]
        for move in self._ordered(board, legal, tt_move, me, depth):
            child = board.copy()
            child.play(move)
            if child.is_terminal():
                # A winning move was already taken above, so a terminal child
                # here is the draw or the opponent being left with no move --
                # and ``terminal_value`` is from the child's point of view.
                score = -child.terminal_value()
            else:
                score = -self._negamax(child, depth - 1, -beta, -alpha)[0]
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, score)
            if alpha >= beta:
                break  # this branch is already worse than one the opponent has

        flag = EXACT
        if best_score <= alpha0:
            flag = UPPER
        elif best_score >= beta:
            flag = LOWER
        self.table[key] = (depth, flag, best_score, best_move)
        return best_score, best_move

    # Remaining depth at which the one-ply blunder check still pays for itself.
    # It costs a board copy per candidate, so it runs near the root where there
    # are few nodes and getting the order right saves whole subtrees.
    SAFETY_DEPTH = 3
    # How many of the top-scoring moves get that check.  A blunder is nearly
    # always among the moves that looked attractive; the tail is already last.
    SAFETY_WIDTH = 10

    def _ordered(self, board: RPS2Board, legal: list[int], tt_move: int | None,
                 me: int, depth: int) -> list[int]:
        """Best guess first: table move, then the static score, blunders last."""
        scores = move_scores(board, me)
        ranked = sorted(legal, key=lambda m: -scores[m])
        if depth >= self.SAFETY_DEPTH:
            losing = {m for m in ranked[:self.SAFETY_WIDTH]
                      if gives_opponent_win(board, m, me)}
            if losing:
                ranked = ([m for m in ranked if m not in losing]
                          + [m for m in ranked if m in losing])
        if tt_move is not None and tt_move in legal:
            ranked = [tt_move] + [m for m in ranked if m != tt_move]
        return ranked


def _display_value(score: float) -> float:
    """Search score -> the -1..+1 number the GUI shows next to the move."""
    if score > WIN_SCORE / 2:
        return 1.0
    if score < -WIN_SCORE / 2:
        return -1.0
    # ``evaluate`` already returns a -1..+1 number, so the leaf scores that
    # reach here need no further squashing.
    return float(np.clip(score, -1.0, 1.0))
