"""Brute-force Ultimate XX: negamax with alpha-beta, on a clock.

The classical answer to the game.  It fits this game better than it fits
Ultimate Tic-Tac-Toe: the tree is a good deal shorter (forty-odd plies against
sixty, because a small board here closes as soon as a line appears rather than
when it fills), the branching factor is the same seven or eight cells the
sending rule leaves open, and the tactics are sharp -- most positions have a
forced answer a few plies down rather than a slow accumulation.

What earns its keep, roughly in order of how much it is worth:

* **Alpha-beta with a transposition table.**  Positions transpose readily once
  the constraint has bounced play around the grid.  The table's stored best
  move also orders the next iteration's search.
* **Move ordering.**  A move that loses on the spot goes last, then the quiet
  squares before the busy ones -- ordering is most of what makes the cutoffs
  happen, and the losing-move test here is pure integer arithmetic.
* **Iterative deepening.**  The clock, not a depth, is the budget: deepen until
  time runs out and play the deepest completed answer.  Deepening is nearly
  free because each pass fills the table for the next.

Mate scores carry the ply they were found at, so the search prefers the
*quickest* win and the *slowest* loss instead of treating all of them alike.

The one structural difference from every other minimax in this project is at the
terminal node.  Elsewhere a finished position means whoever is on turn has lost;
here it means the opposite, because you can only lose on your own move.  So the
terminal score is read off ``terminal_value()`` and negated, rather than assumed.
"""

from __future__ import annotations

import time

import numpy as np

from ..heuristics import STATIC_RANK, evaluate, forced_wins
from ..uxx_game import UltimateXXBoard
from .base import Agent

WIN_SCORE = 1e6      # a win now; each ply of delay costs one point
EXACT, LOWER, UPPER = 0, 1, 2


class _TimeUp(Exception):
    """Raised deep in the search when the clock runs out."""


class MinimaxAgent(Agent):
    def __init__(
        self,
        time_budget: float = 1.0,
        max_depth: int = 81,
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
    def select_move(self, board: UltimateXXBoard, last_move: int | None = None) -> int:
        wins = forced_wins(board, board.to_move)
        if wins:
            self.last_value, self.last_depth = 1.0, 2
            return wins[0]

        self.nodes = 0
        self._deadline = time.time() + self.time_budget if self.time_budget else None
        legal = [int(m) for m in board.legal_moves()]
        best = legal[0]
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
            if depth >= board.ncells - board.move_count:
                break  # nothing left to deepen into: the tree is exhausted
        return best

    # --------------------------------------------------------------- search
    def _negamax(self, board: UltimateXXBoard, depth: int, alpha: float, beta: float
                 ) -> tuple[float, int | None]:
        """Score of ``board`` for the side to move, plus its best move."""
        self.nodes += 1
        if self._deadline is not None and (self.nodes & 255) == 0:
            if time.time() > self._deadline:
                raise _TimeUp

        me = board.to_move
        legal = [int(m) for m in board.legal_moves()]
        if not legal:
            return 0.0, None  # every board claimed with no line: a draw
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
        for move in self._ordered(board, legal, tt_move, me):
            child = board.copy()
            child.play(move)
            if child.is_terminal():
                # The child's value is for *its* side to move, and here that is
                # +1 when the move just played lost the game -- so negating it
                # gives this node the loss, and a draw stays a draw.
                tv = child.terminal_value()
                score = 0.0 if tv == 0.0 else -tv * (WIN_SCORE - child.move_count)
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

    def _ordered(self, board: UltimateXXBoard, legal: list[int], tt_move: int | None,
                 me: int) -> list[int]:
        """Best guess first: table move, then the quiet squares, losers last.

        Unlike the Ultimate Tic-Tac-Toe version this needs no depth gate: "does
        this move complete my line of small boards" is two table lookups and no
        board copy, so it is worth asking at every node.
        """
        def key(move: int) -> tuple[int, int, int]:
            return (board.would_lose(me, move),
                    0 if move == tt_move else 1,
                    STATIC_RANK[move])

        return sorted(legal, key=key)


def _display_value(score: float) -> float:
    """Search score -> the -1..+1 number the GUI shows next to the move."""
    if score > WIN_SCORE / 2:
        return 1.0
    if score < -WIN_SCORE / 2:
        return -1.0
    return float(np.tanh(score / 120.0))
