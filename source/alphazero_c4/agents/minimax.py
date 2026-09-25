"""Brute-force Connect Four: negamax with alpha-beta, on a clock.

The classical answer to this game, and a strong one -- 6x7 Connect Four is
solved, and while a pure-Python search will not solve it from the opening
inside a second, it does see far enough to punish anything tactical.

What earns its keep here, roughly in order of how much it is worth:

* **Alpha-beta with a transposition table.**  Connect Four transposes heavily
  -- the same position arrives by many move orders -- so the table saves a
  large fraction of the tree, and its stored best move also orders the search.
* **Centre-first move ordering.**  The middle column takes part in 13 of the 69
  four-in-a-row windows and the edge column in 3, so the best move is usually
  central and searching it first is what makes the cutoffs happen.
* **Iterative deepening.**  The clock, not a depth, is the budget: deepen until
  time runs out and play the deepest completed answer.  Deepening is nearly
  free because each pass fills the table for the next.
* **A one-ply safety check** in the move ordering: a drop that lets the
  opponent win on the square above it is searched last.

Mate scores carry the ply they were found at, so the search prefers the
*quickest* win and the *slowest* loss instead of treating all of them alike --
without that, a won game looks the same to it as a game won ten moves later,
and a lost one produces an arbitrary move.
"""

from __future__ import annotations

import time

import numpy as np

from ..c4_game import Connect4Board, other
from ..heuristics import centre_order, evaluate, gives_opponent_win, immediate_wins
from .base import Agent

WIN_SCORE = 1e6      # a win now; each ply of delay costs one point
EXACT, LOWER, UPPER = 0, 1, 2


class _TimeUp(Exception):
    """Raised deep in the search when the clock runs out."""


class MinimaxAgent(Agent):
    def __init__(
        self,
        time_budget: float = 1.0,
        max_depth: int = 42,
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
    def select_move(self, board: Connect4Board, last_move: int | None = None) -> int:
        me = board.to_move
        wins = immediate_wins(board, me)
        if wins:
            self.last_value, self.last_depth = 1.0, 1
            return wins[0]
        threats = immediate_wins(board, other(me))
        if len(threats) == 1:
            # Forced: searching a position with one legal answer is a waste of
            # the clock, and with several it is lost anyway.
            self.last_depth = 1
            return threats[0]

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
    def _negamax(self, board: Connect4Board, depth: int, alpha: float, beta: float
                 ) -> tuple[float, int | None]:
        """Score of ``board`` for the side to move, plus its best move."""
        self.nodes += 1
        if self._deadline is not None and (self.nodes & 511) == 0:
            if time.time() > self._deadline:
                raise _TimeUp

        me = board.to_move
        wins = immediate_wins(board, me)
        if wins:
            # Win found: score it by *when*, so a faster win beats a slower one.
            return WIN_SCORE - board.move_count, wins[0]

        legal = [int(m) for m in board.legal_moves()]
        if not legal:
            return 0.0, None  # a full board with no four: a draw, worth nothing
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
            # The child can only be terminal by filling the board: a winning
            # drop was already taken above.
            score = 0.0 if child.is_terminal() else -self._negamax(
                child, depth - 1, -beta, -alpha)[0]
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

    def _ordered(self, board: Connect4Board, legal: list[int], tt_move: int | None,
                 me: int) -> list[int]:
        """Best guess first: table move, then centre-out, losing drops last."""
        order = centre_order(board.cols)
        rank = {c: i for i, c in enumerate(order)}
        losing = {m for m in legal if gives_opponent_win(board, m, me)}

        def key(move: int) -> tuple[int, int, int]:
            return (move in losing, 0 if move == tt_move else 1,
                    rank[move % board.cols])

        return sorted(legal, key=key)


def _display_value(score: float) -> float:
    """Search score -> the -1..+1 number the GUI shows next to the move."""
    if score > WIN_SCORE / 2:
        return 1.0
    if score < -WIN_SCORE / 2:
        return -1.0
    return float(np.tanh(score / 60.0))
