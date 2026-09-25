"""Smart brute force: alpha-beta over a heuristically pruned move list.

Plain minimax is hopeless on 11x11 Hex -- the branching factor starts at 121
and the game lasts ~60 plies.  This agent is the classical answer to that:

* **negamax with alpha-beta** to cut the tree,
* **iterative deepening** under a wall-clock budget, so it always has a move,
* a **transposition table** keyed on the exact position,
* **candidate pruning**: only the top ``beam`` cells by two-distance score are
  searched, which is what turns brute force into *smart* brute force,
* **move ordering** by transposition-table hint then heuristic score, since
  alpha-beta's pruning power lives or dies on ordering,
* **forced-move detection** (win now / block the only threat) as a free
  extension at every node.
"""

from __future__ import annotations

import time

import numpy as np

from ..hex_game import BLACK, WHITE, HexBoard
from ..heuristics import evaluate, immediate_wins, move_scores
from .base import Agent

LOWER, EXACT, UPPER = -1, 0, 1


class MinimaxAgent(Agent):
    def __init__(
        self,
        time_budget: float = 1.0,
        beam: int = 8,
        max_depth: int = 12,
        seed: int | None = None,
        name: str | None = None,
    ):
        self.time_budget = time_budget
        self.beam = beam
        self.max_depth = max_depth
        self.rng = np.random.default_rng(seed)
        self.name = name or f"minimax(beam={beam},{time_budget}s)"
        self.tt: dict[tuple, tuple] = {}
        self.nodes = 0
        self.last_depth = 0

    def reset(self) -> None:
        self.tt.clear()

    # ------------------------------------------------------------------ API
    def select_move(self, board: HexBoard, last_move: int | None = None) -> int:
        deadline = time.time() + self.time_budget
        me = board.to_move
        opp = WHITE if me == BLACK else BLACK

        wins = immediate_wins(board, me)
        if wins:
            return wins[0]
        threats = immediate_wins(board, opp)
        if len(threats) == 1:
            return threats[0]
        if len(threats) > 1:
            # Lost against perfect play; block the most valuable one anyway.
            scores = move_scores(board, me)
            return max(threats, key=lambda m: scores[m])

        best = int(np.argmax(move_scores(board, me)))
        self.nodes = 0
        for depth in range(1, self.max_depth + 1):
            try:
                value, move = self._root(board, depth, deadline)
            except TimeoutError:
                break
            if move is not None:
                best = move
                self.last_depth = depth
            if value is not None and abs(value) >= 0.999:
                break  # proven win/loss, deeper search cannot improve on it
            if time.time() >= deadline:
                break
        return best

    # -------------------------------------------------------------- search
    def _root(self, board: HexBoard, depth: int, deadline: float):
        alpha, beta = -2.0, 2.0
        best_move = None
        best_value = -2.0
        for move in self._candidates(board, depth):
            child = board.copy()
            child.play(move)
            value = -self._negamax(child, depth - 1, -beta, -alpha, deadline)
            if value > best_value:
                best_value, best_move = value, move
            alpha = max(alpha, value)
        return best_value, best_move

    def _negamax(self, board: HexBoard, depth: int, alpha: float, beta: float, deadline: float) -> float:
        self.nodes += 1
        if self.nodes % 128 == 0 and time.time() >= deadline:
            raise TimeoutError
        if board.winner:
            return -1.0  # the side to move has already lost
        if depth <= 0:
            return evaluate(board, board.to_move)

        key = (bytes(board.board), board.to_move)
        hit = self.tt.get(key)
        tt_move = None
        if hit is not None:
            h_depth, h_flag, h_value, h_move = hit
            tt_move = h_move
            if h_depth >= depth:
                if h_flag == EXACT:
                    return h_value
                if h_flag == LOWER and h_value > alpha:
                    alpha = h_value
                elif h_flag == UPPER and h_value < beta:
                    beta = h_value
                if alpha >= beta:
                    return h_value

        me = board.to_move
        opp = WHITE if me == BLACK else BLACK
        wins = immediate_wins(board, me)
        if wins:
            return 1.0
        threats = immediate_wins(board, opp)
        if len(threats) == 1:
            moves = threats  # forced reply, search it at full depth
        elif len(threats) > 1:
            return -1.0
        else:
            moves = self._candidates(board, depth, tt_move)

        original_alpha = alpha
        best_value = -2.0
        best_move = None
        for move in moves:
            child = board.copy()
            child.play(move)
            value = -self._negamax(child, depth - 1, -beta, -alpha, deadline)
            if value > best_value:
                best_value, best_move = value, move
            alpha = max(alpha, value)
            if alpha >= beta:
                break  # beta cutoff

        flag = EXACT
        if best_value <= original_alpha:
            flag = UPPER
        elif best_value >= beta:
            flag = LOWER
        if len(self.tt) < 400_000:
            self.tt[key] = (depth, flag, best_value, best_move)
        return best_value

    def _candidates(self, board: HexBoard, depth: int, tt_move: int | None = None) -> list[int]:
        scores = move_scores(board, board.to_move)
        # Widen near the root, narrow deep down -- the cheap way to spend the
        # budget where it changes the answer.
        width = self.beam if depth >= 3 else max(4, self.beam // 2)
        order = np.argsort(-scores)[:width]
        moves = [int(m) for m in order if np.isfinite(scores[m])]
        if tt_move is not None and tt_move in moves:
            moves.remove(tt_move)
            moves.insert(0, tt_move)
        elif tt_move is not None and board.board[tt_move] == 0:
            moves.insert(0, tt_move)
        return moves
