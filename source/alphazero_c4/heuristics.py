"""Connect Four knowledge, shared by the classical agents.

Three things every non-search Connect Four player needs, and one an alpha-beta
search needs at its horizon:

* **Immediate wins** -- the moves that finish the game now.
* **Immediate losses** -- the moves that hand the opponent a win *on top of*
  the disc just played.  This is the mistake that separates a beginner from a
  merely weak player, and it costs nothing to avoid.
* **Threats** -- empty cells that would complete a four.  A threat the opponent
  cannot cover is what wins games; two of them side by side, or stacked, are
  unanswerable.
* **A static evaluation** -- every four-in-a-row window on the board, scored by
  how close it is to being completed and by who is contesting it.

Everything works on the board's bitboards, so the whole file is integer ops and
``int.bit_count``; the windows are precomputed once per geometry.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .c4_game import Connect4Board, other

# What a window is worth by how many of the four cells are already mine, with
# none of them the opponent's.  Superlinear on purpose: three-of-four is a live
# threat, two is a hint, one is almost nothing.
WINDOW_VALUE = (0.0, 1.0, 6.0, 40.0)
CENTRE_VALUE = 3.0  # per disc, scaled by how central the column is


@lru_cache(maxsize=8)
def window_masks(rows: int, cols: int, h1: int) -> tuple[int, ...]:
    """Bitmasks of every four-in-a-row window on the board.

    69 of them on a standard 6x7.  Built from ``(row, col)`` coordinates rather
    than by shifting, so there is no chance of a mask wrapping across a column
    boundary the way a raw shift can.
    """
    masks = []
    for r in range(rows):
        for c in range(cols):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                cells = [(r + dr * k, c + dc * k) for k in range(4)]
                if any(not (0 <= rr < rows and 0 <= cc < cols) for rr, cc in cells):
                    continue
                mask = 0
                for rr, cc in cells:
                    mask |= 1 << (cc * h1 + (rows - 1 - rr))
                masks.append(mask)
    return tuple(masks)


@lru_cache(maxsize=8)
def centre_weights(cols: int) -> tuple[float, ...]:
    """How much a disc in each column is worth, purely for being central.

    Central discs take part in more windows -- the middle column of a 6x7 board
    is in 13 of them, the edge column in 3 -- so this is a cheap stand-in for
    "counts towards more things".
    """
    mid = (cols - 1) / 2
    return tuple(1.0 - abs(c - mid) / (mid + 1.0) for c in range(cols))


def immediate_wins(board: Connect4Board, player: int) -> list[int]:
    """Legal moves that complete a four for ``player`` right now."""
    return [int(m) for m in board.legal_moves() if board.would_win(player, int(m))]


def gives_opponent_win(board: Connect4Board, move: int, player: int) -> bool:
    """Does playing ``move`` let the opponent win on the square directly above?"""
    row, col = divmod(move, board.cols)
    if row == 0:
        return False  # nothing can be stacked on the top row
    above = move - board.cols
    probe = board.copy()
    probe.play(move)
    if probe.is_terminal():
        return False  # the move ended the game; nothing goes on top of it
    return probe.would_win(other(player), above)


def safe_moves(board: Connect4Board, player: int) -> list[int]:
    """Legal moves that do not hand the opponent an immediate win.

    Returns every legal move when they are *all* losing: a lost position still
    has to play something, and refusing to choose is worse than choosing badly.
    """
    legal = [int(m) for m in board.legal_moves()]
    safe = [m for m in legal if not gives_opponent_win(board, m, player)]
    return safe or legal


def threat_cells(board: Connect4Board, player: int) -> list[int]:
    """Every empty cell -- reachable or not -- that would complete a four.

    Includes cells buried high up a column, which is the point: an unreachable
    threat still decides the endgame, because whoever is forced to play beneath
    it loses.
    """
    mine = board.bitboard(player)
    occupied = mine | board.bitboard(other(player))
    out = []
    for r in range(board.rows):
        for c in range(board.cols):
            bit = 1 << board.bit_of(r, c)
            if occupied & bit:
                continue
            if _connects(mine | bit, board.bits_per_column):
                out.append(r * board.cols + c)
    return out


def _connects(bb: int, h1: int) -> bool:
    for shift in (1, h1, h1 - 1, h1 + 1):
        m = bb & (bb >> shift)
        if m & (m >> (2 * shift)):
            return True
    return False


def evaluate(board: Connect4Board, player: int) -> float:
    """Static score of a *non-terminal* position, from ``player``'s side.

    Positive is good for ``player``.  Uncontested windows only: a window that
    holds discs of both colours is dead and scores nothing for either side,
    which is most of the board in a real game and is what keeps the count
    meaningful.
    """
    mine = board.bitboard(player)
    theirs = board.bitboard(other(player))
    score = 0.0
    for mask in window_masks(board.rows, board.cols, board.bits_per_column):
        m = (mine & mask).bit_count()
        t = (theirs & mask).bit_count()
        if t == 0:
            score += WINDOW_VALUE[m]
        elif m == 0:
            score -= WINDOW_VALUE[t]
    weights = centre_weights(board.cols)
    h1 = board.bits_per_column
    for c in range(board.cols):
        col_mask = ((1 << board.rows) - 1) << (c * h1)
        score += CENTRE_VALUE * weights[c] * (
            (mine & col_mask).bit_count() - (theirs & col_mask).bit_count()
        )
    return score


def move_scores(board: Connect4Board, player: int, defence_weight: float = 1.0
                ) -> np.ndarray:
    """Per-cell score for every legal move; ``-inf`` for illegal ones.

    Play the move, then read the position with :func:`evaluate` -- the
    opponent's threats included, weighted by ``defence_weight`` -- so a move is
    judged by what it leaves behind rather than by what it looks like.
    """
    scores = np.full(board.ncells, -np.inf, dtype=np.float64)
    opponent = other(player)
    for move in board.legal_moves():
        move = int(move)
        probe = board.copy()
        probe.play(move)
        if probe.winner == player:
            scores[move] = 1e9
            continue
        mine = probe.bitboard(player)
        theirs = probe.bitboard(opponent)
        attack = _window_score(probe, mine, theirs)
        defence = _window_score(probe, theirs, mine)
        scores[move] = attack - defence_weight * defence
        if gives_opponent_win(board, move, player):
            scores[move] -= 1e6  # never volunteer the square above
    return scores


def _window_score(board: Connect4Board, mine: int, theirs: int) -> float:
    total = 0.0
    for mask in window_masks(board.rows, board.cols, board.bits_per_column):
        if theirs & mask:
            continue
        total += WINDOW_VALUE[(mine & mask).bit_count()]
    weights = centre_weights(board.cols)
    h1 = board.bits_per_column
    for c in range(board.cols):
        col_mask = ((1 << board.rows) - 1) << (c * h1)
        total += CENTRE_VALUE * weights[c] * (mine & col_mask).bit_count()
    return total


def centre_order(cols: int) -> list[int]:
    """Columns from the middle outwards -- the standard move ordering."""
    mid = (cols - 1) / 2
    return sorted(range(cols), key=lambda c: abs(c - mid))
