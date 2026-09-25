"""Ultimate Tic-Tac-Toe knowledge, shared by the classical agents.

Four things every non-search player here needs, and one an alpha-beta search
needs at its horizon:

* **Immediate wins** -- the moves that take the third small board of a line and
  finish the game now.
* **Immediate losses** -- the moves after which the opponent can finish the game
  on their reply.  In this game that is nearly always about *where the move
  sends them* rather than about what it takes, which is what makes it the
  characteristic beginner's mistake.
* **A static evaluation** on two levels: every line of three inside every small
  board, and every line of three across the grid of small boards, scored by how
  close it is to completion and by whether the opponent is contesting it.
* **The cost of a free choice.**  Sending the opponent to a board that is
  already decided lets them play anywhere, which throws away the only lever
  this game gives you.  It is a real, sizeable term in the evaluation and not a
  tie-breaker.

Everything works on the engine's 9-bit small-board masks, so the whole file is
integer ops and table lookups on the 512 possible masks.
"""

from __future__ import annotations

import numpy as np

from .uttt_game import (BOARD_OF, FULL, LINE_MASKS, NBOARDS, NCELLS, POPCOUNT,
                        SLOT_OF, WON, UltimateBoard, other)

# What a line of three is worth by how many of its cells are already mine, with
# none of them the opponent's.  Superlinear on purpose: two-of-three is a live
# threat, one is barely a hint.
LINE_VALUE = (0.0, 1.0, 6.0, 40.0)

# A small board won is worth far more than a mark placed, so the meta board's
# line score is scaled up before it joins the sum.  Calibrated so that one won
# board (25) outranks any single small-board threat (~10) and two won boards in
# a line (150) outrank almost anything short of a third.
META_SCALE = 25.0

# Which small board a mark is in, and which slot it sits in, both matter, and
# for the same reason: the centre takes part in four lines of three, a corner
# in three, an edge in two.  The same nine weights therefore serve both levels.
CELL_WEIGHT = (1.4, 1.0, 1.4,
               1.0, 1.75, 1.0,
               1.4, 1.0, 1.4)
POSITION_VALUE = 0.7   # per mark, multiplied by both weights

# What it costs to hand the opponent a free choice of board.  Large, because
# the constraint is the whole game: a player who is never confined can pick the
# board they are winning every single turn.
FREE_CHOICE_PENALTY = 12.0


def line_score(mine: int, theirs: int) -> float:
    """Score one 3x3 layer for the owner of ``mine``, uncontested lines only.

    A line holding marks of both sides is dead and scores nothing for either,
    which is most of a finished small board and is what keeps the count
    meaningful.  ``theirs`` is also where drawn small boards go when this is
    called on the *meta* board: a line through a drawn board can never be
    completed by anybody, so it is contested in exactly the sense that matters.
    """
    total = 0.0
    for mask in LINE_MASKS:
        if theirs & mask:
            continue
        total += LINE_VALUE[POPCOUNT[mine & mask]]
    return total


def immediate_wins(board: UltimateBoard, player: int) -> list[int]:
    """Legal moves that win the whole game for ``player`` right now."""
    return [int(m) for m in board.legal_moves() if board.would_win(player, int(m))]


def small_wins(board: UltimateBoard, player: int) -> list[int]:
    """Legal moves that take a small board (whether or not they end the game)."""
    return [int(m) for m in board.legal_moves() if board.wins_small(player, int(m))]


def sends_free(board: UltimateBoard, move: int) -> bool:
    """Would this move leave the opponent free to play anywhere?

    True when the slot played points at a board that is already decided -- or
    at the board the move itself is about to close, which is the case that
    catches players out: winning a small board with its own centre sends the
    opponent to the centre board, and if that is the board just won, they are
    free.
    """
    b, k = BOARD_OF[move], SLOT_OF[move]
    if (board.closed >> k) & 1:
        return True
    if b != k:
        return False
    # The target is the board being played in; it is free only if this move
    # closes it.
    player = board.to_move
    mine = board.small(player, b) | (1 << k)
    return WON[mine] or (mine | board.small(other(player), b)) == FULL


def gives_opponent_win(board: UltimateBoard, move: int, player: int) -> bool:
    """Does playing ``move`` let the opponent finish the game on their reply?"""
    probe = board.copy()
    probe.play(move)
    if probe.is_terminal():
        return False  # the move ended the game; there is no reply
    return bool(immediate_wins(probe, other(player)))


def safe_moves(board: UltimateBoard, player: int) -> list[int]:
    """Legal moves that do not hand the opponent an immediate win.

    Returns every legal move when they are *all* losing: a lost position still
    has to play something, and refusing to choose is worse than choosing badly.
    """
    legal = [int(m) for m in board.legal_moves()]
    safe = [m for m in legal if not gives_opponent_win(board, m, player)]
    return safe or legal


def evaluate(board: UltimateBoard, player: int) -> float:
    """Static score of a *non-terminal* position, from ``player``'s side.

    Positive is good for ``player``.  Three terms, in descending importance:
    the meta board's lines of won boards, each open small board's own lines
    weighted by how much that board is worth, and a small positional bonus for
    marks on squares that take part in more lines.
    """
    opponent = other(player)
    mine_meta, their_meta = board.meta(player), board.meta(opponent)
    # A drawn board blocks a meta line for both sides, so it counts as the
    # other side's mark when scoring either side's lines.
    dead = board.closed & ~mine_meta & ~their_meta
    score = META_SCALE * (line_score(mine_meta, their_meta | dead)
                          - line_score(their_meta, mine_meta | dead))

    for b in range(NBOARDS):
        if (board.closed >> b) & 1:
            continue  # a decided board's own lines no longer matter
        mine, theirs = board.small(player, b), board.small(opponent, b)
        weight = CELL_WEIGHT[b]
        score += weight * (line_score(mine, theirs) - line_score(theirs, mine))
        score += POSITION_VALUE * weight * _position(mine, theirs)
    return score


def _position(mine: int, theirs: int) -> float:
    """Weighted mark count inside one small board, mine minus theirs."""
    total = 0.0
    for k in range(9):
        bit = 1 << k
        if mine & bit:
            total += CELL_WEIGHT[k]
        elif theirs & bit:
            total -= CELL_WEIGHT[k]
    return total


def move_scores(board: UltimateBoard, player: int, defence_weight: float = 1.0
                ) -> np.ndarray:
    """Per-cell score for every legal move; ``-inf`` for illegal ones.

    Play the move, then read the position with :func:`evaluate` -- so a move is
    judged by what it leaves behind rather than by what it looks like -- and
    subtract what it gives away: a free choice of board, or an outright win on
    the reply.  ``defence_weight`` scales how much the opponent's half of the
    static score counts, which is what makes a cautious variant of the agent.
    """
    scores = np.full(NCELLS, -np.inf, dtype=np.float64)
    opponent = other(player)
    for move in board.legal_moves():
        move = int(move)
        probe = board.copy()
        probe.play(move)
        if probe.winner == player:
            scores[move] = 1e9
            continue
        attack = _one_sided(probe, player)
        defence = _one_sided(probe, opponent)
        value = attack - defence_weight * defence
        if probe.active is None and not probe.is_terminal():
            value -= FREE_CHOICE_PENALTY
        if gives_opponent_win(board, move, player):
            value -= 1e6  # never volunteer the game
        scores[move] = value
    return scores


def _one_sided(board: UltimateBoard, player: int) -> float:
    """The half of :func:`evaluate` that belongs to one player."""
    opponent = other(player)
    mine_meta, their_meta = board.meta(player), board.meta(opponent)
    dead = board.closed & ~mine_meta & ~their_meta
    total = META_SCALE * line_score(mine_meta, their_meta | dead)
    for b in range(NBOARDS):
        if (board.closed >> b) & 1:
            continue
        mine, theirs = board.small(player, b), board.small(opponent, b)
        weight = CELL_WEIGHT[b]
        total += weight * line_score(mine, theirs)
        total += POSITION_VALUE * weight * _position(mine, 0)
    return total


def static_order() -> list[int]:
    """Cells ranked by how much they are worth before anything is on the board.

    The product of the two weights: the centre of the centre board first, the
    edge of an edge board last.  Used to order an alpha-beta search, where a
    good first guess is most of what makes the cutoffs happen.
    """
    return sorted(range(NCELLS),
                  key=lambda c: -CELL_WEIGHT[BOARD_OF[c]] * CELL_WEIGHT[SLOT_OF[c]])


# Ranked once: the ordering is a property of the geometry, not of a position.
STATIC_RANK = {cell: i for i, cell in enumerate(static_order())}
