"""Ultimate XX knowledge, shared by the classical agents.

What a player of this game has to be able to see, and none of it is what an
Ultimate Tic-Tac-Toe player looks for:

* **The move that loses now.**  Stepping on a cell that closes a line inside a
  small board takes that board -- and if it is the third of a line of boards you
  already hold, you have just lost.  Every other rule here is downstream of not
  doing that.
* **The move that wins now.**  Send the opponent to a board where *every*
  remaining cell closes a line, and where taking that board completes their own
  line of three, and they are finished: they have to move, and every move loses.
  This is the whole shape of a won position.
* **Pressure.**  How close each side is to being forced into a third board.  Two
  boards of a line with the third still open is a live threat *against yourself*
  -- the mirror image of a threat in an ordinary game -- so the sign of the
  usual line count is flipped, and a line the opponent has blocked is a line you
  are safe on.
* **Room to stand.**  A player with plenty of squares that claim nothing is in
  no danger; a player whose every legal square takes a board is already losing.
  Misere games are decided by running out of harmless moves, so the count of
  them is a real term and not a tie-breaker.
* **The cost of a free choice.**  Sending the opponent to a board that is
  already claimed lets them play anywhere, which throws away the only lever the
  game gives you -- exactly as in Ultimate Tic-Tac-Toe, and for the same reason.

Everything works on the engine's 9-bit masks, so the whole file is integer ops
and table lookups on the 512 possible masks.
"""

from __future__ import annotations

import numpy as np

from .uxx_game import (BOARD_OF, LINE_MASKS, NCELLS, POPCOUNT, SLOT_OF, WON,
                       UltimateXXBoard, other)

# How dangerous a meta line is by how many of its boards I already hold, with
# none of them the opponent's.  Superlinear on purpose: two-of-three means the
# third board is a bomb I have to keep away from for the rest of the game.
LINE_VALUE = (0.0, 1.0, 6.0, 40.0)

# The meta board is the only level that decides anything, so its lines are what
# the score is mostly made of.  Everything else is a correction to it.
META_SCALE = 25.0

# Having harmless squares to play is worth this much at full mobility, and the
# same again in the other direction when every legal square claims a board.
# Sizeable, because "no safe move" *is* the losing condition, arrived at.
MOBILITY = 14.0

# What it costs to hand the opponent a free choice of board.  Large, because
# the constraint is the whole game: a player who is never confined can always
# walk away from the board you were steering them into.
FREE_CHOICE = 12.0

# A live board that would finish my line of three: somewhere I must never be
# sent.  Full weight when it is poisoned (any visit takes it), less when there
# are still harmless squares in it.
MINEFIELD = 9.0
MINEFIELD_SOFT = 0.45

# Which small board a mark is in, and which slot it sits in, both matter, and
# for the same reason as in Ultimate Tic-Tac-Toe with the sign reversed: the
# centre takes part in four lines of three, a corner three, an edge two -- so
# the centre is where a board goes hot soonest, and it is the *quiet* squares
# that are the natural place to start.  The same nine weights serve both levels.
CELL_WEIGHT = (1.4, 1.0, 1.4,
               1.0, 1.75, 1.0,
               1.4, 1.0, 1.4)


def pressure(mine: int, theirs: int) -> float:
    """How close ``mine`` is to completing a line of small boards, i.e. losing.

    Lines the opponent has a board on are dead and score nothing: they can never
    be completed, which here is a good thing rather than a bad one.  There is no
    third case -- a small board cannot end drawn in this game -- so unlike the
    Ultimate Tic-Tac-Toe version of this function nothing else blocks a line.
    """
    total = 0.0
    for mask in LINE_MASKS:
        if theirs & mask:
            continue
        total += LINE_VALUE[POPCOUNT[mine & mask]]
    return total


def losing_moves(board: UltimateXXBoard, player: int) -> list[int]:
    """Legal moves that hand ``player`` the third board of a line, and the game."""
    return [int(m) for m in board.legal_moves() if board.would_lose(player, int(m))]


def safe_moves(board: UltimateXXBoard, player: int) -> list[int]:
    """Legal moves that do not lose on the spot.

    Returns every legal move when they *all* lose: a lost position still has to
    play something, and refusing to choose is worse than choosing badly.
    """
    legal = [int(m) for m in board.legal_moves()]
    safe = [m for m in legal if not board.would_lose(player, m)]
    return safe or legal


def claiming_moves(board: UltimateXXBoard) -> list[int]:
    """Legal moves that close a line and take a small board, for whoever plays.

    No player argument, and that is the point: the mark is the same either way,
    so taking a board is a property of the square and of nothing else.
    """
    return [int(m) for m in board.legal_moves() if board.claims_small(int(m))]


def forced_wins(board: UltimateXXBoard, player: int) -> list[int]:
    """Legal moves after which every reply loses for the opponent.

    One ply of lookahead, and the only kind of "winning move" the game has: you
    never win by doing something, only by leaving the other player nowhere to
    put a mark.
    """
    out = []
    for move in board.legal_moves():
        move = int(move)
        if board.would_lose(player, move):
            continue
        probe = board.copy()
        probe.play(move)
        if probe.is_terminal():
            continue  # nine boards claimed with no line: a draw, not a win
        if not probe.safe_moves():
            out.append(move)
    return out


def evaluate(board: UltimateXXBoard, player: int,
             defence_weight: float = 1.0) -> float:
    """Static score of a *non-terminal* position, from ``player``'s side.

    Positive is good for ``player``.  ``defence_weight`` scales how much the
    opponent's half counts, which is what makes a cautious variant of the agent.
    """
    return (_side(board, player)
            - defence_weight * _side(board, other(player)))


def _side(board: UltimateXXBoard, player: int) -> float:
    """How comfortable the position is for one player, ignoring the other.

    Three terms: how near ``player`` is to completing a line of small boards
    (which is what losing is), what it is like to be them on turn -- how many of
    their squares are harmless, and whether they are confined -- and how many
    live boards there are that they must never be sent into.
    """
    opponent = other(player)
    mine, theirs = board.claims(player), board.claims(opponent)
    value = -META_SCALE * pressure(mine, theirs)

    if board.to_move == player and not board.is_terminal():
        legal = board.legal_moves()
        if len(legal):
            safe = sum(1 for m in legal if not board.would_lose(player, int(m)))
            # +MOBILITY when nothing on offer claims a board, -MOBILITY when
            # everything does, which is the position a move before losing.
            value += MOBILITY * (2.0 * safe / len(legal) - 1.0)
        if board.active is None:
            value += FREE_CHOICE

    for b in board.open_boards():
        if not board.hot(b):
            continue  # nothing in it closes a line yet, so it is nobody's trap
        if WON[mine | (1 << b)]:
            value -= MINEFIELD * (1.0 if board.is_poisoned(b) else MINEFIELD_SOFT)
    return value


def move_scores(board: UltimateXXBoard, player: int,
                defence_weight: float = 1.0) -> np.ndarray:
    """Per-cell score for every legal move; ``-inf`` for illegal ones.

    Play the move, then read the position with :func:`evaluate` -- so a move is
    judged by what it leaves behind rather than by what it looks like -- with
    the two results the game can reach in one ply short-circuited: a move that
    completes our own line of boards loses, and a move after which the opponent
    has nothing safe to play wins.
    """
    scores = np.full(NCELLS, -np.inf, dtype=np.float64)
    for move in board.legal_moves():
        move = int(move)
        probe = board.copy()
        probe.play(move)
        if probe.loser == player:
            scores[move] = -1e9  # never volunteer the game
            continue
        if probe.is_terminal():
            scores[move] = 0.0  # every board claimed, no line: exactly level
            continue
        if not probe.safe_moves():
            scores[move] = 1e9  # they have to move, and every move loses
            continue
        scores[move] = evaluate(probe, player, defence_weight)
    return scores


def static_order() -> list[int]:
    """Cells ranked by how little they commit, before anything is on the board.

    The product of the two weights, *ascending*: the edge of an edge board
    first, the centre of the centre board last.  Ultimate Tic-Tac-Toe wants the
    busy squares because they build the most lines; here building a line is how
    you lose, so the same ranking runs the other way.  Used to order an
    alpha-beta search, where a good first guess is most of what makes the
    cutoffs happen, and to spread a match's openings out.
    """
    return sorted(range(NCELLS),
                  key=lambda c: (CELL_WEIGHT[BOARD_OF[c]] * CELL_WEIGHT[SLOT_OF[c]], c))


# Ranked once: the ordering is a property of the geometry, not of a position.
STATIC_RANK = {cell: i for i, cell in enumerate(static_order())}
