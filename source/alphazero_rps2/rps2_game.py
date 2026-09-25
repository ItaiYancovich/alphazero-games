"""Intransitive (RPS2) game engine.

Chess-like movement on a 9x9 board with capture decided by rock-paper-scissors
instead of by piece value.  Every piece moves one square in any of the eight
directions, like a chess king; a piece may enter a square holding an enemy only
if its type *beats* that enemy's, and the loser is removed.  Reach the enemy's
base corner and the game is over.

Board geometry
--------------
One ``9 x 9`` array indexed by ``(row, col)`` with **row 0 at the top**, so
``square = row * 9 + col`` -- the convention Hex, Connect Four and Ultimate
Tic-Tac-Toe already use.  The game's own notation is chess-like and counts
ranks from the *bottom*, so rank ``k`` is row ``9 - k``:

* file ``a``..``i`` is column ``0``..``8``
* rank ``9``..``1`` is row ``0``..``8``
* **blue base ``a1``** is ``(8, 0)`` = square 72
* **red base ``i9``** is ``(0, 8)`` = square 8

Blue moves first and is the engine's ``BLACK``; the shared arena, ratings and
GUI call the two sides by those names and never look at the colour words.

The move space, and why it is not the board
-------------------------------------------
Every other cell game here answers "where does the next mark go", so a policy
vector with one entry per cell is the whole action space and the shared network
needs no head of its own.  Here a move is a *pair* -- which piece, and which
way -- so the action space is ``81 squares x 8 directions = 648``.

The encoding is ``action = direction * 81 + from``, which is deliberately
plane-major: the network's policy head emits ``8`` channels over the 9x9 board
and flattens them, so channel ``d`` of the head *is* "move the piece on this
square in direction ``d``", and one 3x3 convolution sees a square and every
neighbour it could move to.  Nothing about the trunk changes.

The intransitive cycle
----------------------
Types are numbered ``ROCK=0``, ``PAPER=1``, ``SCISSORS=2`` so that "``a`` beats
``b``" is ``(a - b) % 3 == 1``: rock (0) beats scissors (2), scissors (2) beats
paper (1), paper (1) beats rock (0).  Equal types cannot capture each other and
are mutually immovable walls, which is what makes blocking a real resource --
in a game where every piece moves like a king, the only way to hold a square is
to stand on it with something the opponent cannot take.

How a game ends
---------------
* **base** -- a piece reaches the enemy base square.  Immediate win.
* **no legal moves** -- the side to move cannot move (including having no
  pieces left) and loses.  Unlike chess there is no stalemate-is-a-draw.
* **repetition** -- the same position three times is a draw.  See
  ``REPETITION_LIMIT``: this is the one rule here that the published game does
  not have, and it is the one that stops a shuffling game costing 47% of the
  self-play budget to confirm what it already is.
* **stagnation** -- 200 half-moves with no capture is a draw.  The counter
  resets on every capture.

The last two are what make draws real, so the search's proven-draw state, the
half point in the arena and the ``0`` in the value target all carry over from
Connect Four unchanged.
"""

from __future__ import annotations

import numpy as np

N = 9
NCELLS = N * N            # squares
NDIRS = 8
NACTIONS = NCELLS * NDIRS  # 648: the size of the canonical policy vector

EMPTY = 0
BLUE = 1        # moves first
RED = 2

# The names the shared code (arena, ratings, review, GUI) uses for the two
# sides.  Same integers, so a position is interchangeable with the other games'.
BLACK = BLUE
WHITE = RED

ROCK, PAPER, SCISSORS = 0, 1, 2
TYPE_NAMES = ("rock", "paper", "scissors")
TYPE_LETTERS = "RPS"

# Cell encoding: 0 empty, 1-3 blue R/P/S, 4-6 red R/P/S.  One byte per square,
# which is what lets a position be an ordinary ``uint8`` grid like every other
# game here.
def code(player: int, kind: int) -> int:
    return (player - 1) * 3 + kind + 1


def owner_of(v: int) -> int:
    return 0 if v == 0 else (BLUE if v <= 3 else RED)


def kind_of(v: int) -> int:
    return (v - 1) % 3


def beats(a: int, b: int) -> bool:
    """Does type ``a`` capture type ``b``?  Rock > scissors > paper > rock."""
    return (a - b) % 3 == 1


BLUE_BASE = 8 * N + 0     # a1
RED_BASE = 0 * N + 8      # i9
BASE_OF = {BLUE: BLUE_BASE, RED: RED_BASE}
TARGET_BASE = {BLUE: RED_BASE, RED: BLUE_BASE}

STAGNATION_LIMIT = 200    # half-moves with no capture -> draw
# ...and a draw the published rules do not have, added deliberately.
#
# The same position -- the same pieces on the same squares with the same side
# to move -- occurring three times is a draw.  This is the one place this engine
# departs from the rules at meaf.us/rps2, which stop only at the 200-half-move
# counter, and it is worth being explicit about why.
#
# A repeated position is *zero progress by definition*: nothing on the board has
# changed and it is the same player's turn, so the side to move has now had
# three chances from an identical state and chosen to come back to it.  Under
# the players' own behaviour that is a draw, and playing it out only discovers
# how long the counter takes to expire.  Measured on this run, 42% of self-play
# games reach a third repetition and the plies played after it are **47% of all
# plies generated** -- so roughly half the self-play budget was being spent
# after the game had, in this sense, already ended.
#
# What the shorter rule gives up: with exploration noise in self-play, a game
# can be jostled out of a loop and go on to be decided, and those results are
# now recorded as draws instead.  That is a real cost and it is accepted
# knowingly -- a position the search returns to three times is not one it has
# found a win in; the noise found the exit, not the player.
REPETITION_LIMIT = 3
# A hard ceiling on any single game, well above the stagnation draw, so that a
# rollout or a review can never run forever if the counter is switched off.
MAX_PLIES = 400

# N, NE, E, SE, S, SW, W, NW -- clockwise from straight up the screen, which is
# towards rank 9 and therefore towards red's base.
DIRS = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))
OPPOSITE_DIR = tuple((d + 4) % NDIRS for d in range(NDIRS))
# The two symmetries the *board* has, as maps on directions.
#   180 degree rotation, which is also what canonicalising red into blue does
ROT180_DIR = tuple((d + 4) % NDIRS for d in range(NDIRS))
#   reflection in the a1-i9 diagonal: (dr, dc) -> (-dc, -dr)
FLIP_DIR = tuple(DIRS.index((-dc, -dr)) for dr, dc in DIRS)

# square -> the square one step in each direction, or -1 off the board.
STEP = np.full((NCELLS, NDIRS), -1, dtype=np.int32)
for _s in range(NCELLS):
    _r, _c = divmod(_s, N)
    for _d, (_dr, _dc) in enumerate(DIRS):
        _nr, _nc = _r + _dr, _c + _dc
        if 0 <= _nr < N and 0 <= _nc < N:
            STEP[_s, _d] = _nr * N + _nc
STEP_LIST = [tuple(int(x) for x in STEP[s]) for s in range(NCELLS)]
NEIGHBOURS = [tuple(int(x) for x in STEP[s] if x >= 0) for s in range(NCELLS)]

# 180 degree rotation of a square, which is the geometric half of the swap
# between the two players' points of view.
ROT180 = tuple(NCELLS - 1 - s for s in range(NCELLS))
# The same two maps as arrays, for the one place that applies them in bulk.
_ROT180_ARR = np.array(ROT180, dtype=np.int64)
_ROT180_DIR_ARR = np.array(ROT180_DIR, dtype=np.int64)
# Reflection in the a1-i9 diagonal: (r, c) -> (8 - c, 8 - r).  It fixes both
# bases, so it is a genuine symmetry of the game and not merely of the grid.
FLIP = tuple(((N - 1 - (s % N)) * N + (N - 1 - s // N)) for s in range(NCELLS))

# Chebyshev distance from every square to each base: how many king moves a
# piece standing there still needs.  Used by the heuristics and by a feature
# plane, and cheap enough to precompute once.
def _cheb(a: int, b: int) -> int:
    ar, ac = divmod(a, N)
    br, bc = divmod(b, N)
    return max(abs(ar - br), abs(ac - bc))


DIST_TO = {p: tuple(_cheb(s, BASE_OF[p]) for s in range(NCELLS))
           for p in (BLUE, RED)}

# The opening army, as (square, type) pairs for blue.  Red is the same set
# rotated by 180 degrees, which is also its reflection in the a9-i1 diagonal --
# the starting band is symmetric, so the two descriptions agree.
BLUE_SETUP = (
    (46, ROCK), (56, ROCK), (66, ROCK),                 # b4 c3 d2
    (37, PAPER), (47, PAPER), (57, PAPER), (67, PAPER),  # b5 c4 d3 e2
    (38, SCISSORS), (48, SCISSORS), (58, SCISSORS),      # c5 d4 e3
)


def other(player: int) -> int:
    return RED if player == BLUE else BLUE


def action_of(square: int, direction: int) -> int:
    return direction * NCELLS + square


def split_action(action: int) -> tuple[int, int]:
    """``action`` -> ``(from square, direction)``."""
    return action % NCELLS, action // NCELLS


def action_squares(action: int) -> tuple[int, int]:
    """``action`` -> ``(from square, to square)``; ``to`` is -1 off the board."""
    src, d = split_action(action)
    return src, int(STEP[src, d])


class RPS2Board:
    """Mutable Intransitive position.

    The state is one ``bytearray`` of 81 squares plus a set of occupied squares
    per player, which is what makes move generation proportional to the ten
    pieces a side has rather than to the board.
    """

    __slots__ = ("rows", "cols", "ncells", "cells", "pieces", "to_move",
                 "winner", "drawn", "move_count", "since_capture",
                 "last_move", "end_reason", "_legal", "_history")

    def __init__(self, rows: int = N, cols: int = N):
        # ``rows``/``cols`` exist so the shared trainer, evaluator and GUI can
        # describe this board the way they describe the others.  The game has
        # exactly one size; anything else is a mistake worth catching here.
        if (rows, cols) != (N, N):
            raise ValueError(f"Intransitive is {N}x{N}, not {rows}x{cols}")
        self.rows = N
        self.cols = N
        # The size of the canonical policy vector, which is what the shared
        # search means by ``ncells`` -- here that is moves, not squares.
        self.ncells = NACTIONS
        self.cells = bytearray(NCELLS)
        self.pieces = {BLUE: set(), RED: set()}
        for square, kind in BLUE_SETUP:
            self.cells[square] = code(BLUE, kind)
            self.pieces[BLUE].add(square)
            mirrored = ROT180[square]
            self.cells[mirrored] = code(RED, kind)
            self.pieces[RED].add(mirrored)
        self.to_move = BLUE
        self.winner = 0
        self.drawn = False
        self.move_count = 0
        self.since_capture = 0
        self.last_move: int | None = None
        self.end_reason: str | None = None
        self._legal: np.ndarray | None = None
        # Positions seen since the last capture, for the repetition draw.  A
        # *tuple* rather than a dict or a Counter for one reason: ``copy`` runs
        # once per simulation and a tuple is shared by reference, so copying a
        # position stays the price of a bytearray and two sets.  It is reset on
        # every capture, which is what keeps the scan short -- a capture is
        # irreversible, so nothing before it can ever recur.
        self._history: tuple[bytes, ...] = (self._rep_key(),)

    # ---------------------------------------------------------------- basics
    def copy(self) -> "RPS2Board":
        new = RPS2Board.__new__(RPS2Board)
        new.rows = N
        new.cols = N
        new.ncells = NACTIONS
        new.cells = bytearray(self.cells)
        new.pieces = {BLUE: set(self.pieces[BLUE]), RED: set(self.pieces[RED])}
        new.to_move = self.to_move
        new.winner = self.winner
        new.drawn = self.drawn
        new.move_count = self.move_count
        new.since_capture = self.since_capture
        new.last_move = self.last_move
        new.end_reason = self.end_reason
        new._legal = self._legal
        new._history = self._history     # immutable; shared, never mutated
        return new

    def _rep_key(self) -> bytes:
        """What has to match for two positions to be *the same* position.

        The squares and the side to move, and nothing else.  Deliberately not
        :meth:`key`, which folds in the stagnation counter -- that advances
        every ply, so two identical arrangements would never compare equal and
        the repetition rule would never fire.
        """
        return bytes(self.cells) + bytes((self.to_move,))

    def repetitions(self) -> int:
        """How many times the position on the board has occurred."""
        return self._history.count(self._history[-1]) if self._history else 0

    def piece_at(self, square: int) -> tuple[int, int] | None:
        """``(player, type)`` on a square, or ``None`` if it is empty."""
        v = self.cells[square]
        if not v:
            return None
        return owner_of(v), kind_of(v)

    def counts(self, player: int) -> tuple[int, int, int]:
        """How many rock, paper and scissors ``player`` still has."""
        out = [0, 0, 0]
        for s in self.pieces[player]:
            out[kind_of(self.cells[s])] += 1
        return tuple(out)

    # ---------------------------------------------------------------- moves
    def legal_moves(self) -> np.ndarray:
        """Every legal move, as action indices.  Cached per position."""
        if self._legal is not None:
            return self._legal
        if self.winner or self.drawn:
            self._legal = np.empty(0, dtype=np.int32)
            return self._legal
        me = self.to_move
        cells = self.cells
        out: list[int] = []
        for src in self.pieces[me]:
            mine = kind_of(cells[src])
            steps = STEP_LIST[src]
            for d in range(NDIRS):
                dst = steps[d]
                if dst < 0:
                    continue
                v = cells[dst]
                if v:
                    if owner_of(v) == me or not beats(mine, kind_of(v)):
                        continue
                out.append(d * NCELLS + src)
        out.sort()
        self._legal = np.array(out, dtype=np.int32)
        return self._legal

    def is_legal(self, move: int) -> bool:
        if self.winner or self.drawn or not (0 <= move < NACTIONS):
            return False
        src, d = split_action(int(move))
        v = self.cells[src]
        if not v or owner_of(v) != self.to_move:
            return False
        dst = STEP_LIST[src][d]
        if dst < 0:
            return False
        target = self.cells[dst]
        if not target:
            return True
        if owner_of(target) == self.to_move:
            return False
        return beats(kind_of(v), kind_of(target))

    def play(self, move: int) -> None:
        """Move the side to move's piece, resolving any capture."""
        if self.winner or self.drawn:
            raise ValueError("game already decided")
        move = int(move)
        if not self.is_legal(move):
            raise ValueError(f"move {move} is not playable here")
        src, d = split_action(move)
        dst = STEP_LIST[src][d]
        me = self.to_move
        opp = other(me)
        captured = self.cells[dst] != 0
        if captured:
            self.pieces[opp].discard(dst)
        self.cells[dst] = self.cells[src]
        self.cells[src] = 0
        self.pieces[me].discard(src)
        self.pieces[me].add(dst)
        self.move_count += 1
        self.since_capture = 0 if captured else self.since_capture + 1
        self.last_move = move
        self.to_move = opp
        self._legal = None

        if dst == TARGET_BASE[me]:
            self.winner = me
            self.end_reason = "base"
            return
        if not self.pieces[opp]:
            self.winner = me
            self.end_reason = "no_pieces"
            return
        # "The side to move cannot move" is decided here rather than lazily,
        # because the search asks ``is_terminal`` far more often than it plays
        # a move -- and the move generator has to run for the next ply anyway.
        if len(self.legal_moves()) == 0:
            self.winner = me
            self.end_reason = "no_legal_moves"
            return
        # Repetition first: it is the stricter of the two and, unlike the
        # counter, it can fire at any point in a quiet stretch.
        key = self._rep_key()
        self._history = (key,) if captured else self._history + (key,)
        if REPETITION_LIMIT and self._history.count(key) >= REPETITION_LIMIT:
            self.drawn = True
            self.end_reason = "repetition"
            return
        if self.since_capture >= STAGNATION_LIMIT:
            self.drawn = True
            self.end_reason = "stagnation"

    # ------------------------------------------------------- win detection
    def wins_now(self, move: int) -> bool:
        """Would playing ``move`` reach the enemy base?  (No mutation.)"""
        src, d = split_action(int(move))
        return STEP_LIST[src][d] == TARGET_BASE[self.to_move]

    def is_terminal(self) -> bool:
        return self.winner != 0 or self.drawn

    def terminal_value(self) -> float:
        """Value of a *finished* position for the side to move.

        ``-1`` whenever somebody won -- the base was reached, the last piece
        was taken, or this side has no move -- and ``0`` for the stagnation
        draw.  In every case the side on turn is the one that did not win, so
        the sign convention the shared search assumes holds without a special
        case, unlike Ultimate XX where winning loses.
        """
        return -1.0 if self.winner else 0.0

    def result_for(self, player: int) -> float:
        """+1 won, -1 lost, 0 drawn *or still running*."""
        if self.winner == 0:
            return 0.0
        return 1.0 if self.winner == player else -1.0

    def moves_to_stagnation(self) -> int:
        """Half-moves left before the no-capture draw."""
        return max(0, STAGNATION_LIMIT - self.since_capture)

    def win_path(self) -> list[int]:
        """The squares to highlight when the game is over.

        A base capture is a square, so the base and the piece that reached it
        are the same square; a game won by taking the last piece or by leaving
        the opponent stuck has no square to point at, so the winner's base is
        highlighted instead -- the thing they were defending.
        """
        if not self.winner:
            return []
        if self.end_reason == "base":
            return [TARGET_BASE[self.winner]]
        return [BASE_OF[self.winner]]

    # ------------------------------------------------------- representations
    def array(self) -> np.ndarray:
        """``9 x 9`` uint8 of real piece codes, row 0 at the top."""
        return np.frombuffer(bytes(self.cells), dtype=np.uint8).reshape(N, N)

    def canonical_board(self) -> np.ndarray:
        """The board seen from the side to move.

        ``0`` empty, ``1``/``2``/``3`` own rock/paper/scissors, ``4``/``5``/``6``
        the opponent's.  When red is to move the board is rotated 180 degrees
        as well as recoloured, so the side to move always has its base in the
        bottom-left and advances up and to the right.  That is what lets one
        set of feature planes -- and one policy head -- serve both sides.
        """
        return canonicalise(self.array(), self.to_move)

    def canonical_planes(self, in_planes: int | None = None) -> np.ndarray:
        from .features import PLANES, planes_from_boards

        # ``repetitions`` counts the occurrence on the board as well; the planes
        # want how many came *before* it, which is what says how near the draw
        # is.
        return planes_from_boards(
            self.canonical_board()[None],
            PLANES if in_planes is None else in_planes,
            reps=np.array([self.repetitions() - 1]),
        )[0]

    def canonical_policy_mask(self) -> np.ndarray:
        """``(648,)`` bool: the legal moves, in the canonical frame.

        Vectorised rather than a loop over ``to_canonical_move``, because this
        runs once per position per network evaluation and the search evaluates
        millions of them -- and with a branching factor near forty the loop was
        forty Python calls where the whole rest of the encode is four numpy
        ones.
        """
        mask = np.zeros(NACTIONS, dtype=bool)
        legal = self.legal_moves()
        if not len(legal):
            return mask
        if self.to_move == BLUE:
            mask[legal] = True
        else:
            src, d = legal % NCELLS, legal // NCELLS
            mask[_ROT180_DIR_ARR[d] * NCELLS + _ROT180_ARR[src]] = True
        return mask

    def to_canonical_move(self, move: int) -> int:
        """A real move -> its index in the side-to-move's frame."""
        if self.to_move == BLUE:
            return int(move)
        src, d = split_action(int(move))
        return ROT180_DIR[d] * NCELLS + ROT180[src]

    def from_canonical_move(self, move: int) -> int:
        # The map is an involution: rotating twice is the identity.
        return self.to_canonical_move(move)

    # ------------------------------------------------------------ debugging
    def key(self) -> int:
        """A position key for transposition tables.

        The squares, the side to move and the stagnation counter.  The counter
        is in because two identical arrangements a hundred quiet plies apart
        are genuinely different positions: one of them is a draw very soon.
        """
        key = int.from_bytes(bytes(self.cells), "little")
        key = (key << 9) | self.since_capture
        # ...and how close this position is to its third occurrence, because a
        # position seen twice already is a different node from a fresh one: one
        # of them is a draw the moment it recurs.
        key = (key << 2) | min(self._history.count(self._history[-1]), 3)
        return (key << 1) | (self.to_move - 1)

    def __str__(self) -> str:
        arr = self.array()
        lines = []
        for r in range(N):
            row = []
            for c in range(N):
                v = int(arr[r, c])
                if not v:
                    row.append("." if r * N + c not in (BLUE_BASE, RED_BASE) else "#")
                else:
                    letter = TYPE_LETTERS[kind_of(v)]
                    row.append(letter if owner_of(v) == BLUE else letter.lower())
            lines.append(f"{N - r} | " + " ".join(row))
        lines.append("    " + " ".join(chr(ord("a") + c) for c in range(N)))
        side = "blue" if self.to_move == BLUE else "red"
        lines.append(f"to move: {side}  (quiet plies {self.since_capture})")
        return "\n".join(lines)


def canonicalise(board: np.ndarray, to_move: int) -> np.ndarray:
    """Board array -> ``to_move``'s point of view.

    A free function so a *finished* game can be re-rendered from the point of
    view of whoever was on turn at some earlier position -- which is what the
    auxiliary head's target needs.
    """
    board = np.asarray(board, dtype=np.uint8)
    if to_move == RED:
        board = board[::-1, ::-1]
        # Recolour: 1-3 <-> 4-6, and 0 stays 0.
        board = np.where(board == 0, 0,
                         np.where(board <= 3, board + 3, board - 3))
    return np.ascontiguousarray(board.astype(np.uint8))


def final_occupancy(board: np.ndarray, player: int) -> np.ndarray:
    """A *finished* board from ``player``'s side: 0 empty, 1 own, 2 opponent.

    The auxiliary head's target.  The head emits three classes per square, so
    it says *whether* a square is held and by whom at the end of the game, not
    which of the three types is standing there -- which is the part of the
    answer a value head cannot already infer.
    """
    canon = canonicalise(board, player)
    return np.where(canon == 0, 0, np.where(canon <= 3, 1, 2)).astype(np.uint8)


def move_to_str(move: int, board: "RPS2Board | None" = None) -> str:
    """Coordinate notation, as the game itself writes it: ``e2-e3``/``e2xe3``.

    ``board`` is the position the move is played *in*; with it the label can
    say whether the move captures, which is the difference between the two
    forms.  Without it, a quiet move is assumed.
    """
    src, d = split_action(int(move))
    dst = STEP_LIST[src][d]
    sep = "-"
    if board is not None and dst >= 0 and board.cells[dst]:
        sep = "x"
    return f"{square_to_str(src)}{sep}{square_to_str(dst)}"


def square_to_str(square: int) -> str:
    """``72`` -> ``a1``.  Files left to right, ranks counted from the bottom."""
    if square < 0:
        return "??"
    r, c = divmod(int(square), N)
    return f"{chr(ord('a') + c)}{N - r}"


def str_to_square(text: str) -> int:
    text = text.strip().lower()
    c = ord(text[0]) - ord("a")
    rank = int(text[1:])
    if not (0 <= c < N and 1 <= rank <= N):
        raise ValueError(f"{text!r} is not a square on this board")
    return (N - rank) * N + c


def str_to_move(text: str, board: "RPS2Board | None" = None) -> int:
    """``e2-e3`` or ``e2xe3`` (or ``e2e3``) -> an action index."""
    text = text.strip().lower().replace("-", " ").replace("x", " ")
    parts = text.split()
    if len(parts) == 1 and len(parts[0]) == 4:
        parts = [parts[0][:2], parts[0][2:]]
    if len(parts) != 2:
        raise ValueError(f"{text!r} is not a move")
    src, dst = str_to_square(parts[0]), str_to_square(parts[1])
    for d in range(NDIRS):
        if STEP_LIST[src][d] == dst:
            return d * NCELLS + src
    raise ValueError(f"{text!r} is not a king move")
