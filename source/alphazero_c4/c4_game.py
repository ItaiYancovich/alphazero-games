"""Connect Four game engine.

Board geometry
--------------
``rows x cols`` (6 x 7 by default), indexed by ``(row, col)`` with **row 0 at
the top**, so ``cell = row * cols + col`` reads like the board looks and matches
the convention the Hex engine and the GUI already use.

A move is the *cell that the disc lands in*, not the column.  That is what lets
the network keep the plain AlphaZero policy head -- one logit per cell, illegal
cells masked -- with no Connect-Four-specific head: at most one cell per column
is ever legal, so the mask does the column arithmetic.

Internals
---------
The position is two bitboards, one per player, in the standard Connect Four
layout: ``cols`` columns of ``rows + 1`` bits, bottom-up, with the top bit of
each column left permanently empty.  That sentinel row is what makes the win
test four shifts and a mask -- without it, a shift by ``rows`` would wrap the
top of one column onto the bottom of the next and invent horizontal fours.

Bitboards matter here more than they would elsewhere: MCTS copies a position on
every simulation and asks "is this terminal" on every ply, and both are a couple
of integer ops this way.  The ``rows x cols`` array is materialised only when
something actually wants to look at the board (the network encoder, the GUI).

Unlike Hex, Connect Four **can be drawn** -- a full board with no line of four
is a legitimate result, and a real one between two strong players.
"""

from __future__ import annotations

import numpy as np

EMPTY = 0
RED = 1     # moves first
YELLOW = 2

# The shared code (arena, review, the GUI) speaks of the two sides by the names
# the Hex engine gave them.  Same integers, so a position is interchangeable.
BLACK = RED
WHITE = YELLOW

ROWS = 6
COLS = 7


def other(player: int) -> int:
    return YELLOW if player == RED else RED


class Connect4Board:
    """Mutable Connect Four position with O(1) win detection."""

    __slots__ = ("rows", "cols", "ncells", "_bb", "_heights", "to_move", "winner",
                 "move_count", "_h1", "_full", "_win_at")

    def __init__(self, rows: int = ROWS, cols: int = COLS):
        self.rows = rows
        self.cols = cols
        self.ncells = rows * cols
        self._h1 = rows + 1              # bits per column, including the sentinel
        self._bb = [0, 0, 0]             # indexed by player; slot 0 unused
        self._heights = [0] * cols       # discs already in each column
        self.to_move = RED
        self.winner = 0
        self.move_count = 0
        self._win_at: int | None = None  # cell whose placement ended the game
        self._full = rows * cols

    # ---------------------------------------------------------------- basics
    def copy(self) -> "Connect4Board":
        new = Connect4Board.__new__(Connect4Board)
        new.rows = self.rows
        new.cols = self.cols
        new.ncells = self.ncells
        new._h1 = self._h1
        new._bb = self._bb[:]
        new._heights = self._heights[:]
        new.to_move = self.to_move
        new.winner = self.winner
        new.move_count = self.move_count
        new._win_at = self._win_at
        new._full = self._full
        return new

    def bitboard(self, player: int) -> int:
        """One player's discs, as the raw bitboard the heuristics work on."""
        return self._bb[player]

    @property
    def bits_per_column(self) -> int:
        """Column stride, i.e. ``rows + 1``: the sentinel row is included."""
        return self._h1

    def height(self, col: int) -> int:
        return self._heights[col]

    # ------------------------------------------------------ index arithmetic
    def bit_of(self, row: int, col: int) -> int:
        """Bit index of ``(row, col)``; row 0 is the top of the board."""
        return col * self._h1 + (self.rows - 1 - row)

    def cell_of_bit(self, bit: int) -> int:
        col, height = divmod(bit, self._h1)
        return (self.rows - 1 - height) * self.cols + col

    def landing_cell(self, col: int) -> int | None:
        """Where a disc dropped in ``col`` would come to rest, or None if full."""
        height = self._heights[col]
        if height >= self.rows:
            return None
        return (self.rows - 1 - height) * self.cols + col

    def column_of(self, move: int) -> int:
        return move % self.cols

    # ---------------------------------------------------------------- moves
    def legal_moves(self) -> np.ndarray:
        """The landing cell of every column that is not full."""
        if self.winner or self.move_count >= self._full:
            return np.empty(0, dtype=np.int32)
        cells = [self.landing_cell(c) for c in range(self.cols)
                 if self._heights[c] < self.rows]
        return np.array(cells, dtype=np.int32)

    def legal_columns(self) -> list[int]:
        if self.winner or self.move_count >= self._full:
            return []
        return [c for c in range(self.cols) if self._heights[c] < self.rows]

    def is_legal(self, move: int) -> bool:
        if self.winner or not (0 <= move < self.ncells):
            return False
        col = move % self.cols
        return self.landing_cell(col) == move

    def play(self, move: int) -> None:
        """Drop a disc in ``move``'s column for the side to move."""
        if self.winner:
            raise ValueError("game already decided")
        col = move % self.cols
        landing = self.landing_cell(col)
        if landing is None:
            raise ValueError(f"column {col} is full")
        if landing != move:
            raise ValueError(f"cell {move} is not the landing square of column {col}")
        player = self.to_move
        bit = self.bit_of(move // self.cols, col)
        self._bb[player] |= 1 << bit
        self._heights[col] += 1
        self.move_count += 1
        if self._connects(self._bb[player]):
            self.winner = player
            self._win_at = move
        self.to_move = other(player)

    def play_column(self, col: int) -> None:
        landing = self.landing_cell(col)
        if landing is None:
            raise ValueError(f"column {col} is full")
        self.play(landing)

    # ------------------------------------------------------- win detection
    def _connects(self, bb: int) -> bool:
        """Does this bitboard contain four in a row?

        Four directions, two shifts each: ``bb & (bb >> s)`` marks every pair,
        and shifting that by ``2s`` and intersecting marks every four.
        """
        h1 = self._h1
        for shift in (1, h1, h1 - 1, h1 + 1):
            m = bb & (bb >> shift)
            if m & (m >> (2 * shift)):
                return True
        return False

    def would_win(self, player: int, move: int) -> bool:
        """Would ``player`` playing ``move`` complete a four?  (No mutation.)"""
        col = move % self.cols
        if self.landing_cell(col) != move:
            return False
        bit = self.bit_of(move // self.cols, col)
        return self._connects(self._bb[player] | (1 << bit))

    def is_terminal(self) -> bool:
        return self.winner != 0 or self.move_count >= self._full

    def terminal_value(self) -> float:
        """Value of a finished position for the side to move.

        ``-1`` when the disc just played completed a four (the side on turn has
        lost) and ``0`` when the board filled up with no line -- the draw that
        Hex does not have and that the shared search needs to be told about.
        """
        return -1.0 if self.winner else 0.0

    def result_for(self, player: int) -> float:
        """+1 won, -1 lost, 0 drawn *or still running*."""
        if self.winner == 0:
            return 0.0
        return 1.0 if self.winner == player else -1.0

    def winning_cells(self) -> list[int]:
        """The four (or more) cells that won the game, for highlighting."""
        if not self.winner:
            return []
        bb = self._bb[self.winner]
        h1 = self._h1
        for shift in (1, h1, h1 - 1, h1 + 1):
            m = bb & (bb >> shift)
            m &= m >> (2 * shift)
            if m:
                start = (m & -m).bit_length() - 1
                return [self.cell_of_bit(start + k * shift) for k in range(4)]
        return []

    # ------------------------------------------------------- representations
    def array(self) -> np.ndarray:
        """``rows x cols`` uint8 of real colours, row 0 at the top."""
        out = np.zeros(self.ncells, dtype=np.uint8)
        for player in (RED, YELLOW):
            bb = self._bb[player]
            while bb:
                low = bb & -bb
                out[self.cell_of_bit(low.bit_length() - 1)] = player
                bb ^= low
        return out.reshape(self.rows, self.cols)

    def canonical_board(self) -> np.ndarray:
        """The board seen from the side to move: 0 empty, 1 own, 2 opponent.

        Only the colours are swapped.  Connect Four has no symmetry that maps
        one player onto the other (Hex's transpose has no analogue here), so
        this is all "canonical" can mean -- and it is enough for the network to
        only ever learn one point of view.
        """
        return canonicalise(self.array(), self.to_move)

    def canonical_planes(self, in_planes: int | None = None) -> np.ndarray:
        from .features import PLANES, planes_from_boards

        return planes_from_boards(
            self.canonical_board()[None], PLANES if in_planes is None else in_planes
        )[0]

    def canonical_policy_mask(self) -> np.ndarray:
        """Which policy slots are actually playable, in the canonical frame.

        Empty is not the same as legal here: only the lowest empty cell of a
        column can be played, and letting the network spread prior mass over
        the 30-odd floating cells above them would waste most of the policy.
        """
        mask = np.zeros(self.ncells, dtype=bool)
        for move in self.legal_moves():
            mask[int(move)] = True
        return mask

    def to_canonical_move(self, move: int) -> int:
        """Identity: canonicalising swaps colours only, never geometry."""
        return move

    def from_canonical_move(self, move: int) -> int:
        return move

    # ------------------------------------------------------------ debugging
    def key(self) -> int:
        """A position key for transposition tables.

        The standard Connect Four trick: one player's stones plus the occupancy
        of both determines the position, and adding a bottom sentinel makes the
        sum injective.
        """
        occupied = self._bb[RED] | self._bb[YELLOW]
        bottom = 0
        for c in range(self.cols):
            bottom |= 1 << (c * self._h1)
        return self._bb[self.to_move] + occupied + bottom

    def __str__(self) -> str:
        symbols = {EMPTY: ".", RED: "R", YELLOW: "Y"}
        arr = self.array()
        lines = [" ".join(symbols[int(v)] for v in row) for row in arr]
        lines.append(" ".join(str(c + 1) for c in range(self.cols)))
        lines.append(f"to move: {'RED' if self.to_move == RED else 'YELLOW'}")
        return "\n".join(lines)


def canonicalise(board: np.ndarray, to_move: int) -> np.ndarray:
    """Board array -> ``to_move``'s point of view (0 empty, 1 own, 2 opponent).

    A free function so a *finished* game can be re-rendered from the point of
    view of whoever was to move at some earlier position.
    """
    if to_move == RED:
        out = np.where(board == RED, 1, np.where(board == YELLOW, 2, 0))
    else:
        out = np.where(board == YELLOW, 1, np.where(board == RED, 2, 0))
    return out.astype(np.uint8)


def move_to_str(move: int, cols: int = COLS) -> str:
    """Connect Four notation: the column, numbered from 1."""
    return str(move % cols + 1)


def str_to_move(text: str, board: Connect4Board) -> int:
    """A column number (1-based) -> the cell a disc dropped there would fill."""
    col = int(text.strip()) - 1
    landing = board.landing_cell(col)
    if landing is None:
        raise ValueError(f"column {col + 1} is full")
    return landing


def mirror_move(move: int, cols: int = COLS) -> int:
    """The same move on the left-right mirrored board."""
    row, col = divmod(move, cols)
    return row * cols + (cols - 1 - col)
