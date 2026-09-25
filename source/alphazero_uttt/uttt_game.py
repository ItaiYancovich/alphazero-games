"""Ultimate Tic-Tac-Toe game engine.

Board geometry
--------------
Nine 3x3 boards laid out in a 3x3 grid, which is one ``9 x 9`` array of cells
indexed by ``(row, col)`` with **row 0 at the top**, so ``cell = row * 9 + col``
reads like the board looks and matches the convention Hex and Connect Four
already use.  The small board a cell belongs to is ``(row // 3) * 3 + col // 3``
and its slot inside that board is ``(row % 3) * 3 + col % 3``.

Keeping the position as a plain 9x9 grid is what lets the network keep the
ordinary AlphaZero shape: a convolutional trunk over a 9x9 input and one policy
logit per cell, illegal cells masked.  Nothing about the two-level structure
needs a special head -- the mask and the feature planes carry it.

The rule that makes the game
----------------------------
A move's *slot* names the small board the opponent must play in next.  Play the
centre of a board and your opponent must answer in the centre board; play its
top-left and they must answer in the top-left board.  If that board is already
**decided** -- won by somebody, or full -- the constraint lifts and they may
play anywhere.

So the legal set depends on the last move, not only on the marks: the same
arrangement of Xs and Os is a different position depending on where the
opponent has been sent.  ``active`` is that extra state, and every
representation here carries it -- which is why the canonical board has a fourth
value that Hex and Connect Four do not need.

Internals
---------
Nine 9-bit masks per player, one per small board, plus a 9-bit mask of the
boards each player has won and one of the boards that are closed to further
play.  Everything the search asks on every ply -- legal moves, "is this
terminal", "did that win" -- is then a couple of integer ops and a table lookup,
which matters because MCTS copies a position per simulation.

Like Connect Four and unlike Hex, this game **can be drawn**: every small board
decided with no line of three among them is a real result, and a common one.
"""

from __future__ import annotations

import numpy as np

EMPTY = 0
X = 1       # moves first
O = 2       # noqa: E741 -- the piece is called O; nothing reads it as a digit

# The shared code (arena, review, the GUI) speaks of the two sides by the names
# the Hex engine gave them.  Same integers, so a position is interchangeable.
BLACK = X
WHITE = O

N = 9               # board edge, in cells
NBOARDS = 9
NCELLS = N * N
FULL = 0x1FF        # all nine slots of a small board

# The eight lines of three, as slot triples and as 9-bit masks.  They serve
# twice over: inside a small board, and across the grid of small boards.
LINES = ((0, 1, 2), (3, 4, 5), (6, 7, 8),
         (0, 3, 6), (1, 4, 7), (2, 5, 8),
         (0, 4, 8), (2, 4, 6))
LINE_MASKS = tuple(sum(1 << i for i in line) for line in LINES)

# Every 9-bit mask answered once, since "does this hold a line" is asked on
# every ply of every simulation and there are only 512 masks to ask it of.
WON = tuple(any(m & lm == lm for lm in LINE_MASKS) for m in range(512))
# ...and which line it was, for highlighting the three that won.
WON_LINE: tuple[tuple[int, int, int] | None, ...] = tuple(
    next((line for line, lm in zip(LINES, LINE_MASKS) if m & lm == lm), None)
    for m in range(512)
)
POPCOUNT = tuple(bin(m).count("1") for m in range(512))

# cell -> (small board, slot within it), and back.
BOARD_OF = tuple((c // N // 3) * 3 + (c % N) // 3 for c in range(NCELLS))
SLOT_OF = tuple((c // N % 3) * 3 + (c % N) % 3 for c in range(NCELLS))
CELL_AT = tuple(
    tuple(((b // 3) * 3 + k // 3) * N + (b % 3) * 3 + k % 3 for k in range(9))
    for b in range(NBOARDS)
)


def other(player: int) -> int:
    return O if player == X else X


class UltimateBoard:
    """Mutable Ultimate Tic-Tac-Toe position with O(1) win detection."""

    __slots__ = ("rows", "cols", "ncells", "_small", "_meta", "_closed",
                 "active", "to_move", "winner", "move_count", "_win_at")

    def __init__(self, rows: int = N, cols: int = N):
        # ``rows``/``cols`` exist so the shared trainer, evaluator and GUI can
        # talk about this board the way they talk about the others.  The game
        # has exactly one size, and anything else is a mistake worth catching
        # here rather than three layers down inside a reshape.
        if (rows, cols) != (N, N):
            raise ValueError(f"Ultimate Tic-Tac-Toe is {N}x{N}, not {rows}x{cols}")
        self.rows = N
        self.cols = N
        self.ncells = NCELLS
        self._small = [None, [0] * NBOARDS, [0] * NBOARDS]  # indexed by player
        self._meta = [0, 0, 0]      # small boards each player has won
        self._closed = 0            # ...and every board no longer playable
        self.active: int | None = None   # forced small board; None = play anywhere
        self.to_move = X
        self.winner = 0
        self.move_count = 0
        self._win_at: int | None = None  # cell whose placement ended the game

    # ---------------------------------------------------------------- basics
    def copy(self) -> "UltimateBoard":
        new = UltimateBoard.__new__(UltimateBoard)
        new.rows = N
        new.cols = N
        new.ncells = NCELLS
        new._small = [None, self._small[1][:], self._small[2][:]]
        new._meta = self._meta[:]
        new._closed = self._closed
        new.active = self.active
        new.to_move = self.to_move
        new.winner = self.winner
        new.move_count = self.move_count
        new._win_at = self._win_at
        return new

    def small(self, player: int, board: int) -> int:
        """One player's marks in one small board, as a 9-bit mask."""
        return self._small[player][board]

    def meta(self, player: int) -> int:
        """The small boards ``player`` has won, as a 9-bit mask."""
        return self._meta[player]

    @property
    def closed(self) -> int:
        """The small boards nobody may play in any more, as a 9-bit mask."""
        return self._closed

    def occupied(self, board: int) -> int:
        return self._small[X][board] | self._small[O][board]

    def board_owner(self, board: int) -> int:
        """Who won a small board: ``X``, ``O``, or 0 -- drawn *or* still open."""
        if (self._meta[X] >> board) & 1:
            return X
        if (self._meta[O] >> board) & 1:
            return O
        return 0

    def board_state(self, board: int) -> int:
        """One small board's status, as the GUI wants it.

        ``0`` still open, ``1``/``2`` won by that player, ``3`` full and drawn.
        """
        owner = self.board_owner(board)
        if owner:
            return owner
        return 3 if (self._closed >> board) & 1 else 0

    # ---------------------------------------------------------------- moves
    def open_boards(self) -> list[int]:
        return [b for b in range(NBOARDS) if not (self._closed >> b) & 1]

    def legal_moves(self) -> np.ndarray:
        """Every empty cell the constraint allows, as cell indices."""
        if self.winner or self._closed == FULL:
            return np.empty(0, dtype=np.int32)
        # ``active`` is only ever set to a board that is open, so a forced
        # board always has at least one empty slot.
        boards = [self.active] if self.active is not None else self.open_boards()
        cells = []
        for b in boards:
            occ = self.occupied(b)
            row = CELL_AT[b]
            cells.extend(row[k] for k in range(9) if not (occ >> k) & 1)
        cells.sort()
        return np.array(cells, dtype=np.int32)

    def is_legal(self, move: int) -> bool:
        if self.winner or not (0 <= move < NCELLS):
            return False
        b = BOARD_OF[move]
        if (self._closed >> b) & 1:
            return False
        if self.active is not None and b != self.active:
            return False
        return not (self.occupied(b) >> SLOT_OF[move]) & 1

    def play(self, move: int) -> None:
        """Place the side to move's mark on ``move``."""
        if self.winner:
            raise ValueError("game already decided")
        if not self.is_legal(move):
            raise ValueError(f"cell {move} is not playable here")
        b, k = BOARD_OF[move], SLOT_OF[move]
        player = self.to_move
        mine = self._small[player][b] | (1 << k)
        self._small[player][b] = mine
        self.move_count += 1
        if WON[mine]:
            self._meta[player] |= 1 << b
            self._closed |= 1 << b
            if WON[self._meta[player]]:
                self.winner = player
                self._win_at = move
        elif mine | self._small[other(player)][b] == FULL:
            self._closed |= 1 << b       # full with no line: drawn, and shut
        self.to_move = other(player)
        # The slot played sends the opponent to the small board of the same
        # index -- unless that board is decided, in which case they are free.
        self.active = None if (self._closed >> k) & 1 else k

    # ------------------------------------------------------- win detection
    def wins_small(self, player: int, move: int) -> bool:
        """Would ``player`` playing ``move`` complete that small board?"""
        b, k = BOARD_OF[move], SLOT_OF[move]
        return WON[self._small[player][b] | (1 << k)]

    def would_win(self, player: int, move: int) -> bool:
        """Would ``player`` playing ``move`` win the *game*?  (No mutation.)"""
        b, k = BOARD_OF[move], SLOT_OF[move]
        if not WON[self._small[player][b] | (1 << k)]:
            return False
        return WON[self._meta[player] | (1 << b)]

    def is_terminal(self) -> bool:
        return self.winner != 0 or self._closed == FULL

    def terminal_value(self) -> float:
        """Value of a finished position for the side to move.

        ``-1`` when the mark just played completed the third small board of a
        line (the side on turn has lost) and ``0`` when every small board is
        decided with no line among them -- the draw Hex does not have and that
        the shared search needs to be told about.
        """
        return -1.0 if self.winner else 0.0

    def result_for(self, player: int) -> float:
        """+1 won, -1 lost, 0 drawn *or still running*."""
        if self.winner == 0:
            return 0.0
        return 1.0 if self.winner == player else -1.0

    def winning_boards(self) -> list[int]:
        """The three small boards that won the game, for highlighting."""
        if not self.winner:
            return []
        line = WON_LINE[self._meta[self.winner]]
        return list(line) if line else []

    def winning_cells(self) -> list[int]:
        """The nine marks that won: three in a row inside each winning board."""
        if not self.winner:
            return []
        out: list[int] = []
        for b in self.winning_boards():
            line = WON_LINE[self._small[self.winner][b]]
            if line:
                out.extend(CELL_AT[b][k] for k in line)
        return out

    # ------------------------------------------------------- representations
    def array(self) -> np.ndarray:
        """``9 x 9`` uint8 of real colours, row 0 at the top."""
        out = np.zeros(NCELLS, dtype=np.uint8)
        for player in (X, O):
            for b in range(NBOARDS):
                mask = self._small[player][b]
                row = CELL_AT[b]
                while mask:
                    low = mask & -mask
                    out[row[low.bit_length() - 1]] = player
                    mask ^= low
        return out.reshape(N, N)

    def playable_mask(self) -> np.ndarray:
        """``9 x 9`` bool: the cells that may be played right now."""
        return self.canonical_policy_mask().reshape(N, N)

    def canonical_board(self) -> np.ndarray:
        """The board seen from the side to move.

        ``0`` empty *and playable*, ``1`` own mark, ``2`` opponent's mark, and
        ``3`` empty but out of reach this turn.

        The fourth value is what Hex and Connect Four do not need.  A Connect
        Four disc may only land on the lowest empty cell of a column, but the
        arrangement of discs already says which cell that is; the board a
        player is confined to here follows from the *last move*, so a stored
        position recording only the marks would be ambiguous.  Encoding the
        constraint into the board keeps one array sufficient: the network's
        legality mask is still "the cells that read 0", and a stored training
        example is still a single ``uint8`` grid.
        """
        return canonicalise(self.array(), self.to_move, self.playable_mask())

    def canonical_planes(self, in_planes: int | None = None) -> np.ndarray:
        from .features import PLANES, planes_from_boards

        return planes_from_boards(
            self.canonical_board()[None], PLANES if in_planes is None else in_planes
        )[0]

    def canonical_policy_mask(self) -> np.ndarray:
        """Which policy slots are actually playable, in the canonical frame.

        Identical to "the canonical board reads 0 here", by construction; the
        shared evaluator intersects the two and is welcome to.
        """
        mask = np.zeros(NCELLS, dtype=bool)
        mask[self.legal_moves()] = True
        return mask

    def to_canonical_move(self, move: int) -> int:
        """Identity: canonicalising swaps marks only, never geometry."""
        return move

    def from_canonical_move(self, move: int) -> int:
        return move

    # ------------------------------------------------------------ debugging
    def key(self) -> int:
        """A position key for transposition tables.

        Both players' marks, the forced board and the side to move -- all four,
        because the marks alone do not determine what may be played next.
        """
        key = 0
        for player in (X, O):
            for b in range(NBOARDS):
                key = (key << 9) | self._small[player][b]
        key = (key << 4) | (NBOARDS if self.active is None else self.active)
        return (key << 1) | (self.to_move - 1)

    def __str__(self) -> str:
        symbols = {EMPTY: ".", X: "X", O: "O"}
        arr = self.array()
        lines = []
        for r in range(N):
            cells = [symbols[int(v)] for v in arr[r]]
            lines.append(" | ".join(" ".join(cells[c:c + 3]) for c in (0, 3, 6)))
            if r in (2, 5):
                lines.append("-" * len(lines[-1]))
        where = "anywhere" if self.active is None else f"board {self.active}"
        lines.append(f"to move: {'X' if self.to_move == X else 'O'} ({where})")
        return "\n".join(lines)


def canonicalise(board: np.ndarray, to_move: int, playable: np.ndarray) -> np.ndarray:
    """Board array + legality -> ``to_move``'s point of view.

    A free function so a *finished* game can be re-rendered from the point of
    view of whoever was to move at some earlier position.
    """
    board = np.asarray(board)
    out = np.where(board == to_move, 1, np.where(board == EMPTY, 0, 2))
    out = np.where((board == EMPTY) & ~np.asarray(playable), 3, out)
    return out.astype(np.uint8)


def final_occupancy(board: np.ndarray, player: int) -> np.ndarray:
    """A *finished* board from ``player``'s side: 0 empty, 1 own, 2 opponent.

    The auxiliary head's target.  Playability has no meaning in a position
    nobody can move in, so this is the plain three-class view rather than the
    four-value encoding the network's *input* uses -- the head emits three
    channels per cell and has no fourth class to put "out of reach" in.
    """
    board = np.asarray(board)
    return canonicalise(board, player, np.ones(board.shape, dtype=bool))


def move_to_str(move: int, n: int = N) -> str:
    """Coordinates over the whole 9x9 grid: file a-i, rank 1-9 from the top.

    Naming the small board and the slot separately (``4/7``, and every author
    picks a different order for the two halves) reads worse than one square --
    and the square already says which board it is in.
    """
    r, c = divmod(move, n)
    return f"{chr(ord('a') + c)}{r + 1}"


def str_to_move(text: str, board: "UltimateBoard | None" = None) -> int:
    """``e5`` -> the cell index.  ``board`` is accepted for symmetry only."""
    text = text.strip().lower()
    c = ord(text[0]) - ord("a")
    r = int(text[1:]) - 1
    if not (0 <= c < N and 0 <= r < N):
        raise ValueError(f"{text!r} is not a square on this board")
    return r * N + c
