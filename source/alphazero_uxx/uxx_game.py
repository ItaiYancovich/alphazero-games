"""Ultimate XX game engine -- Ultimate Tic-Tac-Toe played misere, with one mark.

The rules, in full
------------------
Nine 3x3 boards in a 3x3 grid, exactly the geometry Ultimate Tic-Tac-Toe uses.
Two differences, and between them they change everything:

* **There is only one mark.**  Both players place X.  A small board is *claimed*
  by whoever completes three Xs in a row, column or diagonal inside it -- not by
  whoever owns the marks, because nobody owns them.  Every mark on the board is
  available to both players and to every line.
* **Winning the big board loses the match.**  Claim three small boards in a
  line and you have lost; your opponent wins.  Nothing else ends the game
  except running out of small boards.

The sending rule is Ultimate Tic-Tac-Toe's, unchanged: the *slot* a move
occupies names the small board the opponent must answer in, and if that board is
already claimed the constraint lifts and they may play anywhere.

What the shared mark does to the game
-------------------------------------
In Ultimate Tic-Tac-Toe a line belongs to whoever built it, so a small board is
a race.  Here a line is built by both players together and collected by whoever
happens to be standing there when it closes -- so a small board is not a race
but a **minefield**.  The cells that would complete a line are *hot*: whoever
plays one claims the board, wanted or not.  A board whose every empty cell is
hot is *poisoned* -- being sent there is not a choice at all.

That is the whole game.  You do not build; you steer.  Losing means being the
one who has to take the third board of a line, and winning means leaving your
opponent nowhere to stand.

Termination
-----------
A 3x3 board cannot hold seven Xs without a line -- six is the maximum, and the
seventh always closes one (``tests/test_uxx.py`` checks all 512 masks).  So a
small board is always claimed before it fills, and *claimed* and *closed* are
the same thing here: there is no drawn small board, which Ultimate Tic-Tac-Toe
does have.  Every move therefore brings the game nearer to nine claimed boards,
and nine claimed boards with no line among them is a draw -- rarer than in
Ultimate Tic-Tac-Toe but real: 4/5 splits with neither side holding a line exist
(``{0,1,5,6}`` against ``{2,3,4,7,8}``, say).

Internals
---------
One 9-bit mask per small board for the marks, and a 9-bit mask per player for
the boards they have claimed.  Everything the search asks on every ply is then a
couple of integer ops and a table lookup, which matters because MCTS copies a
position per simulation.

Who placed each mark is *also* tracked, in ``_by``, and is used for nothing but
display: the rules never look at it, because they cannot -- a mark is a mark.
"""

from __future__ import annotations

import numpy as np

EMPTY = 0
FIRST = 1       # moves first
SECOND = 2

# The shared code (arena, review, the GUI) speaks of the two sides by the names
# the Hex engine gave them.  Same integers, so a position is interchangeable.
BLACK = FIRST
WHITE = SECOND

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
# ...and which line it was, for highlighting the three that closed a board.
WON_LINE: tuple[tuple[int, int, int] | None, ...] = tuple(
    next((line for line, lm in zip(LINES, LINE_MASKS) if m & lm == lm), None)
    for m in range(512)
)
POPCOUNT = tuple(bin(m).count("1") for m in range(512))

# The *hot* slots of a small board: empty slots where placing a mark closes a
# line, and so claims the board for whoever plays there.  This is the table the
# whole game turns on, and like the others it is answered once for all 512
# masks rather than recomputed per node.
HOT = tuple(
    0 if WON[m] else
    sum(1 << k for k in range(9) if not (m >> k) & 1 and WON[m | (1 << k)])
    for m in range(512)
)

# cell -> (small board, slot within it), and back.
BOARD_OF = tuple((c // N // 3) * 3 + (c % N) // 3 for c in range(NCELLS))
SLOT_OF = tuple((c // N % 3) * 3 + (c % N) % 3 for c in range(NCELLS))
CELL_AT = tuple(
    tuple(((b // 3) * 3 + k // 3) * N + (b % 3) * 3 + k % 3 for k in range(9))
    for b in range(NBOARDS)
)


def other(player: int) -> int:
    return SECOND if player == FIRST else FIRST


class UltimateXXBoard:
    """Mutable Ultimate XX position with O(1) claim and loss detection."""

    __slots__ = ("rows", "cols", "ncells", "_marks", "_by", "_claim",
                 "active", "to_move", "winner", "loser", "move_count", "_lose_at")

    def __init__(self, rows: int = N, cols: int = N):
        # ``rows``/``cols`` exist so the shared trainer, evaluator and GUI can
        # talk about this board the way they talk about the others.  The game
        # has exactly one size, and anything else is a mistake worth catching
        # here rather than three layers down inside a reshape.
        if (rows, cols) != (N, N):
            raise ValueError(f"Ultimate XX is {N}x{N}, not {rows}x{cols}")
        self.rows = N
        self.cols = N
        self.ncells = NCELLS
        self._marks = [0] * NBOARDS          # every mark, whoever placed it
        self._by = [None, [0] * NBOARDS, [0] * NBOARDS]   # ...and who did, for display
        self._claim = [0, 0, 0]              # small boards each player has claimed
        self.active: int | None = None       # forced small board; None = play anywhere
        self.to_move = FIRST
        self.winner = 0
        self.loser = 0                       # who completed a line of small boards
        self.move_count = 0
        self._lose_at: int | None = None     # cell whose placement ended the game

    # ---------------------------------------------------------------- basics
    def copy(self) -> "UltimateXXBoard":
        new = UltimateXXBoard.__new__(UltimateXXBoard)
        new.rows = N
        new.cols = N
        new.ncells = NCELLS
        new._marks = self._marks[:]
        new._by = [None, self._by[1][:], self._by[2][:]]
        new._claim = self._claim[:]
        new.active = self.active
        new.to_move = self.to_move
        new.winner = self.winner
        new.loser = self.loser
        new.move_count = self.move_count
        new._lose_at = self._lose_at
        return new

    def marks(self, board: int) -> int:
        """Every mark in one small board, as a 9-bit mask.  Nobody owns them."""
        return self._marks[board]

    def placed_by(self, player: int, board: int) -> int:
        """The marks in one small board that ``player`` put there.

        Display only.  No rule reads this: a line of three counts whoever built
        it, which is what makes this game the game it is.
        """
        return self._by[player][board]

    def claims(self, player: int) -> int:
        """The small boards ``player`` has claimed, as a 9-bit mask."""
        return self._claim[player]

    @property
    def closed(self) -> int:
        """The small boards nobody may play in any more, as a 9-bit mask.

        Identical to "claimed by somebody": a small board cannot fill without
        closing a line, so there is no third way for one to end.
        """
        return self._claim[FIRST] | self._claim[SECOND]

    def occupied(self, board: int) -> int:
        return self._marks[board]

    def board_owner(self, board: int) -> int:
        """Who claimed a small board: ``FIRST``, ``SECOND``, or 0 for still open."""
        if (self._claim[FIRST] >> board) & 1:
            return FIRST
        if (self._claim[SECOND] >> board) & 1:
            return SECOND
        return 0

    def board_state(self, board: int) -> int:
        """One small board's status, as the GUI wants it.

        ``0`` still open, ``1``/``2`` claimed by that player.  There is no
        drawn-board value: a board is claimed before it can fill.
        """
        return self.board_owner(board)

    def hot(self, board: int) -> int:
        """The empty slots of ``board`` that would claim it, as a 9-bit mask."""
        return HOT[self._marks[board]]

    def is_poisoned(self, board: int) -> bool:
        """Is every empty slot of an open board a claiming one?

        Whoever is sent to a poisoned board takes it whether they want it or
        not, which is how a position is won and lost here.
        """
        if (self.closed >> board) & 1:
            return False
        mask = self._marks[board]
        return HOT[mask] == (FULL & ~mask)

    # ---------------------------------------------------------------- moves
    def open_boards(self) -> list[int]:
        closed = self.closed
        return [b for b in range(NBOARDS) if not (closed >> b) & 1]

    def legal_moves(self) -> np.ndarray:
        """Every empty cell the constraint allows, as cell indices."""
        if self.loser or self.closed == FULL:
            return np.empty(0, dtype=np.int32)
        # ``active`` is only ever set to a board that is open, and an open
        # board always has an empty slot -- six marks is the most a 3x3 can
        # hold without a line -- so a forced board is never a dead end.
        boards = [self.active] if self.active is not None else self.open_boards()
        cells = []
        for b in boards:
            occ = self._marks[b]
            row = CELL_AT[b]
            cells.extend(row[k] for k in range(9) if not (occ >> k) & 1)
        cells.sort()
        return np.array(cells, dtype=np.int32)

    def is_legal(self, move: int) -> bool:
        if self.loser or not (0 <= move < NCELLS):
            return False
        b = BOARD_OF[move]
        if (self.closed >> b) & 1:
            return False
        if self.active is not None and b != self.active:
            return False
        return not (self._marks[b] >> SLOT_OF[move]) & 1

    def play(self, move: int) -> None:
        """Place a mark on ``move`` for the side to move."""
        if self.loser:
            raise ValueError("game already decided")
        if not self.is_legal(move):
            raise ValueError(f"cell {move} is not playable here")
        b, k = BOARD_OF[move], SLOT_OF[move]
        player = self.to_move
        marks = self._marks[b] | (1 << k)
        self._marks[b] = marks
        self._by[player][b] |= 1 << k
        self.move_count += 1
        if WON[marks]:
            # The line closed, so this player takes the board -- which is not
            # necessarily good news for them.
            self._claim[player] |= 1 << b
            if WON[self._claim[player]]:
                self.loser = player
                self.winner = other(player)
                self._lose_at = move
        self.to_move = other(player)
        # The slot played sends the opponent to the small board of the same
        # index -- unless that board is claimed, in which case they are free.
        self.active = None if (self.closed >> k) & 1 else k

    # ------------------------------------------------------ loss detection
    def claims_small(self, move: int) -> bool:
        """Would playing ``move`` close a line and claim its small board?

        No player argument: the mark is the same either way, so the answer is a
        property of the square, and the *mover* is simply whoever gets it.
        """
        return WON[self._marks[BOARD_OF[move]] | (1 << SLOT_OF[move])]

    def would_lose(self, player: int, move: int) -> bool:
        """Would ``player`` playing ``move`` lose the game?  (No mutation.)"""
        b, k = BOARD_OF[move], SLOT_OF[move]
        if not WON[self._marks[b] | (1 << k)]:
            return False
        return WON[self._claim[player] | (1 << b)]

    def safe_moves(self) -> list[int]:
        """The legal moves that do not lose the game on the spot."""
        me = self.to_move
        return [int(m) for m in self.legal_moves() if not self.would_lose(me, int(m))]

    def is_terminal(self) -> bool:
        return self.loser != 0 or self.closed == FULL

    def terminal_value(self) -> float:
        """Value of a finished position for the side to move.

        ``+1`` -- not ``-1``, which is what every other game here returns --
        when the mark just played completed a line of small boards: the player
        who did that has *lost*, so the side now on turn has won without
        touching the board.  ``0`` when all nine boards are claimed with no line
        among them.  The value is never negative: you can only lose on your own
        move.
        """
        return 1.0 if self.loser else 0.0

    def result_for(self, player: int) -> float:
        """+1 won, -1 lost, 0 drawn *or still running*."""
        if self.loser == 0:
            return 0.0
        return -1.0 if self.loser == player else 1.0

    def losing_boards(self) -> list[int]:
        """The three small boards whose line lost the game, for highlighting."""
        if not self.loser:
            return []
        line = WON_LINE[self._claim[self.loser]]
        return list(line) if line else []

    def losing_cells(self) -> list[int]:
        """The nine marks that lost it: the line inside each of those boards.

        A claimed board is closed, so its marks never change again and the line
        read back now is the line that closed it.
        """
        if not self.loser:
            return []
        out: list[int] = []
        for b in self.losing_boards():
            line = WON_LINE[self._marks[b]]
            if line:
                out.extend(CELL_AT[b][k] for k in line)
        return out

    # ------------------------------------------------------- representations
    def array(self) -> np.ndarray:
        """``9 x 9`` uint8 of who *placed* each mark, row 0 at the top.

        Every mark is an X and the rules do not care who put it there; this is
        for the move list and the review scrubber, which rebuild a position from
        the moves and want to show it the way it was played.
        """
        out = np.zeros(NCELLS, dtype=np.uint8)
        for player in (FIRST, SECOND):
            for b in range(NBOARDS):
                mask = self._by[player][b]
                row = CELL_AT[b]
                while mask:
                    low = mask & -mask
                    out[row[low.bit_length() - 1]] = player
                    mask ^= low
        return out.reshape(N, N)

    def mark_array(self) -> np.ndarray:
        """``9 x 9`` bool: where the marks are, which is all the rules see."""
        out = np.zeros(NCELLS, dtype=bool)
        for b in range(NBOARDS):
            mask = self._marks[b]
            row = CELL_AT[b]
            while mask:
                low = mask & -mask
                out[row[low.bit_length() - 1]] = True
                mask ^= low
        return out.reshape(N, N)

    def playable_mask(self) -> np.ndarray:
        """``9 x 9`` bool: the cells that may be played right now."""
        return self.canonical_policy_mask().reshape(N, N)

    def canonical_board(self) -> np.ndarray:
        """The board seen from the side to move.

        Five values, where Hex and Connect Four need three::

            0  empty and playable
            1  a mark, in a small board the side to move has claimed
            2  a mark, in a small board the opponent has claimed
            3  empty but out of reach this turn
            4  a mark, in a small board nobody has claimed yet

        The extra two both carry something the marks cannot say.  Value 3 is the
        constraint, as in Ultimate Tic-Tac-Toe: which board you are confined to
        follows from the *last move*, not from the position, so a grid recording
        only the marks would be ambiguous.  Values 1, 2 and 4 carry the meta
        board, which here is genuinely new information -- a mark has no owner,
        so *who claimed a small board* cannot be read off its marks the way it
        can when the two sides play different symbols.  A claimed board always
        holds at least the three marks of the line that closed it, so the claim
        always survives the round trip.

        All of it in one array is what makes this the whole position: two
        positions with the same grid are the same position, which is what the
        GUI's "has the board changed" check and any transposition table want.
        """
        return canonicalise(self.mark_array(), self.playable_mask(),
                            self._claim[self.to_move], self._claim[other(self.to_move)])

    def canonical_policy_mask(self) -> np.ndarray:
        """Which policy slots are actually playable, in the canonical frame.

        Identical to "the canonical board reads 0 here", by construction; the
        shared evaluator intersects the two and is welcome to.
        """
        mask = np.zeros(NCELLS, dtype=bool)
        mask[self.legal_moves()] = True
        return mask

    def to_canonical_move(self, move: int) -> int:
        """Identity: canonicalising relabels claims only, never geometry."""
        return move

    def from_canonical_move(self, move: int) -> int:
        return move

    # ------------------------------------------------------------ debugging
    def key(self) -> int:
        """A position key for transposition tables.

        The marks, both claim masks, the forced board and the side to move.
        The claims are stored rather than derived because they cannot be
        derived: which of two identical piles of Xs belongs to whom is a fact
        about the order they were played in.
        """
        key = 0
        for b in range(NBOARDS):
            key = (key << 9) | self._marks[b]
        key = (key << 9) | self._claim[FIRST]
        key = (key << 9) | self._claim[SECOND]
        key = (key << 4) | (NBOARDS if self.active is None else self.active)
        return (key << 1) | (self.to_move - 1)

    def __str__(self) -> str:
        arr = self.mark_array()
        owner = {0: " ", FIRST: "1", SECOND: "2"}
        lines = []
        for r in range(N):
            cells = ["X" if v else "." for v in arr[r]]
            lines.append(" | ".join(" ".join(cells[c:c + 3]) for c in (0, 3, 6)))
            if r in (2, 5):
                lines.append("-" * len(lines[-1]))
        claimed = "".join(owner[self.board_owner(b)] for b in range(NBOARDS))
        where = "anywhere" if self.active is None else f"board {self.active}"
        lines.append(f"boards: {claimed}")
        lines.append(f"to move: {self.to_move} ({where})")
        if self.loser:
            lines.append(f"player {self.loser} completed a line and lost")
        return "\n".join(lines)


def canonicalise(marks: np.ndarray, playable: np.ndarray,
                 mine: int, theirs: int) -> np.ndarray:
    """Marks + legality + the two claim masks -> the side to move's point of view.

    A free function so a *finished* game can be re-rendered from the point of
    view of whoever was to move at some earlier position.  ``mine`` and
    ``theirs`` are 9-bit masks over the small boards.
    """
    marks = np.asarray(marks).astype(bool)
    claim = np.zeros((3, 3), dtype=np.uint8)
    for b in range(NBOARDS):
        if (mine >> b) & 1:
            claim[b // 3, b % 3] = 1
        elif (theirs >> b) & 1:
            claim[b // 3, b % 3] = 2
    per_cell = np.repeat(np.repeat(claim, 3, axis=0), 3, axis=1)
    out = np.where(marks, np.where(per_cell == 0, 4, per_cell), 0)
    out = np.where(~marks & ~np.asarray(playable), 3, out)
    return out.astype(np.uint8)


def move_to_str(move: int, n: int = N) -> str:
    """Coordinates over the whole 9x9 grid: file a-i, rank 1-9 from the top."""
    r, c = divmod(move, n)
    return f"{chr(ord('a') + c)}{r + 1}"


def str_to_move(text: str, board: "UltimateXXBoard | None" = None) -> int:
    """``e5`` -> the cell index.  ``board`` is accepted for symmetry only."""
    text = text.strip().lower()
    c = ord(text[0]) - ord("a")
    r = int(text[1:]) - 1
    if not (0 <= c < N and 0 <= r < N):
        raise ValueError(f"{text!r} is not a square on this board")
    return r * N + c
