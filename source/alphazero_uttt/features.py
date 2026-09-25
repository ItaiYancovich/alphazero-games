"""Input planes for the Ultimate Tic-Tac-Toe network.

Twenty-three planes, in the *canonical* frame (the side to move is always
player 1)::

     0  own marks
     1  opponent marks
     2  constant 1
     3  playable cells -- the small board the last move sent us to

    the small board this cell sits IN (each value over its own nine cells):
     4  won by us
     5  won by the opponent
     6  full and drawn

    the small board this cell SENDS TO (each value over the cells that reach it):
     7  won by us
     8  won by the opponent
     9  full and drawn
    10  decided -- playing here hands the opponent a free move anywhere

    geometry (constant; the same for every position):
    11  this cell is the centre of its small board
    12  this cell is a corner of its small board
    13  this cell is in the centre small board
    14  this cell is in a corner small board
    15  this cell's block and slot coincide -- it can decide its own destination

    tactics:
    16  playing here completes a line for us in this small board
    17  the opponent would complete a line here in this small board
    18  playing here wins the game outright
    19  the board this cell sends to: the opponent has an immediate win waiting
    20  the board this cell sends to: and that win takes the game

    the meta board:
    21  winning the small board this cell is in completes a meta line for us
    22  winning the small board this cell is in completes a meta line for them

Planes 0-2 are the plain AlphaZero encoding.  Everything after them answers
something the marks alone do not say in any number of 3x3 convolutions:

* **Plane 3 is the game.**  Which small board you are confined to is the whole
  strategy of Ultimate Tic-Tac-Toe -- a move is chosen at least as much for
  where it *sends the opponent* as for what it takes -- and it is not a
  function of the marks: the same 81 squares are a different position depending
  on the last move played.

* **Planes 4-6 are the meta board**, a property of nine cells that the trunk
  would otherwise re-derive at every layer before it could reason about the
  line of three that actually decides the game.

* **Planes 7-10 are the destination**, and they are the ones this game cannot do
  without.  Planes 4-6 spread each small board's status over *its own* cells;
  these spread the same 3x3 grid over the cells that would *send the opponent
  there*.  The cell at board ``(br, bc)``, slot ``(sr, sc)`` sends the opponent
  to small board ``(sr, sc)``, so the two are the same numbers scattered
  differently -- ``np.tile`` against ``np.repeat`` -- and no 3x3 convolution
  turns one into the other: it is a permutation relating a cell to a block nine
  cells away.

* **Planes 11-15 are geometry, and they are free.**  The tower is 3x3
  convolutions on a 9x9 grid: a window centred on a slot centre covers exactly
  one small board, and a window one cell over straddles four of them.  Nothing
  in planes 0-10 says which case a given window is in -- the net can recover it
  by counting from the border through the zero padding, but that spends depth
  and capacity on something a constant mask gives away.  Planes 7-10 do carry
  slot-aligned structure, but only once boards start being decided; in the
  opening they are all zero, which is exactly when shape matters most.  These
  five are precomputed masks, identical for every position, so they cost nothing
  to encode.  Plane 15 also marks the nine cells where plane 10 is blind (below).

* **Planes 16-20 are the tactics a prior has to know.**  Search discovers all of
  these one ply later, but the prior is what decides where the search spends a
  budget of a few hundred simulations, and a prior that cannot see "this hands
  them an immediate win" pays to rediscover it at every node.  Plane 19 is the
  defining blunder of the game.

* **Planes 21-22 are why a small board matters.**  Not every board is worth
  winning; the ones on a live meta line are.

All twenty-three are functions of the canonical board array alone, which is what
keeps self-play and training honest: the board carries its own legality (see
:meth:`~alphazero_uttt.uttt_game.UltimateBoard.canonical_board`), so a stored
example and a live position encode identically.  They are also *equivariant*
under the eight symmetries -- rotations take small boards to small boards, slots
to matching slots, and lines to lines -- which is what lets the encoder sit
behind the augmentation rather than beside it.  ``tests/test_uttt.py`` checks
that on every plane and every symmetry.

Plane 10, and only plane 10, reads the board as it stands with no lookahead, so
it misses the one case where a move decides its own destination: a cell whose
block and slot indices coincide, played to win or fill that block.  That is 0.8%
of legal moves, measured, and every mismatch is that case.  Plane 15 marks those
nine cells, and planes 16-17 say whether a line is about to close there, so the
trunk has what it needs to cover the gap in one layer -- which is cheaper than
putting a per-cell lookahead in an encoder that runs on every batch of every
search.

The encoder is batched and takes boards, not positions, for the same reason the
Hex and Connect Four ones do: self-play holds a live board and training holds a
stored ``uint8`` array, and both must go through this code or they will
silently drift.
"""

from __future__ import annotations

import numpy as np

from .uttt_game import LINES, N, WON

PLANES = 23

# The seven-plane encoding this game was first trained with: marks, constant,
# constraint and the three meta-board planes.  Kept so a checkpoint saved before
# the rest existed still loads and plays -- which is what lets a run score
# itself against its own predecessor.
LEGACY_UTTT_PLANES = 7

# The eight symmetries of the square.  The 3x3-of-3x3 layout is invariant under
# all of them -- a rotation takes small boards to small boards and slots to the
# matching slots -- so unlike Connect Four, where gravity picks a direction,
# the full dihedral group is available for augmentation.
SYMMETRIES = 8

_WON = np.array(WON, dtype=bool)          # 512-entry "this mask holds a line"
_BIT = (1 << np.arange(9)).astype(np.int64)

# ------------------------------------------------------------------ geometry
# Row/column of each cell's slot within its small board, and of the small board
# within the grid.  Cell (r, c) is slot (r % 3, c % 3) of board (r // 3, c // 3).
_rows, _cols = np.divmod(np.arange(N * N), N)
_rows = _rows.reshape(N, N)
_cols = _cols.reshape(N, N)
_slot_r, _slot_c = _rows % 3, _cols % 3
_blk_r, _blk_c = _rows // 3, _cols // 3

SLOT_CENTRE = ((_slot_r == 1) & (_slot_c == 1)).astype(np.float32)
SLOT_CORNER = ((_slot_r != 1) & (_slot_c != 1)).astype(np.float32)
BOARD_CENTRE = ((_blk_r == 1) & (_blk_c == 1)).astype(np.float32)
BOARD_CORNER = ((_blk_r != 1) & (_blk_c != 1)).astype(np.float32)
# Block index == slot index: playing here sends the opponent to the very board
# the move was played in, so the move can decide its own destination.
SELF_SEND = ((_slot_r == _blk_r) & (_slot_c == _blk_c)).astype(np.float32)

# --------------------------------------------------------- line completion
# For each slot, the pairs that complete a line with it.  A slot lies on two,
# three or four lines, so the table is padded and carries a validity mask.
_completing: list[list[tuple[int, int]]] = [[] for _ in range(9)]
for _line in LINES:
    for _k in _line:
        _completing[_k].append(tuple(x for x in _line if x != _k))
_MAX_LINES = max(len(p) for p in _completing)
COMPLETING_PAIRS = np.zeros((9, _MAX_LINES, 2), dtype=np.int64)
COMPLETING_VALID = np.zeros((9, _MAX_LINES), dtype=bool)
for _k, _pairs in enumerate(_completing):
    for _i, _pair in enumerate(_pairs):
        COMPLETING_PAIRS[_k, _i] = _pair
        COMPLETING_VALID[_k, _i] = True


def planes_from_boards(boards: np.ndarray, in_planes: int = PLANES) -> np.ndarray:
    """``(N, 9, 9)`` canonical boards -> ``(N, planes, 9, 9)`` float32.

    ``boards`` uses the canonical encoding: 0 empty and playable, 1 side to
    move, 2 opponent, 3 empty but not playable this turn.
    """
    if in_planes not in (PLANES, LEGACY_UTTT_PLANES):
        raise ValueError(f"Ultimate Tic-Tac-Toe uses {PLANES} input planes "
                         f"(or {LEGACY_UTTT_PLANES} for a legacy checkpoint), "
                         f"not {in_planes}")
    boards = np.asarray(boards, dtype=np.uint8)
    if boards.ndim != 3 or boards.shape[1:] != (N, N):
        raise ValueError(f"expected (N, {N}, {N}), got {boards.shape}")
    n = boards.shape[0]
    own = boards == 1
    opp = boards == 2
    empty = ~(own | opp)

    own_cells = _by_board(own, n)
    opp_cells = _by_board(opp, n)
    empty_cells = _by_board(empty, n)
    own_board = _has_line(own_cells)
    opp_board = _has_line(opp_cells)
    # "Full" counts marks only: an empty cell reads 0 or 3 depending on whether
    # it is reachable, and neither is a mark.
    full = (own_cells | opp_cells).all(axis=-1)
    drawn_board = full & ~own_board & ~opp_board
    closed_board = own_board | opp_board | drawn_board

    out = np.empty((n, in_planes, N, N), dtype=np.float32)
    out[:, 0] = own
    out[:, 1] = opp
    out[:, 2] = 1.0
    out[:, 3] = boards == 0
    out[:, 4] = _spread(own_board)
    out[:, 5] = _spread(opp_board)
    out[:, 6] = _spread(drawn_board)
    if in_planes == LEGACY_UTTT_PLANES:
        return out

    out[:, 7] = _send(own_board)
    out[:, 8] = _send(opp_board)
    out[:, 9] = _send(drawn_board)
    out[:, 10] = _send(closed_board)

    out[:, 11] = SLOT_CENTRE
    out[:, 12] = SLOT_CORNER
    out[:, 13] = BOARD_CENTRE
    out[:, 14] = BOARD_CORNER
    out[:, 15] = SELF_SEND

    # Which empty slots would complete a line, for each side.  Only open boards
    # count: a line inside a board that is already won or full changes nothing.
    open_board = ~closed_board
    mine_here = _completes(own_cells) & empty_cells & open_board[..., None]
    theirs_here = _completes(opp_cells) & empty_cells & open_board[..., None]

    # Meta lines.  ``WON[meta | bit(j)]`` is "owning j as well would finish it",
    # which is what makes a particular small board worth contesting.
    own_meta = (own_board.reshape(n, 9) * _BIT).sum(1)
    opp_meta = (opp_board.reshape(n, 9) * _BIT).sum(1)
    meta_needs_me = _WON[own_meta[:, None] | _BIT[None, :]].reshape(n, 3, 3)
    meta_needs_opp = _WON[opp_meta[:, None] | _BIT[None, :]].reshape(n, 3, 3)

    out[:, 16] = _to_grid(mine_here, n)
    out[:, 17] = _to_grid(theirs_here, n)
    # Completing a small board only wins the game if that board is on a meta
    # line we would then hold.
    out[:, 18] = out[:, 16] * _spread(meta_needs_me)
    # An immediate win waiting for the opponent in the board we would send them
    # to.  ``_send`` rather than ``_spread``: this is a fact about the
    # destination, so it belongs on the cells that reach it.
    opp_win_waiting = theirs_here.any(axis=-1)
    out[:, 19] = _send(opp_win_waiting)
    out[:, 20] = _send(opp_win_waiting & meta_needs_opp)
    out[:, 21] = _spread(meta_needs_me)
    out[:, 22] = _spread(meta_needs_opp)
    return out


def _by_board(flags: np.ndarray, n: int) -> np.ndarray:
    """``(n, 9, 9)`` -> ``(n, 3, 3, 9)``, regrouped as (board row, col, slot).

    The reshape splits rows into (board row, slot row) and columns into
    (board col, slot col); the transpose brings the two board axes together
    ahead of the two slot axes.
    """
    return (flags.reshape(n, 3, 3, 3, 3).transpose(0, 1, 3, 2, 4)
            .reshape(n, 3, 3, 9))


def _to_grid(per_slot: np.ndarray, n: int) -> np.ndarray:
    """``(n, 3, 3, 9)`` -> ``(n, 9, 9)``: the inverse of :func:`_by_board`.

    The same axis swap, which is its own inverse.
    """
    return (per_slot.reshape(n, 3, 3, 3, 3).transpose(0, 1, 3, 2, 4)
            .reshape(n, N, N))


def _has_line(cells: np.ndarray) -> np.ndarray:
    """``(..., 9)`` occupancy flags -> ``(...)`` "these hold a line of three"."""
    out = np.zeros(cells.shape[:-1], dtype=bool)
    for a, b, c in LINES:
        out |= cells[..., a] & cells[..., b] & cells[..., c]
    return out


def _completes(cells: np.ndarray) -> np.ndarray:
    """``(..., 9)`` occupancy -> ``(..., 9)`` "a mark here finishes a line".

    Says nothing about whether the slot is free; the caller intersects with the
    empties, because "the opponent would complete a line on that square" is only
    interesting while the square is still there to take.
    """
    a = cells[..., COMPLETING_PAIRS[:, :, 0]]
    b = cells[..., COMPLETING_PAIRS[:, :, 1]]
    return (a & b & COMPLETING_VALID).any(axis=-1)


def _spread(per_board: np.ndarray) -> np.ndarray:
    """``(N, 3, 3)`` per small board -> ``(N, 9, 9)``, each value over its cells.

    The cell at board ``(br, bc)``, slot ``(sr, sc)`` -- grid position
    ``(3*br + sr, 3*bc + sc)`` -- receives ``per_board[br, bc]``: the status of
    the board it sits in.
    """
    return np.repeat(np.repeat(per_board, 3, axis=-2), 3, axis=-1)


def _send(per_board: np.ndarray) -> np.ndarray:
    """``(N, 3, 3)`` per small board -> ``(N, 9, 9)``, by *destination*.

    The same numbers as :func:`_spread` and a different scatter: the cell at
    board ``(br, bc)``, slot ``(sr, sc)`` receives ``per_board[sr, sc]``, the
    status of the board that playing there sends the opponent to.  Indexing by
    the slot rather than by the block is exactly ``tile`` rather than
    ``repeat``, since ``(3*br + sr) % 3 == sr``.
    """
    reps = (1,) * (per_board.ndim - 2) + (3, 3)
    return np.tile(per_board, reps)


def transform_boards(a: np.ndarray, sym: int) -> np.ndarray:
    """One of the eight symmetries of the square, applied to the last two axes.

    ``sym`` is ``rotations + 4 * flipped``: rotate by ``sym % 4`` quarter turns,
    then mirror left-right if ``sym >= 4``.  Works on boards and on policies
    alike -- a policy is a 9x9 grid of one number per cell, so the same
    geometric map moves both, and no per-move index arithmetic is needed.
    """
    out = np.rot90(np.asarray(a), sym % 4, axes=(-2, -1))
    if sym >= 4:
        out = out[..., ::-1]
    return np.ascontiguousarray(out)
