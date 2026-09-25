"""Opening-book keys that treat the eight symmetries of the board as one.

The 3x3-of-3x3 grid is invariant under every rotation and reflection of the
square: a rotation takes small boards to small boards and slots to slots.  A
book keyed by the exact layout therefore stores -- and, worse, *searches* --
mirror images separately.  Measured on the books built before this module:
39% of the first book's positions and 73% of the first-player book's were
copies of another, and after ``e5`` the opponent's eight replies are only two
distinct positions (four corners, four edges).

Here a position's key is the lexicographically smallest of its eight images,
read from the side to move (``UltimateBoard.canonical_board``: own / opponent /
playable / out of reach, which also pins down the forced board).  A book stores
its moves *in that canonical frame*; a lookup finds the symmetry that took the
live board to the canonical one and maps the stored move back through it.

Keys are prefixed ``s:`` so a book can hold them beside the old exact keys, and
a reader can tell the two apart.
"""

from __future__ import annotations

import numpy as np

from .features import transform_boards
from .uttt_game import N, UltimateBoard, str_to_move

PREFIX = "s:"
_IDX = np.arange(N * N).reshape(N, N)
# _TO[s][cell]: where ``cell`` lands under symmetry s; _FROM[s] undoes it.
_FROM = [transform_boards(_IDX, s).reshape(-1) for s in range(8)]
_TO = [np.argsort(f) for f in _FROM]


def sym_key(board: UltimateBoard) -> tuple[str, int]:
    """The canonical key of ``board`` and the symmetry that produces it."""
    cb = board.canonical_board()
    best, best_s = None, 0
    for s in range(8):
        k = "".join(map(str, transform_boards(cb, s).reshape(-1)))
        if best is None or k < best:
            best, best_s = k, s
    return PREFIX + best, best_s


def to_canon(sym: int, cell: int) -> int:
    """A move on the live board, expressed in the canonical frame."""
    return int(_TO[sym][cell])


def from_canon(sym: int, cell: int) -> int:
    """A move stored in the canonical frame, on the live board."""
    return int(_FROM[sym][cell])


def canon_entry(entry: dict, sym: int) -> dict:
    """``entry`` (moves on the live board) with its moves in the canonical frame."""
    out = dict(entry)
    out["move"] = to_canon(sym, int(entry["move"]))
    out["top"] = [dict(t, move=to_canon(sym, int(t["move"]))) for t in entry.get("top", [])]
    out["frame"] = "canonical"
    return out


def live_entry(entry: dict, sym: int) -> dict:
    """A canonical-frame ``entry`` with its moves mapped back onto the live board."""
    out = dict(entry)
    out["move"] = from_canon(sym, int(entry["move"]))
    out["top"] = [dict(t, move=from_canon(sym, int(t["move"]))) for t in entry.get("top", [])]
    return out


def lookup(book: dict, board: UltimateBoard) -> dict | None:
    """The book's entry for ``board``, with moves on the live board, or None.

    Canonical keys first; an old exact key (a book built before this module)
    is still honoured.
    """
    key, sym = sym_key(board)
    entry = book.get(key)
    if entry is not None:
        return live_entry(entry, sym)
    return book.get(str(board.key()))


def canonicalise_book(book: dict) -> dict:
    """Re-key a whole book canonically, keeping the deepest entry of each class.

    Every entry records the line that reached it, so its position can be
    rebuilt and re-keyed whatever format it was stored in.
    """
    out: dict[str, dict] = {}
    for key, entry in book.items():
        if key.startswith(PREFIX):
            cand_key, cand = key, entry
        else:
            b = UltimateBoard()
            for m in entry.get("line", "").split():
                b.play(str_to_move(m))
            if str(b.key()) != key:
                continue            # cannot be rebuilt; drop rather than guess
            cand_key, sym = sym_key(b)
            cand = canon_entry(entry, sym)
        if cand_key not in out or cand.get("sims", 0) > out[cand_key].get("sims", 0):
            out[cand_key] = cand
    return out
