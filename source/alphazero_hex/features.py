"""Input planes for the Hex network.

The network used to see three planes -- own stones, opponent stones, and a
constant -- which meant every forward pass had to rediscover *connectivity*
from scratch.  Connectivity is the entire game: Hex is won by joining two
edges, and "is this group already touching my edge" is a global graph property
that a 3x3 convolution can only propagate one cell per layer.  Handing it to
the network as an input costs one wider stem convolution (1.7k -> 5.8k
parameters) and frees the whole trunk to reason about threats instead.

The planes, all in the *canonical* frame where the side to move owns the top
and bottom edges and the opponent owns left and right::

    0  own stones
    1  opponent stones
    2  empty cells
    3  constant 1 (lets zero-padded convolutions locate the border)
    4  own stones in a group touching the top edge
    5  own stones in a group touching the bottom edge
    6  opponent stones in a group touching the left edge
    7  opponent stones in a group touching the right edge
    8  empty cells that carry an own bridge
    9  empty cells that carry an opponent bridge

Everything here is derived from the board array alone, and everything is
computed for a whole *batch* at once.  Both properties are deliberate:

* Board-derivable means self-play (which has a live ``HexBoard``) and training
  (which has a stored ``uint8`` array) can call the same function, so the two
  cannot drift apart.  A silent mismatch there would poison a multi-day run
  and look like the network simply failing to learn.
* Batched means the fixed numpy overhead is paid once per evaluator batch
  (~20 positions) rather than once per position, which is the difference
  between ~1% and ~20% of self-play time.
"""

from __future__ import annotations

import numpy as np

LEGACY_PLANES = 3  # the original own/opp/ones encoding
PLANES = 10

# Hex adjacency on a rhombus; see hex_game for the geometry.  The set is
# symmetric (every offset's negation is also present), which is what lets the
# flood fill below propagate in both directions with a single sweep.
NEIGHBOURS = ((-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0))

# A bridge is two friendly stones that share exactly two empty neighbours: the
# pair is "virtually connected" because whichever carrier the opponent takes,
# the other still joins them.  Each entry is (partner offset, carrier, carrier).
# The six offsets come in +/- pairs, so every bridge is found from both ends.
BRIDGES = (
    ((-2, 1), (-1, 0), (-1, 1)),
    ((-1, 2), (-1, 1), (0, 1)),
    ((1, 1), (0, 1), (1, 0)),
    ((2, -1), (1, 0), (1, -1)),
    ((1, -2), (1, -1), (0, -1)),
    ((-1, -1), (0, -1), (-1, 0)),
)


def _shift(a: np.ndarray, dr: int, dc: int) -> np.ndarray:
    """``out[..., r, c] = a[..., r + dr, c + dc]``, zero outside the board."""
    out = np.zeros_like(a)
    n = a.shape[-1]
    rs0, rs1 = max(0, dr), min(n, n + dr)
    cs0, cs1 = max(0, dc), min(n, n + dc)
    if rs0 >= rs1 or cs0 >= cs1:
        return out
    out[..., rs0 - dr:rs1 - dr, cs0 - dc:cs1 - dc] = a[..., rs0:rs1, cs0:cs1]
    return out


def _reachable(mask: np.ndarray, seed: np.ndarray) -> np.ndarray:
    """Cells of ``mask`` reachable from ``seed`` while staying inside ``mask``.

    Iterated dilation rather than a union-find: a serpentine chain needs one
    round per cell of its length, but each round is six slice-copies over the
    whole batch, so a 60-stone 11x11 batch converges in well under a
    millisecond.
    """
    reach = mask & seed
    while True:
        grow = reach.copy()
        for dr, dc in NEIGHBOURS:
            grow |= _shift(reach, dr, dc)
        grow &= mask
        if np.array_equal(grow, reach):
            return reach
        reach = grow


def _bridge_carriers(stones: np.ndarray, empty: np.ndarray) -> np.ndarray:
    """Empty cells acting as a carrier for at least one bridge of ``stones``."""
    carriers = np.zeros_like(stones)
    for (dr, dc), (ar, ac), (br, bc) in BRIDGES:
        pair = (
            stones
            & _shift(stones, dr, dc)
            & _shift(empty, ar, ac)
            & _shift(empty, br, bc)
        )
        if not pair.any():
            continue
        # pair is indexed by the near endpoint; a carrier at offset (ar, ac)
        # from that endpoint lands at -(ar, ac) under _shift's convention.
        carriers |= _shift(pair, -ar, -ac)
        carriers |= _shift(pair, -br, -bc)
    return carriers & empty


def planes_from_boards(boards: np.ndarray, in_planes: int = PLANES) -> np.ndarray:
    """Encode canonical boards ``(B, n, n)`` as ``(B, in_planes, n, n)`` float32.

    ``boards`` uses 0 empty / 1 side-to-move / 2 opponent, already canonicalised
    (see :meth:`HexBoard.canonical_board`).  ``in_planes`` of 3 reproduces the
    original encoding so pre-feature checkpoints keep working unchanged.
    """
    boards = np.asarray(boards)
    if boards.ndim != 3:
        raise ValueError(f"expected (B, n, n) boards, got shape {boards.shape}")
    b, n, _ = boards.shape
    own = boards == 1
    opp = boards == 2

    out = np.zeros((b, in_planes, n, n), dtype=np.float32)
    out[:, 0] = own
    out[:, 1] = opp
    if in_planes == LEGACY_PLANES:
        out[:, 2] = 1.0
        return out
    if in_planes != PLANES:
        raise ValueError(f"in_planes must be {LEGACY_PLANES} or {PLANES}, got {in_planes}")

    empty = boards == 0
    out[:, 2] = empty
    out[:, 3] = 1.0

    # Connectivity.  In the canonical frame the side to move always joins top
    # to bottom and the opponent always joins left to right.
    top = np.zeros((b, n, n), dtype=bool)
    top[:, 0, :] = True
    bottom = np.zeros((b, n, n), dtype=bool)
    bottom[:, n - 1, :] = True
    left = np.zeros((b, n, n), dtype=bool)
    left[:, :, 0] = True
    right = np.zeros((b, n, n), dtype=bool)
    right[:, :, n - 1] = True

    out[:, 4] = _reachable(own, top)
    out[:, 5] = _reachable(own, bottom)
    out[:, 6] = _reachable(opp, left)
    out[:, 7] = _reachable(opp, right)

    out[:, 8] = _bridge_carriers(own, empty)
    out[:, 9] = _bridge_carriers(opp, empty)
    return out
