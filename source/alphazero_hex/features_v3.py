"""Input planes for the v3 network: v2's ten, plus what connection theory knows.

v2's planes say which stones already touch an edge and which cells carry a
bridge.  What they cannot say is *how far* anything is from connecting, which is
the question every Hex move is really answering.  The classical answer is
Anshelevich's **two-distance**: the number of stones still needed to join a
cell to an edge when the opponent always cuts the cheapest route, so a route
only counts where it has two independent ways through.  It is what the
rule-based and alpha-beta opponents here are built on, and handing it to the
network is the "knowledge as input, never as a leash" rule this project trains
by: the planes inform, and the zero-initialised stem weights of
:func:`alphazero_hex.net_v3.grow_from` let training use them only as far as
they actually help.

The planes, all in the canonical frame (the side to move joins top to bottom)::

    0-9   v2's planes (:func:`alphazero_hex.features.planes_from_boards`)
    10    swap available -- the only plane not derivable from the board
    11    own two-distance to the top edge      / n, capped at 1
    12    own two-distance to the bottom edge   / n, capped at 1
    13    opponent two-distance to the left edge  / n, capped at 1
    14    opponent two-distance to the right edge / n, capped at 1
    15    own connection cost through the cell, minus the board's cheapest, / n
    16    the same for the opponent

Planes 15-16 are zero exactly on the cells of each side's cheapest connection,
which is where the fight is.

Two-distance is an iterated relaxation, so it runs as a numba kernel over the
whole batch.  It reproduces :func:`alphazero_hex.heuristics.two_distance`
exactly, including its iteration cap, and ``tests/test_features_v3.py`` checks
that on random boards.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from .features import PLANES as V2_PLANES
from .features import planes_from_boards
from .heuristics import padded_neighbours

PLANES_V3 = V2_PLANES + 7
_INF = 1e6


@njit(cache=True)
def _two_distance_batch(flat, player, src, nei, n, out):
    """Two-distance for ``player`` from the edge cells in ``src``, per board.

    ``flat`` is ``(B, n*n)`` with 1 / 2 stones; ``nei`` is ``(n*n, 6)`` padded
    with the sentinel ``n*n``.  Jacobi iteration with the same cap and update
    rule as ``heuristics.two_distance``, so the fixpoint is identical.
    """
    b_count, cells = flat.shape
    opp = 2 if player == 1 else 1
    d = np.empty(cells + 1)
    new = np.empty(cells)
    for b in range(b_count):
        for i in range(cells + 1):
            d[i] = _INF
        for _ in range(2 * n + 4):
            changed = False
            for i in range(cells):
                stone = flat[b, i]
                if stone == opp:
                    new[i] = _INF
                    continue
                s = 0.0 if src[i] else _INF
                # smallest and second smallest of the six neighbours plus
                # the source edge counted twice (an edge cannot be cut)
                m1 = s
                m2 = s
                for k in range(6):
                    v = d[nei[i, k]]
                    if v < m1:
                        m2 = m1
                        m1 = v
                    elif v < m2:
                        m2 = v
                val = m1 if stone == player else m2 + 1.0
                if val > _INF:
                    val = _INF
                if val > d[i]:
                    val = d[i]
                new[i] = val
                if val != d[i]:
                    changed = True
            for i in range(cells):
                d[i] = new[i]
            if not changed:
                break
        for i in range(cells):
            out[b, i] = d[i]


_SRC_CACHE: dict[int, tuple[np.ndarray, ...]] = {}


def _sources(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cached = _SRC_CACHE.get(n)
    if cached is None:
        idx = np.arange(n * n)
        rows, cols = idx // n, idx % n
        cached = (rows == 0, rows == n - 1, cols == 0, cols == n - 1)
        _SRC_CACHE[n] = cached
    return cached


def two_distances(boards: np.ndarray) -> np.ndarray:
    """``(B, 4, n*n)`` raw two-distances: own top, own bottom, opp left, opp right."""
    b, n, _ = boards.shape
    flat = np.ascontiguousarray(boards.reshape(b, n * n), dtype=np.uint8)
    nei = padded_neighbours(n).astype(np.int64)
    top, bottom, left, right = _sources(n)
    out = np.empty((b, 4, n * n))
    for j, (player, src) in enumerate(((1, top), (1, bottom), (2, left), (2, right))):
        _two_distance_batch(flat, player, src, nei, n, out[:, j])
    return out


def planes_v3(boards: np.ndarray, swap_flags: np.ndarray | None = None) -> np.ndarray:
    """Encode canonical boards ``(B, n, n)`` as ``(B, 17, n, n)`` float32."""
    boards = np.asarray(boards)
    b, n, _ = boards.shape
    out = np.empty((b, PLANES_V3, n, n), dtype=np.float32)
    out[:, :V2_PLANES] = planes_from_boards(boards, V2_PLANES)
    out[:, 10] = 0.0 if swap_flags is None else \
        np.asarray(swap_flags, dtype=np.float32).reshape(b, 1, 1)

    d = two_distances(boards)                      # (B, 4, n*n)
    out[:, 11:15] = np.minimum(d / n, 1.0).reshape(b, 4, n, n)
    empty = (boards.reshape(b, n * n) == 0)
    for plane, (a, c) in ((15, (0, 1)), (16, (2, 3))):
        total = d[:, a] + d[:, c] - empty          # an empty cell is counted by both halves
        total = np.minimum(total, _INF)
        best = total.min(axis=1, keepdims=True)
        rel = np.where(best >= _INF, 1.0, np.minimum((total - best) / n, 1.0))
        out[:, plane] = rel.reshape(b, n, n)
    return out
