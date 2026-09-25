"""Input planes for the Connect Four network.

Three planes, in the *canonical* frame (the side to move is always player 1)::

    0  own discs
    1  opponent discs
    2  constant 1

This is the plain AlphaZero encoding, deliberately: unlike Hex -- where
connectivity is a global graph property a 3x3 convolution can only propagate one
cell per layer, and where handing it to the network as an input paid for itself
several times over -- a Connect Four threat is four cells in a line.  That *is*
a local pattern, so the trunk can see it directly and there is nothing worth
precomputing.  The constant plane is still worth its channel: zero-padded
convolutions cannot otherwise tell the floor of the board from empty space, and
the floor is where every disc lands.

The encoder is batched and takes boards, not positions, for the same reason the
Hex one does: self-play holds a live board and training holds a stored ``uint8``
array, and both must go through the same code or they will silently drift.
"""

from __future__ import annotations

import numpy as np

LEGACY_PLANES = 3
PLANES = 3


def planes_from_boards(boards: np.ndarray, in_planes: int = PLANES) -> np.ndarray:
    """``(N, rows, cols)`` canonical boards -> ``(N, planes, rows, cols)`` float32.

    ``boards`` uses the canonical encoding: 0 empty, 1 side to move, 2 opponent.
    """
    if in_planes != PLANES:
        raise ValueError(f"Connect Four uses {PLANES} input planes, not {in_planes}")
    boards = np.asarray(boards, dtype=np.uint8)
    if boards.ndim != 3:
        raise ValueError(f"expected (N, rows, cols), got {boards.shape}")
    n, rows, cols = boards.shape
    out = np.empty((n, PLANES, rows, cols), dtype=np.float32)
    out[:, 0] = boards == 1
    out[:, 1] = boards == 2
    out[:, 2] = 1.0
    return out


def mirror_boards(boards: np.ndarray) -> np.ndarray:
    """Left-right mirror -- Connect Four's only symmetry."""
    return np.ascontiguousarray(np.asarray(boards)[..., ::-1])
