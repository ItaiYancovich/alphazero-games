"""What the N-player search needs a position to provide.

The companion to :mod:`alphazero_core.state`, for the games that one does not
fit.  :class:`~alphazero_core.state.GameState` describes a two-player, perfect
information game on a grid: a move is a cell, a position is a board, and a value
is one number that flips sign every ply.  Splendor is none of those things, and
neither is most of the rest of board gaming, so this protocol asks for less:

* **Any number of seats.**  Seats are numbered ``1..players``; a result is a
  *vector* with one entry per seat, so nothing has to be zero-sum between
  exactly two people.  ``to_move`` is ``0`` in a finished position.
* **A fixed action space.**  Moves are indices into one list of ``nactions``
  actions that never changes; which of them are legal does.  That is what lets
  the policy head have a fixed width without the game having a board.
* **Chance and hidden information.**  ``determinize`` returns a position
  consistent with everything one seat knows, with everything it does not
  re-sampled.  A perfect-information game returns itself.

Nothing here is enforced at runtime -- the search only ever calls these methods
-- but writing it down is what makes the next game a bounded job.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class VecState(Protocol):
    players: int      # number of seats, 2 or more
    to_move: int      # 1..players while running, 0 once finished
    move_count: int
    winner: int       # 0 while running or drawn, else the winning seat

    def copy(self) -> "VecState": ...

    def is_terminal(self) -> bool: ...

    def legal_mask(self) -> np.ndarray:
        """``(nactions,) bool``: which actions are legal for ``to_move`` now."""

    def legal_moves(self) -> np.ndarray:
        """The indices of the legal actions -- ``flatnonzero(legal_mask())``."""

    def play(self, move: int) -> None: ...

    def result_vector(self) -> np.ndarray:
        """``(players + 1,)`` of scores in ``[-1, 1]``, index 0 unused.

        Zero everywhere while the game is running.  Index 0 is skipped so a
        seat number indexes directly, which is the convention every game in
        this repository uses for its players.
        """

    def determinize(self, seat: int, rng: np.random.Generator) -> "VecState":
        """A position ``seat`` could not tell from this one, freshly sampled.

        Deterministic, perfect-information games return ``self.copy()``.
        """


def rotate_to_absolute(values: np.ndarray, to_move: int, players: int) -> np.ndarray:
    """Seat-relative network output -> a seat-indexed vector.

    The network is fed a *seat-relative* view -- the player to move is always
    described first -- so its k-th value belongs to the seat ``k`` turns after
    the mover.  Returns ``(players + 1,)`` with index 0 unused.
    """
    out = np.zeros(players + 1, dtype=np.float32)
    for k in range(players):
        seat = (to_move - 1 + k) % players + 1
        out[seat] = values[k]
    return out


def rotate_to_relative(values: np.ndarray, to_move: int, players: int,
                       seats: int) -> np.ndarray:
    """The inverse: a seat-indexed vector -> the ``seats`` outputs the net trains on.

    Entries past ``players`` are zero -- a four-seat head fed a two-player game
    has two outputs with nothing to predict, and they are masked out of the loss
    rather than taught to output zero.
    """
    out = np.zeros(seats, dtype=np.float32)
    for k in range(min(players, seats)):
        seat = (to_move - 1 + k) % players + 1
        out[k] = values[seat]
    return out
