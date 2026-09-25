"""What the shared search needs a game position to provide.

This is a *documentation* protocol: nothing here is enforced at runtime (the
search only ever calls the methods below), but writing it down is what makes
"add another game" a bounded job rather than a reading exercise across four
modules.

Moves are plain ints, and the network sees positions in a *canonical* frame --
the frame in which the side to move is always player 1.  ``to_canonical_move``
is the map from a real move to its index in that frame; for Hex it transposes
the board when White is to move, for Connect Four it is the identity.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class GameState(Protocol):
    ncells: int  # number of cells, and the size of the canonical policy vector
    to_move: int
    winner: int  # 0 while running or drawn, else the winning player
    move_count: int

    def copy(self) -> "GameState": ...

    def legal_moves(self) -> np.ndarray: ...

    def is_legal(self, move: int) -> bool: ...

    def play(self, move: int) -> None: ...

    def is_terminal(self) -> bool: ...

    def terminal_value(self) -> float:
        """Value of a *finished* position for the side to move.

        ``-1`` when the move just played ended the game (whoever is on turn has
        lost) and ``0`` for a draw.  Hex can never draw and always returns -1;
        Connect Four returns 0 on a full board.
        """

    def canonical_board(self) -> np.ndarray:
        """``rows x cols`` uint8: 0 empty, 1 side to move, 2 opponent."""

    def to_canonical_move(self, move: int) -> int: ...

    def from_canonical_move(self, move: int) -> int: ...
