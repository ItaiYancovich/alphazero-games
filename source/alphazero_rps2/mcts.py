"""Intransitive's view of the shared PUCT search.

The search is game-independent and lives in :mod:`alphazero_core.mcts`; this
re-export keeps every module in this package importing it from its own package.
Two of its features matter here rather than merely applying:

* **Draws.**  The no-capture rule makes ``DRAW`` a real proven state, as it is
  in Connect Four and Ultimate Tic-Tac-Toe and is not in Hex.
* **The value convention.**  A finished position is worth ``-1`` to the side to
  move -- somebody reached a base, took the last piece, or left this side with
  no move, and in all three cases the side on turn is the loser.  Ultimate XX
  is the game where that is not true, which is why it does not use this search.
"""

from __future__ import annotations

from alphazero_core.mcts import (BATCH, DRAW, LOSS, UNKNOWN, WIN,  # noqa: F401
                                 MCTSConfig, Search, run_search, solved_value)

__all__ = ["MCTSConfig", "Search", "run_search", "BATCH",
           "UNKNOWN", "WIN", "LOSS", "DRAW", "solved_value"]
