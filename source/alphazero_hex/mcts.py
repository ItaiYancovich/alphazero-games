"""Hex's view of the shared PUCT search.

The search itself is game-independent and lives in
:mod:`alphazero_core.mcts`; this module re-exports it under the name every Hex
script, test and the GUI already import.
"""

from __future__ import annotations

from alphazero_core.mcts import (DRAW, LOSS, UNKNOWN, WIN,  # noqa: F401
                                 MCTSConfig, Search, solved_value)

__all__ = ["MCTSConfig", "Search", "UNKNOWN", "WIN", "LOSS", "DRAW", "solved_value"]
