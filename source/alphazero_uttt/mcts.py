"""Ultimate Tic-Tac-Toe's view of the shared PUCT search.

The search is game-independent and lives in :mod:`alphazero_core.mcts`; this
re-export keeps every module in this package importing it from its own package.
The draw handling in the solver (``DRAW`` as a proven state distinct from "not
proven") is needed here exactly as it is in Connect Four -- Hex is the odd one
out in not being able to draw.
"""

from __future__ import annotations

from alphazero_core.mcts import (BATCH, DRAW, LOSS, UNKNOWN, WIN,  # noqa: F401
                                 MCTSConfig, Search, run_search, solved_value)

__all__ = ["MCTSConfig", "Search", "run_search", "BATCH",
           "UNKNOWN", "WIN", "LOSS", "DRAW", "solved_value"]
