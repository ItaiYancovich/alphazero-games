"""Splendor's search: the shared max^n, chance-aware PUCT, re-exported.

Kept as its own module for the same reason ``alphazero_hex.mcts`` is: the game
packages are what the rest of the project imports, and which of them share an
implementation is an implementation detail.
"""

from __future__ import annotations

from alphazero_core.vecmcts import (BATCH, VecMCTSConfig,  # noqa: F401
                                    VecSearch, run_vec_search)

from .splendor_game import NACTIONS


def new_search(board, cfg: VecMCTSConfig, rng) -> VecSearch:
    """A search rooted at ``board``, sized to this game's action space."""
    return VecSearch(board, cfg, rng, NACTIONS)
