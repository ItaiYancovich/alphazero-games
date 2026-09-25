"""The trained agent: network + PUCT search (and a search-free ablation).

Both are game-independent -- they see a position only through the board
protocol -- so they live in :mod:`alphazero_core.agents` and are re-exported
here under the names the registry, tournament and GUI use.
"""

from __future__ import annotations

from alphazero_core.agents import AlphaZeroAgent, PolicyOnlyAgent  # noqa: F401

__all__ = ["AlphaZeroAgent", "PolicyOnlyAgent"]
