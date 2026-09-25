"""The trained agent: network + PUCT search (and a search-free ablation).

Neither depends on anything Hex-specific -- they see a position only through the
board protocol -- so both live in :mod:`alphazero_core.agents` and are
re-exported here under the names the tournament, registry and GUI already use.
"""

from __future__ import annotations

from alphazero_core.agents import AlphaZeroAgent, PolicyOnlyAgent  # noqa: F401

__all__ = ["AlphaZeroAgent", "PolicyOnlyAgent"]
