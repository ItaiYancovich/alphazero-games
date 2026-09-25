"""Common agent interface.

Defined game-independently in :mod:`alphazero_core.agents` and re-exported here,
which is where every Hex module imports it from.
"""

from __future__ import annotations

from alphazero_core.agents import Agent, RandomAgent  # noqa: F401

__all__ = ["Agent", "RandomAgent"]
