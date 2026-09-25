"""Common agent interface, shared with the other games.

Defined in :mod:`alphazero_core.agents` and re-exported here so that
Intransitive modules import it from their own package.
"""

from __future__ import annotations

from alphazero_core.agents import Agent, RandomAgent  # noqa: F401

__all__ = ["Agent", "RandomAgent"]
