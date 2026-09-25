"""Common agent interface, shared with Hex and Connect Four.

Defined in :mod:`alphazero_core.agents` and re-exported here so that Ultimate
Tic-Tac-Toe modules import it from their own package.
"""

from __future__ import annotations

from alphazero_core.agents import Agent, RandomAgent  # noqa: F401

__all__ = ["Agent", "RandomAgent"]
