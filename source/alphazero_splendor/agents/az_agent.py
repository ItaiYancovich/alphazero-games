"""The trained Splendor agents: the shared vector search and policy head."""

from __future__ import annotations

from alphazero_core.vecagents import (VecMCTSAgent as AlphaZeroAgent,  # noqa: F401
                                      VecPolicyAgent as PolicyOnlyAgent)
