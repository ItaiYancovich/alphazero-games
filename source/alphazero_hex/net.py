"""Hex's view of the shared policy/value network.

The tower is game-independent and lives in :mod:`alphazero_core.net`; this
module re-exports it, keeping ``HexNet`` as the name under which every existing
checkpoint, script and test refers to it.  Checkpoints store a config dict and a
state dict, not a pickled class, so files written before the move load here
unchanged.
"""

from __future__ import annotations

from alphazero_core.net import (IN_PLANES, LEGACY_PLANES, GlobalPoolBlock,  # noqa: F401
                                HexNet, NetConfig, PVNet, ResBlock,
                                count_parameters, load_checkpoint, save_checkpoint)

__all__ = ["NetConfig", "HexNet", "PVNet", "ResBlock", "GlobalPoolBlock",
           "save_checkpoint", "load_checkpoint", "count_parameters",
           "IN_PLANES", "LEGACY_PLANES"]
