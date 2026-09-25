"""Connect Four's view of the shared policy/value network.

The same fully convolutional tower Hex uses, unchanged: it is size-agnostic and
happily takes a 6x7 input, and the policy head emits one logit per cell, which
is exactly the per-cell policy this game's landing-square moves index into.

``NetConfig`` defaults (3 input planes, no global pooling, no auxiliary head)
are the plain AlphaZero baseline, which is what the Connect Four run trains.
"""

from __future__ import annotations

from alphazero_core.net import (IN_PLANES, LEGACY_PLANES, GlobalPoolBlock,  # noqa: F401
                                NetConfig, PVNet, ResBlock, count_parameters,
                                load_checkpoint, save_checkpoint)

# The name this package refers to the network by; the class itself is shared.
C4Net = PVNet

__all__ = ["NetConfig", "C4Net", "PVNet", "ResBlock", "GlobalPoolBlock",
           "save_checkpoint", "load_checkpoint", "count_parameters",
           "IN_PLANES", "LEGACY_PLANES"]
