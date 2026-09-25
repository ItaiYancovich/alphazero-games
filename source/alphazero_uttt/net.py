"""Ultimate Tic-Tac-Toe's view of the shared policy/value network.

The same fully convolutional tower Hex and Connect Four use, unchanged: it is
size-agnostic and happily takes a 9x9 input, and the policy head emits one logit
per cell, which is exactly the per-cell policy this game's moves index into.

The one configuration choice worth stating is ``gp_every``.  Hex turned global
pooling on because connectivity is a whole-board property; the same argument
applies here for a different reason.  A 3x3 convolution moves information one
cell per layer, but which small board is open, and how the line of three across
the *upper* board is going, are facts about squares nine cells apart.  The
feature planes already broadcast the meta board, so pooling is a smaller win
here than in Hex -- but it is still the cheap way for one corner of the grid to
see the other, and ``train_uttt.py`` switches it on.
"""

from __future__ import annotations

from alphazero_core.net import (IN_PLANES, LEGACY_PLANES, GlobalPoolBlock,  # noqa: F401
                                NetConfig, PVNet, ResBlock, count_parameters,
                                load_checkpoint, save_checkpoint)

# The name this package refers to the network by; the class itself is shared.
UtttNet = PVNet

__all__ = ["NetConfig", "UtttNet", "PVNet", "ResBlock", "GlobalPoolBlock",
           "save_checkpoint", "load_checkpoint", "count_parameters",
           "IN_PLANES", "LEGACY_PLANES"]
