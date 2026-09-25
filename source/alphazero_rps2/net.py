"""Intransitive's view of the shared policy/value network.

The same fully convolutional tower every other game here uses, with one
configuration field turned on: ``policy_planes=8``.

That field is the whole of what this game asks the network for.  Everywhere
else a move is a cell, so one logit per cell is the action space and the head
needs no notion of what a move *is*.  Here a move is a piece and one of eight
directions, and the natural way to say that convolutionally is eight output
channels over the same 9x9 grid: channel ``d`` at square ``s`` is the logit for
"the piece standing on ``s`` moves in direction ``d``".  Flattened, that is
exactly the ``direction * 81 + square`` encoding the engine uses, so the search
indexes straight into it.

Keeping it in the head rather than bolting a dense action head on the side is
what preserves the property the trunk is built around: a move stays a *local*
question.  The 3x3 window centred on a square already contains every square
that piece could move to, so one convolution can weigh "step there" against
"step there instead" -- which a flattened fully connected head would have to
learn from scratch, at eighty times the parameters.

``gp_every`` is on for the reason Hex has it.  The game is a race between two
corners nine squares apart, and whether a rush is faster than the defence
forming against it is a whole-board comparison that a stack of 3x3
convolutions propagates one square per layer.
"""

from __future__ import annotations

from alphazero_core.net import (IN_PLANES, LEGACY_PLANES, GlobalPoolBlock,  # noqa: F401
                                NetConfig, PVNet, ResBlock, count_parameters,
                                load_checkpoint, save_checkpoint)

# The name this package refers to the network by; the class itself is shared.
RPS2Net = PVNet

__all__ = ["NetConfig", "RPS2Net", "PVNet", "ResBlock", "GlobalPoolBlock",
           "save_checkpoint", "load_checkpoint", "count_parameters",
           "IN_PLANES", "LEGACY_PLANES"]
